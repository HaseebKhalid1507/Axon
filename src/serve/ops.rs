//! `axon serve`'s operational surface: `--api-key` (Bearer auth), `GET
//! /metrics` (Prometheus text), `GET /slots` (what each slot holds), and the
//! stalled-client check behind `--slot-idle-ms`.
//!
//! Nothing here touches the model, the caches or sampling: the counters are
//! fed by the connection threads (requests, their usage, token timings) and
//! by one snapshot the engine thread publishes per loop iteration (the
//! slots), so the output bits are what they were without it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::net::TcpStream;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Whether `headers` carry `Authorization: Bearer <key>`. The token is
/// compared in constant time (for a given key length).
pub fn authorized(headers: &[(String, String)], key: &str) -> bool {
    let Some(v) = headers
        .iter()
        .find(|(n, _)| n == "authorization")
        .map(|(_, v)| v.as_str())
    else {
        return false;
    };
    let Some((scheme, token)) = v.split_once(' ') else {
        return false;
    };
    let token = token.trim();
    // An empty key (a library caller's Opts; the CLI refuses it) never
    // authenticates an empty token.
    !token.is_empty()
        && !key.is_empty()
        && scheme.eq_ignore_ascii_case("bearer")
        && ct_eq(token.as_bytes(), key.as_bytes())
}

/// The 401's message.
pub const AUTH_MSG: &str = "a valid API key is needed: send 'Authorization: Bearer <key>'";

/// `a == b`, in time that depends only on the lengths.
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let mut d = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0);
        d |= std::hint::black_box(x) as usize;
    }
    d == 0
}

/// The routes `/metrics` labels requests with (anything else: "other", so
/// a scanner can't grow the label set).
pub const ROUTES: &[&str] = &[
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/models",
    "/health",
    "/identity",
    "/metrics",
    "/slots",
    "/v1/responses",
    "/api/chat",
    "/api/generate",
    "/api/tags",
    "/api/show",
    "/api/ps",
    "/api/version",
];

pub fn route_label(path: &str) -> &'static str {
    ROUTES
        .iter()
        .find(|r| **r == path)
        .copied()
        .unwrap_or("other")
}

/// Time to first token, seconds.
const TTFT_BOUNDS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0,
];
/// Time between a request's text events (about a token each), seconds.
const ITL_BOUNDS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56,
];

/// A batched step that fed prompt beside running rows (what `--step-ms`
/// aims at), seconds: 5 ms apart to 60, then coarser.
const STEP_BOUNDS: &[f64] = &[
    0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.07, 0.08, 0.1,
    0.13, 0.16, 0.2, 0.32, 0.64,
];

/// A Prometheus histogram over fixed bounds.
pub struct Histogram {
    bounds: &'static [f64],
    /// Per bucket (not cumulative), the last one +Inf.
    counts: Vec<AtomicU64>,
    sum_ns: AtomicU64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Histogram {
        Histogram {
            bounds,
            counts: (0..=bounds.len()).map(|_| AtomicU64::new(0)).collect(),
            sum_ns: AtomicU64::new(0),
        }
    }

    pub fn observe(&self, d: Duration) {
        let s = d.as_secs_f64();
        let i = self
            .bounds
            .iter()
            .position(|&b| s <= b)
            .unwrap_or(self.bounds.len());
        self.counts[i].fetch_add(1, Ordering::Relaxed);
        self.sum_ns
            .fetch_add(d.as_nanos().min(u64::MAX as u128) as u64, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, help: &str) {
        let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} histogram");
        let mut acc = 0;
        for (i, c) in self.counts.iter().enumerate() {
            acc += c.load(Ordering::Relaxed);
            match self.bounds.get(i) {
                Some(b) => {
                    let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {acc}");
                }
                None => {
                    let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {acc}");
                }
            }
        }
        let sum = self.sum_ns.load(Ordering::Relaxed) as f64 / 1e9;
        let _ = writeln!(out, "{name}_sum {sum}\n{name}_count {acc}");
    }
}

/// What a slot (or the single-request cache) holds.
#[derive(Clone, Debug, Default)]
pub struct SlotState {
    /// Tokens whose rows it holds, from position 0 (a prefix a later
    /// request can reuse).
    pub held: usize,
    pub busy: bool,
    /// The request in it: its number (the `#n` of the log) and arrival.
    pub request: Option<(u64, Instant)>,
    /// KV pages it holds (a paged pool).
    pub pages: Option<usize>,
}

impl SlotState {
    fn json(&self, now: Instant) -> serde_json::Value {
        let mut j = serde_json::json!({
            "busy": self.busy,
            "held_tokens": self.held,
            "request_id": self.request.map(|r| r.0),
            "age_ms": self.request.map(|r| now.saturating_duration_since(r.1).as_millis() as u64),
        });
        if let Some(p) = self.pages {
            j["pages_used"] = p.into();
        }
        j
    }
}

/// The engine's last published slot states.
#[derive(Default)]
pub struct SlotsView {
    /// The slots: the batch's pool, or (one request at a time) the one
    /// cache as slot 0 (empty until a request ran: a free slot 0). A
    /// request running alone (`Engine::job`) runs in one of them: with a
    /// pool its cache is a view of a slot's rows (`Batch::lend_solo`).
    pub pool: Vec<SlotState>,
    /// A paged pool's KV pages and what gave way for them.
    pub pages: Option<PagesView>,
}

/// A paged pool's pages for GET /slots.
#[derive(Clone, Copy, Debug, Default)]
pub struct PagesView {
    pub free: usize,
    pub total: usize,
    /// Positions per page.
    pub size: usize,
    /// Requests preempted for pages so far.
    pub preempted: u64,
    /// Prompts being prefilled that gave their rows up so far.
    pub unfed: u64,
}

impl SlotsView {
    /// Slots in total, and busy.
    pub fn counts(&self) -> (usize, usize) {
        let total = self.pool.len().max(1);
        let busy = self.pool.iter().filter(|s| s.busy).count();
        (total, busy.min(total))
    }
}

/// The scheduler's state for /metrics and GET /slots (`Batch::publish`,
/// the engine loop's waiting requests).
#[derive(Clone, Debug, Default)]
pub struct SchedView {
    /// Requests the engine took off the queue that wait for a slot: id,
    /// prompt tokens, arrival.
    pub waiting: Vec<(u64, usize, Instant)>,
    /// Long prompts prefilling, and the most at once (`Batch::lanes`).
    pub lanes_busy: usize,
    pub lanes: usize,
    /// The step cost model (`engine::StepCost`): µs per prompt token per
    /// (depth + 8192), a pass's fixed µs, decode-only step ms per row
    /// bucket (1, 2, 3-4, ..., 129-256).
    pub tok_us: Option<f64>,
    pub pass_us: Option<f64>,
    pub dec_ms: Vec<Option<f64>>,
    /// AXON_STEP_COST=legacy (the rollback's budgets).
    pub legacy: bool,
}

/// Counters the connection threads and the engine feed.
pub struct Metrics {
    requests: Mutex<BTreeMap<(&'static str, u16), u64>>,
    pub prompt_tokens: AtomicU64,
    pub cached_tokens: AtomicU64,
    /// Completion requests handed to the engine and not yet answered.
    pub in_flight: AtomicUsize,
    /// Streams cancelled by `--slot-idle-ms`.
    pub idle_evictions: AtomicU64,
    /// Prompts refused by `--max-prompt-tokens`.
    pub refused_prompts: AtomicU64,
    /// gpt-oss tool calls begun but not finished as a JSON object
    /// (returned as reasoning).
    pub tool_calls_dropped: AtomicU64,
    pub ttft: Histogram,
    pub itl: Histogram,
    /// Batched steps that fed prompt beside running rows, and those with
    /// at least 16 rows.
    pub step: Histogram,
    pub step16: Histogram,
    /// The last such step's prompt tokens.
    pub step_prompt_tokens: AtomicUsize,
    pub slots: Mutex<SlotsView>,
    pub sched: Mutex<SchedView>,
    /// The KV RAM tier (`--cache-ram`; 0: off, and none of these shown):
    /// its size, bytes and pages stored, pages written (D2H), prompt tokens
    /// restored instead of prefilled, and the requests that had some.
    pub kv_ram_cap: AtomicU64,
    pub kv_ram_bytes: AtomicU64,
    pub kv_ram_pages: AtomicU64,
    pub kv_ram_written: AtomicU64,
    pub kv_ram_restored: AtomicU64,
    pub kv_ram_hits: AtomicU64,
    /// Page-locked host memory the tier has taken (a pool's; the CPU's
    /// pages are ordinary memory: 0).
    pub kv_ram_pinned: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Metrics {
        Metrics {
            requests: Mutex::default(),
            prompt_tokens: AtomicU64::new(0),
            cached_tokens: AtomicU64::new(0),
            in_flight: AtomicUsize::new(0),
            idle_evictions: AtomicU64::new(0),
            refused_prompts: AtomicU64::new(0),
            tool_calls_dropped: AtomicU64::new(0),
            ttft: Histogram::new(TTFT_BOUNDS),
            itl: Histogram::new(ITL_BOUNDS),
            step: Histogram::new(STEP_BOUNDS),
            step16: Histogram::new(STEP_BOUNDS),
            step_prompt_tokens: AtomicUsize::new(0),
            slots: Mutex::default(),
            kv_ram_cap: AtomicU64::new(0),
            kv_ram_bytes: AtomicU64::new(0),
            kv_ram_pages: AtomicU64::new(0),
            kv_ram_written: AtomicU64::new(0),
            kv_ram_restored: AtomicU64::new(0),
            kv_ram_hits: AtomicU64::new(0),
            kv_ram_pinned: AtomicU64::new(0),
            sched: Mutex::default(),
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Metrics {
    pub fn request(&self, route: &'static str, status: u16) {
        *lock(&self.requests).entry((route, status)).or_insert(0) += 1;
    }

    pub fn publish_pool(&self, pool: Vec<SlotState>, pages: Option<PagesView>) {
        let mut v = lock(&self.slots);
        v.pool = pool;
        v.pages = pages;
    }

    /// A batched step of `rows` rows that fed `prompt` prompt tokens took `d`.
    pub fn observe_step(&self, d: Duration, rows: usize, prompt: usize) {
        self.step.observe(d);
        if rows >= 16 {
            self.step16.observe(d);
        }
        self.step_prompt_tokens.store(prompt, Ordering::Relaxed);
    }

    /// The engine's scheduler state (its waiting list kept: `publish_waiting`).
    pub fn publish_sched(&self, v: SchedView) {
        let mut s = lock(&self.sched);
        let waiting = std::mem::take(&mut s.waiting);
        *s = SchedView { waiting, ..v };
    }

    /// The requests the engine loop holds, waiting for a slot.
    pub fn publish_waiting(&self, w: Vec<(u64, usize, Instant)>) {
        lock(&self.sched).waiting = w;
    }

    /// Slot `i`'s state: where a request running alone runs.
    pub fn publish_slot(&self, i: usize, s: SlotState) {
        let mut v = lock(&self.slots);
        if v.pool.len() <= i {
            v.pool.resize(i + 1, SlotState::default());
        }
        // A request alone doesn't know the pages: the pool's last word.
        let pages = s.pages.or(v.pool[i].pages);
        v.pool[i] = SlotState { pages, ..s };
    }

    /// Requests waiting for a slot: those in flight less those running.
    fn queue_depth(&self, busy: usize) -> usize {
        self.in_flight.load(Ordering::Relaxed).saturating_sub(busy)
    }

    /// GET /slots.
    pub fn slots_json(&self, ctx: usize) -> String {
        let now = Instant::now();
        let v = lock(&self.slots);
        let (_, busy) = v.counts();
        let one = [SlotState::default()];
        let slots = if v.pool.is_empty() { &one[..] } else { &v.pool };
        let pool: Vec<_> = slots
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut j = s.json(now);
                j["id"] = i.into();
                j
            })
            .collect();
        let waiting: Vec<_> = (lock(&self.sched).waiting.iter())
            .map(|&(id, toks, t)| {
                serde_json::json!({"id": id, "prompt_tokens": toks,
                    "age_ms": now.saturating_duration_since(t).as_millis() as u64})
            })
            .collect();
        let mut j = serde_json::json!({
            "ctx": ctx,
            "slots": pool,
            "queue_depth": self.queue_depth(busy),
            "waiting": waiting,
        });
        // Graphs given up / steps run kernel by kernel for lack of VRAM
        // (process-wide: never fatal, the same bits).
        let (evicted, eager) = crate::cuda::qwen2::graph_oom_counts();
        j["graphs_evicted_oom"] = evicted.into();
        j["steps_eager_oom"] = eager.into();
        if let Some(p) = v.pages {
            j["pages_free"] = p.free.into();
            j["pages_total"] = p.total.into();
            j["page_tokens"] = p.size.into();
            j["preempted"] = p.preempted.into();
            j["prefills_unfed"] = p.unfed.into();
        }
        j.to_string()
    }

    /// GET /metrics: Prometheus text format 0.0.4.
    pub fn render(&self, generated: u64) -> String {
        let mut o = String::new();
        let counter = |o: &mut String, name: &str, help: &str, v: u64| {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} counter\n{name} {v}");
        };
        let gauge = |o: &mut String, name: &str, help: &str, v: usize| {
            let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
        };
        let _ = writeln!(
            o,
            "# HELP axon_requests_total HTTP requests answered, by route and status.\n\
             # TYPE axon_requests_total counter"
        );
        for ((r, s), n) in lock(&self.requests).iter() {
            let _ = writeln!(o, "axon_requests_total{{route=\"{r}\",status=\"{s}\"}} {n}");
        }
        counter(
            &mut o,
            "axon_generated_tokens_total",
            "Tokens generated, all requests.",
            generated,
        );
        counter(
            &mut o,
            "axon_prompt_tokens_total",
            "Prompt tokens of completed requests.",
            self.prompt_tokens.load(Ordering::Relaxed),
        );
        counter(
            &mut o,
            "axon_prompt_cached_tokens_total",
            "Prompt tokens whose KV rows were reused (prefix cache).",
            self.cached_tokens.load(Ordering::Relaxed),
        );
        counter(
            &mut o,
            "axon_slot_idle_evictions_total",
            "Streams cancelled because the client stopped reading (--slot-idle-ms).",
            self.idle_evictions.load(Ordering::Relaxed),
        );
        counter(
            &mut o,
            "axon_prompt_refused_total",
            "Prompts refused by --max-prompt-tokens.",
            self.refused_prompts.load(Ordering::Relaxed),
        );
        counter(
            &mut o,
            "axon_tool_calls_dropped_total",
            "gpt-oss tool calls begun but not finished as a JSON object (returned as reasoning).",
            self.tool_calls_dropped.load(Ordering::Relaxed),
        );
        if self.kv_ram_cap.load(Ordering::Relaxed) > 0 {
            let g = |o: &mut String, name: &str, help: &str, v: &AtomicU64| {
                let v = v.load(Ordering::Relaxed);
                let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
            };
            g(
                &mut o,
                "axon_kv_ram_capacity_bytes",
                "The KV RAM tier's size (--cache-ram).",
                &self.kv_ram_cap,
            );
            g(
                &mut o,
                "axon_kv_ram_bytes",
                "KV page bytes stored in the RAM tier.",
                &self.kv_ram_bytes,
            );
            g(
                &mut o,
                "axon_kv_ram_pinned_bytes",
                "Page-locked host memory the RAM tier has taken (it grows as pages are stored).",
                &self.kv_ram_pinned,
            );
            g(
                &mut o,
                "axon_kv_ram_pages",
                "KV pages stored in the RAM tier.",
                &self.kv_ram_pages,
            );
            let c = |o: &mut String, name: &str, help: &str, v: &AtomicU64| {
                counter(o, name, help, v.load(Ordering::Relaxed))
            };
            c(
                &mut o,
                "axon_kv_ram_written_pages_total",
                "KV pages copied into the RAM tier.",
                &self.kv_ram_written,
            );
            c(
                &mut o,
                "axon_kv_ram_restored_tokens_total",
                "Prompt tokens whose KV rows were copied back from the RAM tier instead of prefilled.",
                &self.kv_ram_restored,
            );
            c(
                &mut o,
                "axon_kv_ram_hits_total",
                "Requests that had KV rows copied back from the RAM tier.",
                &self.kv_ram_hits,
            );
        }
        let (total, busy) = lock(&self.slots).counts();
        gauge(
            &mut o,
            "axon_requests_in_flight",
            "Completion requests queued or running.",
            self.in_flight.load(Ordering::Relaxed),
        );
        gauge(
            &mut o,
            "axon_queue_depth",
            "Completion requests waiting for a slot.",
            self.queue_depth(busy),
        );
        gauge(&mut o, "axon_slots_busy", "Slots running a request.", busy);
        gauge(&mut o, "axon_slots_free", "Slots free.", total - busy);
        self.ttft.render(
            &mut o,
            "axon_ttft_seconds",
            "Time from a request's arrival (its body read) to its first text.",
        );
        self.itl.render(
            &mut o,
            "axon_itl_seconds",
            "Time between a request's text events (about one token each).",
        );
        self.step.render(
            &mut o,
            "axon_step_seconds",
            "Batched steps that fed prompt beside running rows (--step-ms aims at it).",
        );
        self.step16.render(
            &mut o,
            "axon_step_seconds_rows16",
            "axon_step_seconds, steps of at least 16 rows.",
        );
        gauge(
            &mut o,
            "axon_step_prompt_tokens",
            "Prompt tokens of the last batched step that fed prompt beside running rows.",
            self.step_prompt_tokens.load(Ordering::Relaxed),
        );
        let s = lock(&self.sched).clone();
        gauge(
            &mut o,
            "axon_waiting_requests",
            "Requests the engine holds, waiting for a slot.",
            s.waiting.len(),
        );
        gauge(
            &mut o,
            "axon_prefill_lanes_busy",
            "Long prompts prefilling (--prefill-lanes).",
            s.lanes_busy,
        );
        gauge(
            &mut o,
            "axon_step_cost_legacy",
            "1: AXON_STEP_COST=legacy (the budgets before the step cost model).",
            s.legacy as usize,
        );
        gauge(
            &mut o,
            "axon_prefill_lanes",
            "Long prompts prefilling at once at most.",
            s.lanes,
        );
        let fgauge = |o: &mut String, name: &str, help: &str, v: Option<f64>| {
            if let Some(v) = v {
                let _ = writeln!(o, "# HELP {name} {help}\n# TYPE {name} gauge\n{name} {v}");
            }
        };
        fgauge(
            &mut o,
            "axon_step_cost_tok_us",
            "Step cost model: us per prompt token per (depth + 8192), two-pass.",
            s.tok_us,
        );
        fgauge(
            &mut o,
            "axon_step_cost_pass_us",
            "Step cost model: a pass's fixed us.",
            s.pass_us,
        );
        if s.dec_ms.iter().any(Option::is_some) {
            let _ = writeln!(
                o,
                "# HELP axon_step_cost_dec_ms Step cost model: decode-only step ms by rows bucket (upper bound).\n\
                 # TYPE axon_step_cost_dec_ms gauge"
            );
            for (k, v) in s.dec_ms.iter().enumerate() {
                if let Some(v) = v {
                    let _ = writeln!(
                        o,
                        "axon_step_cost_dec_ms{{rows_le=\"{}\"}} {v}",
                        1usize << k
                    );
                }
            }
        }
        o
    }
}

/// Whether `unacked` can read anything here: Linux SIOCOUTQ (= TIOCOUTQ),
/// 0x5411 on the asm-generic ioctl numbering. mips, powerpc, sparc and
/// alpha number it differently, so they are left out (None, as elsewhere).
pub const OUTQ_PROBE: bool = cfg!(all(
    target_os = "linux",
    any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv64",
        target_arch = "s390x",
        target_arch = "loongarch64"
    )
));

/// `unacked`, with one warning on stderr (if `log`: not --quiet) the first
/// time it can't be read (only a blocked write is detected then).
pub fn unacked_or_warn(s: &TcpStream, log: bool) -> Option<u64> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let u = unacked(s);
    if u.is_none() && OUTQ_PROBE && log && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "axon serve: --slot-idle-ms: can't read a socket's unacknowledged bytes (SIOCOUTQ); \
             only a write blocked for the idle time is detected"
        );
    }
    u
}

/// Bytes written to `s` the peer hasn't acknowledged yet (Linux SIOCOUTQ),
/// None where that can't be read (`OUTQ_PROBE`: the arches where the
/// number is right; the call is compiled on every Linux but never made on
/// the others).
#[cfg(target_os = "linux")]
pub fn unacked(s: &TcpStream) -> Option<u64> {
    use std::os::fd::AsRawFd;
    if !OUTQ_PROBE {
        return None;
    }
    unsafe extern "C" {
        fn ioctl(fd: std::ffi::c_int, req: std::ffi::c_ulong, ...) -> std::ffi::c_int;
    }
    const SIOCOUTQ: std::ffi::c_ulong = 0x5411;
    let mut n: std::ffi::c_int = 0;
    // SAFETY: SIOCOUTQ writes one int through the pointer, on an fd we own.
    let r = unsafe { ioctl(s.as_raw_fd(), SIOCOUTQ, &mut n as *mut std::ffi::c_int) };
    (r == 0 && n >= 0).then_some(n as u64)
}

#[cfg(not(target_os = "linux"))]
pub fn unacked(_: &TcpStream) -> Option<u64> {
    None
}

/// `--slot-idle-ms`: a stream whose client has stopped reading. Bytes it
/// was sent wait unacknowledged once its receive window is full; when the
/// acknowledged count hasn't moved for `idle` while bytes wait, the client
/// is stalled (a slow but reading client keeps it moving). A write that
/// blocks for `idle` is the other half (the socket's write timeout).
pub struct Stall {
    idle: Duration,
    acked: u64,
    since: Instant,
}

impl Stall {
    pub fn new(idle: Duration) -> Stall {
        Stall {
            idle,
            acked: 0,
            since: Instant::now(),
        }
    }

    pub fn idle(&self) -> Duration {
        self.idle
    }

    /// `written`: bytes written to the socket so far; `unacked`: of them,
    /// those not acknowledged. Whether it has stalled for `idle`.
    pub fn check(&mut self, written: u64, unacked: Option<u64>) -> bool {
        let Some(u) = unacked else {
            return false;
        };
        let acked = written.saturating_sub(u);
        if u == 0 || acked != self.acked {
            self.acked = acked;
            self.since = Instant::now();
            return false;
        }
        self.since.elapsed() >= self.idle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ct_eq_and_bearer() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
        assert!(!ct_eq(b"", b"a"));
        assert!(ct_eq(b"", b""));
        let h = |v: &str| vec![("authorization".to_string(), v.to_string())];
        assert!(authorized(&h("Bearer k1"), "k1"));
        assert!(authorized(&h("bearer  k1"), "k1"));
        assert!(!authorized(&h("Bearer k2"), "k1"));
        assert!(!authorized(&h("Basic k1"), "k1"));
        assert!(!authorized(&h("Bearer"), "k1"));
        assert!(!authorized(&h("k1"), "k1"));
        assert!(!authorized(&[], "k1"));
        // An empty key never accepts an empty token.
        assert!(!authorized(&h("Bearer "), ""));
        assert!(!authorized(&h("Bearer"), ""));
        assert!(!authorized(&h("Bearer x"), ""));
    }

    #[test]
    fn histogram_is_cumulative() {
        let h = Histogram::new(&[0.1, 1.0]);
        for ms in [50, 50, 500, 5000] {
            h.observe(Duration::from_millis(ms));
        }
        let mut o = String::new();
        h.render(&mut o, "x", "y");
        assert!(o.contains("x_bucket{le=\"0.1\"} 2\n"), "{o}");
        assert!(o.contains("x_bucket{le=\"1\"} 3\n"), "{o}");
        assert!(o.contains("x_bucket{le=\"+Inf\"} 4\n"), "{o}");
        assert!(o.contains("x_count 4\n"), "{o}");
        assert!(o.contains("x_sum 5.6\n"), "{o}");
    }

    #[test]
    fn stall_needs_waiting_bytes_and_no_progress() {
        let mut s = Stall::new(Duration::from_millis(30));
        assert!(!s.check(100, Some(0)));
        std::thread::sleep(Duration::from_millis(40));
        // Nothing waiting: not stalled, however long.
        assert!(!s.check(100, Some(0)));
        // Bytes wait but some were acknowledged since: progress.
        assert!(!s.check(200, Some(50)));
        std::thread::sleep(Duration::from_millis(40));
        // The same count acknowledged for 40 ms with bytes waiting.
        assert!(s.check(300, Some(150)));
        assert!(!s.check(300, None), "unknown: never stalled");
    }

    #[test]
    fn slot_counts() {
        let mut v = SlotsView::default();
        assert_eq!(v.counts(), (1, 0));
        v.pool = vec![SlotState::default(); 4];
        assert_eq!(v.counts(), (4, 0));
        // A request alone runs in a slot: one busy, not one more.
        v.pool[0].busy = true;
        assert_eq!(v.counts(), (4, 1));
        v.pool[0].busy = false;
        v.pool[2].busy = true;
        v.pool[3].busy = true;
        assert_eq!(v.counts(), (4, 2));
        // One request at a time: the one cache is slot 0.
        let m = Metrics::default();
        let slot = |m: &Metrics| -> serde_json::Value {
            serde_json::from_str::<serde_json::Value>(&m.slots_json(64)).unwrap()["slots"].clone()
        };
        assert_eq!(slot(&m).as_array().unwrap().len(), 1);
        assert_eq!(slot(&m)[0]["busy"], false);
        m.publish_slot(
            0,
            SlotState {
                held: 9,
                busy: true,
                request: None,
                pages: None,
            },
        );
        assert_eq!(lock(&m.slots).counts(), (1, 1));
        assert_eq!(slot(&m)[0]["held_tokens"], 9);
        let pv = PagesView {
            free: 10,
            total: 12,
            size: 64,
            preempted: 2,
            unfed: 1,
        };
        m.publish_pool(vec![SlotState::default(); 3], Some(pv));
        m.publish_slot(
            2,
            SlotState {
                held: 5,
                busy: true,
                request: Some((7, Instant::now())),
                pages: Some(2),
            },
        );
        assert_eq!(slot(&m).as_array().unwrap().len(), 3);
        assert_eq!(slot(&m)[2]["held_tokens"], 5);
        assert_eq!(slot(&m)[2]["pages_used"], 2);
        let all = serde_json::from_str::<serde_json::Value>(&m.slots_json(64)).unwrap();
        assert_eq!(
            (all["pages_free"].clone(), all["pages_total"].clone()),
            (10.into(), 12.into())
        );
        assert_eq!(all["preempted"], 2);
        assert_eq!(all["prefills_unfed"], 1);
        assert_eq!(lock(&m.slots).counts(), (3, 1));
    }
}
