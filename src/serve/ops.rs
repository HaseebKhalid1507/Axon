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
}

impl SlotState {
    fn json(&self, now: Instant) -> serde_json::Value {
        serde_json::json!({
            "busy": self.busy,
            "held_tokens": self.held,
            "request_id": self.request.map(|r| r.0),
            "age_ms": self.request.map(|r| now.saturating_duration_since(r.1).as_millis() as u64),
        })
    }
}

/// The engine's last published slot states.
#[derive(Default)]
pub struct SlotsView {
    /// The batch's pool (empty: one request at a time).
    pub pool: Vec<SlotState>,
    /// The engine's own cache (`Engine::job`: a request running alone).
    /// Becomes slot 0 once #60 (one cache: the solo path runs in slot 0) lands.
    pub solo: SlotState,
}

impl SlotsView {
    /// Slots in total, and busy: the pool's (or, with none, the one cache).
    /// A request running alone beside an idle pool counts as one busy.
    pub fn counts(&self) -> (usize, usize) {
        let total = self.pool.len().max(1);
        let busy = self.pool.iter().filter(|s| s.busy).count() + usize::from(self.solo.busy);
        (total, busy.min(total))
    }
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
    pub ttft: Histogram,
    pub itl: Histogram,
    pub slots: Mutex<SlotsView>,
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
            ttft: Histogram::new(TTFT_BOUNDS),
            itl: Histogram::new(ITL_BOUNDS),
            slots: Mutex::default(),
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

    pub fn publish_pool(&self, pool: Vec<SlotState>) {
        lock(&self.slots).pool = pool;
    }

    pub fn publish_solo(&self, solo: SlotState) {
        lock(&self.slots).solo = solo;
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
        let pool: Vec<_> = v
            .pool
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let mut j = s.json(now);
                j["id"] = i.into();
                j
            })
            .collect();
        serde_json::json!({
            "ctx": ctx,
            "slots": pool,
            "solo": v.solo.json(now),
            "queue_depth": self.queue_depth(busy),
        })
        .to_string()
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

/// `unacked`, with one warning on stderr the first time it can't be read
/// (only a blocked write is detected then).
pub fn unacked_or_warn(s: &TcpStream) -> Option<u64> {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let u = unacked(s);
    if u.is_none() && OUTQ_PROBE && !WARNED.swap(true, Ordering::Relaxed) {
        eprintln!(
            "axon serve: --slot-idle-ms: can't read a socket's unacknowledged bytes (SIOCOUTQ); \
             only a write blocked for the idle time is detected"
        );
    }
    u
}

/// Bytes written to `s` the peer hasn't acknowledged yet (Linux SIOCOUTQ),
/// None where that can't be read.
#[cfg(all(
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
))]
pub fn unacked(s: &TcpStream) -> Option<u64> {
    use std::os::fd::AsRawFd;
    unsafe extern "C" {
        fn ioctl(fd: std::ffi::c_int, req: std::ffi::c_ulong, ...) -> std::ffi::c_int;
    }
    const SIOCOUTQ: std::ffi::c_ulong = 0x5411;
    let mut n: std::ffi::c_int = 0;
    // SAFETY: SIOCOUTQ writes one int through the pointer, on an fd we own.
    let r = unsafe { ioctl(s.as_raw_fd(), SIOCOUTQ, &mut n as *mut std::ffi::c_int) };
    (r == 0 && n >= 0).then_some(n as u64)
}

#[cfg(not(all(
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
)))]
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
        v.solo.busy = true;
        assert_eq!(v.counts(), (1, 1));
        v.pool = vec![SlotState::default(); 4];
        assert_eq!(v.counts(), (4, 1));
        v.solo.busy = false;
        v.pool[2].busy = true;
        v.pool[3].busy = true;
        assert_eq!(v.counts(), (4, 2));
    }
}
