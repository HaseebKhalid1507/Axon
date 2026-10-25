//! RAM available to this process: the host's `MemAvailable`, bounded by
//! the tightest memory cgroup limit above the process (a container, a
//! systemd scope or slice). `gguf::mem_available` and `moe::check_ram` ask
//! this: under a cgroup limit tighter than the host's free memory, a model
//! the host could hold is still OOM-killed by the kernel, so the limit is
//! what the load has to fit in.
//!
//! cgroup v2: at each level from the process's cgroup up to the mount
//! root, `memory.max` (`max`: no limit) less what the cgroup uses that
//! can't be reclaimed: `memory.current` less its file pages (`memory.stat`
//! `inactive_file` + `active_file`, page cache the kernel drops before it
//! OOM-kills). The tightest level wins. `memory.high` is not counted: over
//! it the kernel throttles and reclaims, it doesn't kill. Swap
//! (`memory.swap.max`) isn't counted either, as `MemAvailable` doesn't.
//!
//! cgroup v1 (the memory controller on a v1 hierarchy): the same walk with
//! `memory.limit_in_bytes` (a value of 2^62 or more: no limit),
//! `memory.usage_in_bytes` and `memory.stat`'s `total_inactive_file` +
//! `total_active_file` (else the cgroup's own `inactive_file` +
//! `active_file`). Every ancestor's limit is counted, as kernels since 5.x
//! always account hierarchically (`use_hierarchy` is gone).
//!
//! Anything unreadable is skipped: no cgroupfs, no memory controller, a
//! level without the files (the v2 root has no `memory.max`) leaves the
//! host's figure. Paths are read under `root` so the tests run against
//! synthetic trees.

use std::fmt;
use std::path::{Path, PathBuf};

/// Bytes available, and what bounds them.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Avail {
    pub bytes: u64,
    pub bound: Bound,
}

/// What bounds `Avail::bytes`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Bound {
    /// The host's `MemAvailable`.
    Host,
    /// A cgroup's limit: its path as `/proc/self/cgroup` names it (in the
    /// process's cgroup namespace), the limit in bytes, v1 or v2.
    Cgroup { path: String, limit: u64, v1: bool },
}

impl fmt::Display for Avail {
    /// What bound the figure, for messages: "host MemAvailable" or
    /// "cgroup /a/b memory.max 1000 MB".
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.bound {
            Bound::Host => write!(f, "host MemAvailable"),
            Bound::Cgroup { path, limit, v1 } => write!(
                f,
                "cgroup {path} {} {} MB",
                if *v1 {
                    "memory.limit_in_bytes"
                } else {
                    "memory.max"
                },
                limit >> 20
            ),
        }
    }
}

/// `avail_at` of the real filesystem.
pub(crate) fn avail() -> Option<Avail> {
    avail_at(Path::new("/"))
}

/// The smaller of the host's `MemAvailable` (`root`/proc/meminfo) and the
/// tightest cgroup level's room (`cgroup_room`); whichever is known if one
/// isn't, None if neither is. A tie goes to the host.
pub(crate) fn avail_at(root: &Path) -> Option<Avail> {
    let host = host_available(root).map(|bytes| Avail {
        bytes,
        bound: Bound::Host,
    });
    match (host, cgroup_room(root)) {
        (Some(h), Some(c)) if c.bytes < h.bytes => Some(c),
        (Some(h), _) => Some(h),
        (None, c) => c,
    }
}

/// `MemAvailable` from `root`/proc/meminfo, in bytes.
fn host_available(root: &Path) -> Option<u64> {
    let s = std::fs::read_to_string(root.join("proc/meminfo")).ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// The cgroup hierarchy that holds the memory controller.
#[derive(Debug, PartialEq)]
struct Hier {
    v1: bool,
    /// The process's cgroup path (`/proc/self/cgroup`).
    path: String,
    /// Where the hierarchy is mounted, and the cgroup its mount shows at
    /// that point (mountinfo's root field: `/` unless only a subtree is
    /// mounted, as in a container without its own cgroup namespace).
    mount: PathBuf,
    mount_root: String,
}

/// The memory controller's hierarchy of this process, from
/// `root`/proc/self/cgroup and `root`/proc/self/mountinfo. A v1 line with
/// the memory controller wins (on a hybrid system the v2 hierarchy has no
/// memory controller then); else the v2 line (`0::path`). Without
/// mountinfo, or no matching mount in it, the usual mount points:
/// /sys/fs/cgroup (v2), /sys/fs/cgroup/memory (v1).
fn hierarchy(root: &Path) -> Option<Hier> {
    let cg = std::fs::read_to_string(root.join("proc/self/cgroup")).ok()?;
    let mut v2 = None;
    let mut v1 = None;
    for l in cg.lines() {
        let mut f = l.splitn(3, ':');
        let (Some(id), Some(ctl), Some(path)) = (f.next(), f.next(), f.next()) else {
            continue;
        };
        if id == "0" && ctl.is_empty() {
            v2 = Some(path.to_string());
        } else if ctl.split(',').any(|c| c == "memory") {
            v1 = Some(path.to_string());
        }
    }
    let (is_v1, path) = match (v1, v2) {
        (Some(p), _) => (true, p),
        (None, Some(p)) => (false, p),
        (None, None) => return None,
    };
    let mi = std::fs::read_to_string(root.join("proc/self/mountinfo")).unwrap_or_default();
    let found = mi.lines().find_map(|l| {
        // id parent maj:min root mountpoint opts [optional...] - fstype source superopts
        let (pre, post) = l.split_once(" - ")?;
        let pre: Vec<&str> = pre.split(' ').collect();
        let post: Vec<&str> = post.split(' ').collect();
        let (mroot, mpoint) = (*pre.get(3)?, *pre.get(4)?);
        let fstype = *post.first()?;
        let ok = match is_v1 {
            false => fstype == "cgroup2",
            true => fstype == "cgroup" && post.get(2)?.split(',').any(|o| o == "memory"),
        };
        ok.then(|| (unescape(mpoint), unescape(mroot)))
    });
    let (mount, mount_root) = found.unwrap_or_else(|| {
        let p = if is_v1 {
            "/sys/fs/cgroup/memory"
        } else {
            "/sys/fs/cgroup"
        };
        (p.to_string(), "/".to_string())
    });
    Some(Hier {
        v1: is_v1,
        path,
        mount: root.join(mount.trim_start_matches('/')),
        mount_root,
    })
}

/// mountinfo's octal escapes (`\040` for a space and so on).
fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let oct = b
            .get(i + 1..i + 4)
            .filter(|o| b[i] == b'\\' && o.iter().all(|c| (b'0'..=b'7').contains(c)));
        match oct {
            Some(o) => {
                out.push(
                    o.iter()
                        .fold(0u8, |a, c| a.wrapping_mul(8).wrapping_add(c - b'0')),
                );
                i += 4;
            }
            None => {
                out.push(b[i]);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// v1 limits at or above this are "no limit" (the kernel reports
/// PAGE_COUNTER_MAX pages, 2^63 rounded down to a page, when none is set).
const V1_UNLIMITED: u64 = 1 << 62;

/// The tightest cgroup level's room (module docs), None if no level up
/// to the mount root has a readable limit.
fn cgroup_room(root: &Path) -> Option<Avail> {
    let h = hierarchy(root)?;
    // The cgroup below the mount's root: the mount shows `mount_root` at
    // `mount`. A path outside it (it shouldn't be) is taken as is.
    let rel = match h.path.strip_prefix(h.mount_root.trim_end_matches('/')) {
        Some(r) if r.is_empty() || r.starts_with('/') => r,
        _ => h.path.as_str(),
    };
    let rel = rel.trim_matches('/');
    let mut parts: Vec<&str> = rel.split('/').filter(|p| !p.is_empty()).collect();
    let (limit_f, usage_f) = match h.v1 {
        true => ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        false => ("memory.max", "memory.current"),
    };
    let mut best: Option<Avail> = None;
    loop {
        let dir = parts.iter().fold(h.mount.clone(), |d, p| d.join(p));
        if let Some(limit) = read_limit(&dir.join(limit_f), h.v1) {
            let usage = read_u64(&dir.join(usage_f)).unwrap_or(0);
            let file = file_pages(&dir.join("memory.stat"), h.v1);
            let room = limit.saturating_sub(usage.saturating_sub(file));
            if best.as_ref().is_none_or(|b| room < b.bytes) {
                let below = parts.join("/");
                let path = match h.mount_root.trim_end_matches('/') {
                    "" => format!("/{below}"),
                    r if below.is_empty() => r.to_string(),
                    r => format!("{r}/{below}"),
                };
                best = Some(Avail {
                    bytes: room,
                    bound: Bound::Cgroup {
                        path,
                        limit,
                        v1: h.v1,
                    },
                });
            }
        }
        if parts.pop().is_none() {
            break;
        }
    }
    best
}

/// A limit file: None for no limit (`max`, v1's huge value) or unreadable.
fn read_limit(p: &Path, v1: bool) -> Option<u64> {
    let s = std::fs::read_to_string(p).ok()?;
    let s = s.trim();
    if s == "max" {
        return None;
    }
    let v: u64 = s.parse().ok()?;
    (!v1 || v < V1_UNLIMITED).then_some(v)
}

fn read_u64(p: &Path) -> Option<u64> {
    std::fs::read_to_string(p).ok()?.trim().parse().ok()
}

/// Reclaimable file pages from a `memory.stat` (0 if unreadable): v2's
/// `inactive_file` + `active_file`; v1's hierarchical `total_*` ones, else
/// the cgroup's own.
fn file_pages(p: &Path, v1: bool) -> u64 {
    let Ok(s) = std::fs::read_to_string(p) else {
        return 0;
    };
    let get = |k: &str| {
        s.lines().find_map(|l| {
            let (key, v) = l.split_once(' ')?;
            (key == k).then(|| v.trim().parse::<u64>().ok()).flatten()
        })
    };
    let sum = |a: &str, b: &str| Some(get(a)?.saturating_add(get(b).unwrap_or(0)));
    match v1 {
        true => sum("total_inactive_file", "total_active_file")
            .or_else(|| sum("inactive_file", "active_file")),
        false => sum("inactive_file", "active_file"),
    }
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic root (proc + cgroupfs) under the temp dir, removed on drop.
    struct Root(PathBuf);

    impl Root {
        fn new(tag: &str) -> Root {
            let p = std::env::temp_dir().join(format!("axon-memcg-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(p.join("proc/self")).unwrap();
            Root(p)
        }
        fn put(&self, rel: &str, s: &str) -> &Root {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, s).unwrap();
            self
        }
        /// A v2 cgroup level: memory.max, memory.current, memory.stat files.
        fn v2(&self, dir: &str, max: &str, current: u64, file: (u64, u64)) -> &Root {
            let d = format!("sys/fs/cgroup/{dir}");
            self.put(&format!("{d}/memory.max"), &format!("{max}\n"));
            self.put(&format!("{d}/memory.current"), &format!("{current}\n"));
            self.put(
                &format!("{d}/memory.stat"),
                &format!(
                    "anon 1\nfile 9\ninactive_file {}\nactive_file {}\nshmem 0\n",
                    file.0, file.1
                ),
            )
        }
        fn avail(&self) -> Option<Avail> {
            avail_at(&self.0)
        }
    }

    impl Drop for Root {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const MB: u64 = 1 << 20;
    const GB: u64 = 1 << 30;
    const MEMINFO_16G: &str =
        "MemTotal:       32000000 kB\nMemFree:  100 kB\nMemAvailable:   16777216 kB\n";
    const V2_MOUNT: &str = "46 44 0:29 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:7 - cgroup2 cgroup2 rw,nsdelegate\n";

    fn cg(path: &str, limit: u64, v1: bool) -> Bound {
        Bound::Cgroup {
            path: path.into(),
            limit,
            v1,
        }
    }

    /// No cgroup files at all: the host's figure, as before.
    #[test]
    fn host_only_without_cgroupfs() {
        let r = Root::new("hostonly");
        r.put("proc/meminfo", MEMINFO_16G);
        assert_eq!(
            r.avail(),
            Some(Avail {
                bytes: 16 * GB,
                bound: Bound::Host
            })
        );
        // /proc/self/cgroup but no cgroupfs under it: still the host.
        r.put("proc/self/cgroup", "0::/a/b\n");
        assert_eq!(r.avail().unwrap().bound, Bound::Host);
        // Nothing readable at all.
        let e = Root::new("nothing");
        assert_eq!(e.avail(), None);
    }

    /// v2: the tightest level up the tree binds; "max" levels and the root
    /// (no memory.max) are skipped; file pages count as free.
    #[test]
    fn v2_tightest_level_wins() {
        let r = Root::new("v2walk");
        r.put("proc/meminfo", MEMINFO_16G)
            .put("proc/self/cgroup", "0::/user.slice/app.slice/job.scope\n")
            .put("proc/self/mountinfo", V2_MOUNT)
            .put("sys/fs/cgroup/memory.stat", "anon 5\n")
            .v2("user.slice", "max", 3 * GB, (0, 0))
            // 23 GB slice, 10 GB used of which 2 GB file: 15 GB room.
            .v2(
                "user.slice/app.slice",
                &(23 * GB).to_string(),
                10 * GB,
                (GB, GB),
            )
            // 1000 MB scope, 300 MB used of which 100 MB file: 800 MB room.
            .v2(
                "user.slice/app.slice/job.scope",
                &(1000 * MB).to_string(),
                300 * MB,
                (60 * MB, 40 * MB),
            );
        let a = r.avail().unwrap();
        assert_eq!(a.bytes, 800 * MB);
        assert_eq!(
            a.bound,
            cg("/user.slice/app.slice/job.scope", 1000 * MB, false)
        );
        assert_eq!(
            a.to_string(),
            "cgroup /user.slice/app.slice/job.scope memory.max 1000 MB"
        );
        // The scope unlimited: the slice binds (15 GB < host 16 GB).
        r.put(
            "sys/fs/cgroup/user.slice/app.slice/job.scope/memory.max",
            "max\n",
        );
        let a = r.avail().unwrap();
        assert_eq!(a.bytes, 15 * GB);
        assert_eq!(a.bound, cg("/user.slice/app.slice", 23 * GB, false));
        // The slice roomier than the host: the host binds.
        r.put(
            "sys/fs/cgroup/user.slice/app.slice/memory.current",
            &(GB).to_string(),
        );
        assert_eq!(r.avail().unwrap().bound, Bound::Host);
        assert_eq!(r.avail().unwrap().to_string(), "host MemAvailable");
    }

    /// Usage above the limit (or file pages above usage) saturates at 0
    /// room instead of wrapping; a level with only memory.max counts the
    /// whole limit; no meminfo leaves the cgroup's figure.
    #[test]
    fn v2_saturates_and_tolerates_missing_files() {
        let r = Root::new("v2sat");
        r.put("proc/self/cgroup", "0::/c\n")
            .put("proc/self/mountinfo", V2_MOUNT);
        r.v2("c", "1000", 5000, (10, 10));
        assert_eq!(r.avail().unwrap().bytes, 0);
        r.v2("c", "1000", 10, (500, 500));
        assert_eq!(r.avail().unwrap().bytes, 1000);
        let _ = std::fs::remove_file(r.0.join("sys/fs/cgroup/c/memory.current"));
        let _ = std::fs::remove_file(r.0.join("sys/fs/cgroup/c/memory.stat"));
        assert_eq!(r.avail().unwrap().bytes, 1000);
        // Garbage in memory.max: no limit there.
        r.put("sys/fs/cgroup/c/memory.max", "lots\n");
        assert_eq!(r.avail(), None);
    }

    /// No mountinfo: the default mount point. A container without its own
    /// cgroup namespace: /proc/self/cgroup names the host path, the mount
    /// shows that cgroup at /sys/fs/cgroup (mountinfo root field).
    #[test]
    fn v2_mount_root_and_default_mount() {
        let r = Root::new("v2mnt");
        r.put("proc/meminfo", MEMINFO_16G)
            .put("proc/self/cgroup", "0::/a\n");
        r.v2("a", &(2 * GB).to_string(), 0, (0, 0));
        assert_eq!(r.avail().unwrap().bound, cg("/a", 2 * GB, false));

        let c = Root::new("v2ns");
        c.put("proc/meminfo", MEMINFO_16G)
            .put("proc/self/cgroup", "0::/system.slice/docker-1.scope\n")
            .put(
                "proc/self/mountinfo",
                "1 0 0:1 / / rw - overlay overlay rw\n\
                 9 1 0:29 /system.slice/docker-1.scope /sys/fs/cgroup ro,nosuid - cgroup2 cgroup rw\n",
            );
        // The container's own cgroup is the mount's top directory.
        c.v2("", &(512 * MB).to_string(), 12 * MB, (0, 0));
        let a = c.avail().unwrap();
        assert_eq!(a.bytes, 500 * MB);
        assert_eq!(a.bound, cg("/system.slice/docker-1.scope", 512 * MB, false));
    }

    /// v1: the memory controller's line and mount, its file names, its
    /// "unlimited" value, total_* stats preferred; it wins over a v2 line
    /// without the controller (hybrid).
    #[test]
    fn v1_memory_controller() {
        let r = Root::new("v1");
        let d = "sys/fs/cgroup/memory";
        r.put("proc/meminfo", MEMINFO_16G)
            .put("proc/self/cgroup", "12:cpu,cpuacct:/x\n7:memory:/kube/pod1\n0::/x\n")
            .put(
                "proc/self/mountinfo",
                "30 25 0:26 / /sys/fs/cgroup/memory rw,nosuid - cgroup cgroup rw,memory\n\
                 31 25 0:27 / /sys/fs/cgroup/unified rw - cgroup2 cgroup2 rw\n",
            )
            .put(&format!("{d}/memory.limit_in_bytes"), "9223372036854771712\n")
            .put(&format!("{d}/kube/memory.limit_in_bytes"), &format!("{}\n", 4 * GB))
            .put(&format!("{d}/kube/memory.usage_in_bytes"), &format!("{}\n", GB))
            .put(&format!("{d}/kube/pod1/memory.limit_in_bytes"), &format!("{}\n", 2 * GB))
            .put(&format!("{d}/kube/pod1/memory.usage_in_bytes"), &format!("{}\n", GB))
            .put(
                &format!("{d}/kube/pod1/memory.stat"),
                &format!(
                    "inactive_file 1\nactive_file 1\ntotal_inactive_file {}\ntotal_active_file {}\n",
                    256 * MB,
                    256 * MB
                ),
            );
        let a = r.avail().unwrap();
        assert_eq!(a.bytes, 2 * GB - GB + 512 * MB);
        assert_eq!(a.bound, cg("/kube/pod1", 2 * GB, true));
        assert_eq!(
            a.to_string(),
            "cgroup /kube/pod1 memory.limit_in_bytes 2048 MB"
        );
        // Only the cgroup's own stats: those.
        r.put(
            &format!("{d}/kube/pod1/memory.stat"),
            &format!("inactive_file {}\n", 100 * MB),
        );
        assert_eq!(r.avail().unwrap().bytes, GB + 100 * MB);
        // The pod unlimited: the parent binds (4 - 1 GB, no stat file).
        r.put(
            &format!("{d}/kube/pod1/memory.limit_in_bytes"),
            "9223372036854771712\n",
        );
        assert_eq!(r.avail().unwrap().bytes, 3 * GB);
        assert_eq!(r.avail().unwrap().bound, cg("/kube", 4 * GB, true));
    }

    #[test]
    fn mountinfo_escapes() {
        assert_eq!(unescape(r"/a\040b"), "/a b");
        assert_eq!(unescape(r"/a\\b"), r"/a\\b");
        assert_eq!(unescape(r"/x\04"), r"/x\04");
    }

    /// The real machine: whatever it is, the figure is at most the host's.
    #[test]
    fn real_avail_at_most_host() {
        let (Some(a), Some(h)) = (avail(), host_available(Path::new("/"))) else {
            return;
        };
        assert!(a.bytes <= h, "{a:?} > host {h}");
        eprintln!("mem avail: {} MB ({a})", a.bytes >> 20);
    }
}
