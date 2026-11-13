//! `axon pull`: a GGUF file from Hugging Face into the models directory,
//! by a short name (`ALIASES`: about ten checksummed models for local
//! coding agents) or as `OWNER/REPO/FILE.gguf`. No registry: a name is a
//! fixed Hugging Face file at a pinned commit with its SHA-256; any other
//! file's checksum is the one Hugging Face publishes (the LFS oid, in the
//! `x-linked-etag` header of the file's URL).
//!
//! The bytes come through `curl` (on every Linux and on Windows 10+; axon
//! has no TLS of its own): HTTPS only, redirects included (`--proto
//! =https --proto-redir =https`), streamed into this process, hashed
//! (`crate::sha256`) and appended to `FILE.part` as they arrive. An
//! interrupted pull resumes where the `.part` ends (its bytes hashed
//! again first). The file is renamed into place only once its size and
//! SHA-256 match; a mismatch deletes the `.part`.
//!
//! Where: `$AXON_MODELS_DIR`, else `$XDG_DATA_HOME/axon/models`, else
//! `~/.local/share/axon/models`, as `OWNER/REPO/FILE` under it, so two
//! repos' files of one name never collide. `gen`, `chat`, `serve` and
//! `info` take a name, an `OWNER/REPO/FILE.gguf` or a pulled file's name
//! where they take a path (`resolve`).
//!
//! Every name from the command line or a server is untrusted: owners,
//! repos and file names are single path components of `[A-Za-z0-9._-]`
//! that start with a letter or digit (`component`), so none is `.`, `..`,
//! absolute or a curl option, and the file lands under the models
//! directory, nowhere else.

use crate::sha256::Sha256;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

/// A short name for a Hugging Face file.
pub struct Alias {
    pub name: &'static str,
    /// `OWNER/REPO`.
    pub repo: &'static str,
    pub file: &'static str,
    /// The commit the file is fetched at (40 hex digits).
    pub rev: &'static str,
    /// Its SHA-256 (Hugging Face's LFS oid), 64 hex digits.
    pub sha256: &'static str,
    pub size: u64,
    /// For `axon pull --list`.
    pub what: &'static str,
}

/// The names `axon pull` knows. Each checksum is Hugging Face's LFS oid
/// for the file at `rev` (`x-linked-etag`, checked 2026-11-12; where bella
/// has the file, its `.sha256` agrees). Architectures axon runs: qwen2,
/// qwen3, qwen3moe, llama, gemma3, gpt-oss.
pub const ALIASES: &[Alias] = &[
    Alias {
        name: "qwen2.5-coder-1.5b",
        repo: "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        file: "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        rev: "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
        sha256: "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
        size: 1_117_320_768,
        what: "Qwen2.5 Coder 1.5B Instruct, Q4_K_M: small and fast; a draft model for the 7B/14B",
    },
    Alias {
        name: "qwen2.5-coder-7b",
        repo: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        file: "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        rev: "13fb94bfda8c8cf22497dc57b78f391a9acb426a",
        sha256: "509287f78cb4d4cf6b3843734733b914b2c158e43e22a7f4bf5e963800894d3c",
        size: 4_683_073_536,
        what: "Qwen2.5 Coder 7B Instruct, Q4_K_M (8 GB GPUs)",
    },
    Alias {
        name: "qwen2.5-coder-14b",
        repo: "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
        file: "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        rev: "d0a692ef765eefbf2fabb130b3cb2e8917e3d225",
        sha256: "c1e659736d89ac1065fb495330fb824d94001974a4bfa78e7270e43476a8d940",
        size: 8_988_110_272,
        what: "Qwen2.5 Coder 14B Instruct, Q4_K_M (12 GB GPUs)",
    },
    Alias {
        name: "qwen3-4b",
        repo: "Qwen/Qwen3-4B-GGUF",
        file: "Qwen3-4B-Q4_K_M.gguf",
        rev: "bc640142c66e1fdd12af0bd68f40445458f3869b",
        sha256: "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
        size: 2_497_280_256,
        what: "Qwen3 4B, Q4_K_M (thinking; 6 GB GPUs)",
    },
    Alias {
        name: "qwen3-8b",
        repo: "Qwen/Qwen3-8B-GGUF",
        file: "Qwen3-8B-Q4_K_M.gguf",
        rev: "7c41481f57cb95916b40956ab2f0b139b296d974",
        sha256: "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
        size: 5_027_783_488,
        what: "Qwen3 8B, Q4_K_M (thinking; 8 GB GPUs)",
    },
    Alias {
        name: "qwen3-14b",
        repo: "Qwen/Qwen3-14B-GGUF",
        file: "Qwen3-14B-Q4_K_M.gguf",
        rev: "530227a7d994db8eca5ab5ced2fb692b614357fd",
        sha256: "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0",
        size: 9_001_752_960,
        what: "Qwen3 14B, Q4_K_M (thinking; 12 GB GPUs)",
    },
    Alias {
        name: "qwen3-30b-a3b",
        repo: "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF",
        file: "Qwen_Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf",
        rev: "6c6e8692f43e4ca663f7ece8229a1361090d3a4c",
        sha256: "382b4f5a164d200f93790ee0e339fae12852896d23485cfb203ce868fea33a95",
        size: 18_632_183_808,
        what: "Qwen3 30B-A3B Instruct 2507, Q4_K_M (mixture of experts: experts on the CPU, the rest on the GPU)",
    },
    Alias {
        name: "qwen3-coder-30b-a3b",
        repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        file: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        rev: "b17cb02dd882d5b6ab62fc777ad2995f19668350",
        sha256: "fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad",
        size: 18_556_689_568,
        what: "Qwen3 Coder 30B-A3B Instruct, Q4_K_M (mixture of experts, as qwen3-30b-a3b)",
    },
    Alias {
        name: "llama3.1-8b",
        repo: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        file: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        rev: "bf5b95e96dac0462e2a09145ec66cae9a3f12067",
        sha256: "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c",
        size: 4_920_739_232,
        what: "Llama 3.1 8B Instruct, Q4_K_M (8 GB GPUs)",
    },
    Alias {
        name: "gpt-oss-20b",
        repo: "ggml-org/gpt-oss-20b-GGUF",
        file: "gpt-oss-20b-MXFP4.gguf",
        rev: "41e8a94d15d62070b36cf4553a0250717ecf46b8",
        sha256: "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901",
        size: 12_109_566_624,
        what: "gpt-oss 20B, MXFP4 (mixture of experts: experts on the CPU, the rest on the GPU)",
    },
    Alias {
        name: "gemma3-4b",
        repo: "bartowski/google_gemma-3-4b-it-GGUF",
        file: "google_gemma-3-4b-it-Q4_K_M.gguf",
        rev: "71506238f970075ca85125cd749c28b1b0eee84e",
        sha256: "4996030242583a40aa151ff93f49ed787ac8c25e4120c3ae4588b2e2a7d1ae94",
        size: 2_489_758_112,
        what: "Gemma 3 4B Instruct, Q4_K_M (text only; 6 GB GPUs)",
    },
    Alias {
        name: "qwen2.5-0.5b",
        repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        rev: "9217f5db79a29953eb74d5343926648285ec7e67",
        sha256: "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        size: 491_400_032,
        what: "Qwen2.5 0.5B Instruct, Q4_K_M: tiny, for trying axon out; a draft model for Qwen2.5",
    },
];

/// The alias called `name`, if any.
pub fn alias(name: &str) -> Option<&'static Alias> {
    ALIASES.iter().find(|a| a.name == name)
}

/// Whether `s` is a safe single path component named `what`: 1 to `max`
/// bytes of `[A-Za-z0-9._-]`, starting with a letter or digit (so never
/// `.`, `..`, hidden, or an option), not ending in a dot.
fn component(s: &str, what: &str, max: usize) -> Result<(), String> {
    let ok = !s.is_empty()
        && s.len() <= max
        && s.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric())
        && !s.ends_with('.')
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
    match ok {
        true => Ok(()),
        false => Err(format!(
            "bad {what} '{}': letters, digits, '.', '_' and '-' only, starting with a letter or digit",
            s.escape_debug()
        )),
    }
}

/// A file in a Hugging Face model repo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfFile {
    pub owner: String,
    pub repo: String,
    /// Its path in the repo, by component (`dir/file.gguf`: two).
    pub path: Vec<String>,
}

impl HfFile {
    /// `OWNER/REPO/PATH.gguf` (PATH may have directories), optionally after
    /// `hf.co/` or `huggingface.co/`, or a file's URL as the website shows it
    /// (`https://huggingface.co/OWNER/REPO/resolve/main/PATH`, or `/blob/`).
    /// Err: why it isn't one (`component`'s rules for every part; a `.gguf`
    /// file; not one part of a split GGUF, which axon can't load).
    pub fn parse(spec: &str) -> Result<HfFile, String> {
        let url = spec.strip_prefix("https://");
        let s = url.unwrap_or(spec);
        let host = s
            .strip_prefix("huggingface.co/")
            .or_else(|| s.strip_prefix("hf.co/"));
        if url.is_some() && host.is_none() {
            return Err(format!(
                "'{}' isn't a huggingface.co URL",
                spec.escape_debug()
            ));
        }
        let mut parts: Vec<&str> = host.unwrap_or(s).split('/').collect();
        // A URL as the website shows it: /resolve/main/ or /blob/main/
        // between the repo and the path (required after https://).
        let site = parts.len() >= 5 && matches!(parts[2], "resolve" | "blob") && parts[3] == "main";
        if url.is_some() && !site {
            return Err(format!(
                "'{}' isn't a file URL (https://huggingface.co/OWNER/REPO/resolve/main/FILE)",
                spec.escape_debug()
            ));
        }
        if site && host.is_some() {
            parts.drain(2..4);
        }
        if parts.len() < 3 {
            return Err(format!(
                "'{}' isn't OWNER/REPO/FILE.gguf (a Hugging Face file) or a name `axon pull --list` knows",
                spec.escape_debug()
            ));
        }
        component(parts[0], "owner", 96)?;
        component(parts[1], "repo", 96)?;
        if parts.len() > 8 {
            return Err("too many directories in the file's path".into());
        }
        for p in &parts[2..] {
            component(p, "file path component", 200)?;
        }
        let file = parts[parts.len() - 1];
        if !file.to_ascii_lowercase().ends_with(".gguf") {
            return Err(format!("'{file}' isn't a .gguf file"));
        }
        if is_split_part(file) {
            return Err(format!(
                "'{file}' is one part of a split GGUF, which axon can't load yet; pick a single-file \
                 quantization"
            ));
        }
        Ok(HfFile {
            owner: parts[0].into(),
            repo: parts[1].into(),
            path: parts[2..].iter().map(|p| p.to_string()).collect(),
        })
    }

    /// `OWNER/REPO`.
    pub fn repo_id(&self) -> String {
        format!("{}/{}", self.owner, self.repo)
    }

    /// Its file name (the path's last component).
    pub fn file_name(&self) -> &str {
        self.path.last().expect("parse makes at least one")
    }

    /// Its URL at revision `rev` (a commit or `main`).
    pub fn url(&self, rev: &str) -> String {
        format!(
            "https://huggingface.co/{}/{}/resolve/{rev}/{}",
            self.owner,
            self.repo,
            self.path.join("/")
        )
    }

    /// Where it's kept under models directory `dir`: `dir/OWNER/REPO/PATH`,
    /// its path in the repo kept (so two files of one name in different
    /// directories of a repo don't collide).
    pub fn dest(&self, dir: &Path) -> PathBuf {
        let mut d = dir.join(&self.owner).join(&self.repo);
        d.extend(&self.path);
        d
    }
}

/// Whether `file` is one part of a split GGUF (`name-00001-of-00003.gguf`).
fn is_split_part(file: &str) -> bool {
    let stem = &file[..file.len() - ".gguf".len()];
    let b = stem.as_bytes();
    // "-NNNNN-of-NNNNN" at the end.
    b.len() >= 15 && {
        let t = &b[b.len() - 15..];
        t[0] == b'-'
            && t[1..6].iter().all(u8::is_ascii_digit)
            && &t[6..10] == b"-of-"
            && t[10..].iter().all(u8::is_ascii_digit)
    }
}

impl Alias {
    /// Its file (the table's repo and file always parse: a test checks).
    pub fn hf(&self) -> HfFile {
        HfFile::parse(&format!("{}/{}", self.repo, self.file)).expect("alias table")
    }
}

/// The models directory: `$AXON_MODELS_DIR`, else `$XDG_DATA_HOME/axon/models`,
/// else `~/.local/share/axon/models`; None without any absolute one.
pub fn models_dir() -> Option<PathBuf> {
    models_dir_from(
        std::env::var_os("AXON_MODELS_DIR"),
        std::env::var_os("XDG_DATA_HOME"),
        std::env::var_os("HOME"),
    )
}

/// `models_dir` from those variables' values. Relative or empty ones are
/// ignored (the XDG Base Directory rule).
fn models_dir_from(
    axon: Option<std::ffi::OsString>,
    xdg: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    let abs = |v: Option<std::ffi::OsString>| v.map(PathBuf::from).filter(|p| p.is_absolute());
    if let Some(d) = abs(axon) {
        return Some(d);
    }
    let data = abs(xdg).or_else(|| abs(home).map(|h| h.join(".local").join("share")))?;
    Some(data.join("axon").join("models"))
}

/// A model argument of `gen`, `chat`, `serve` or `info` as a path: as given
/// if it exists; an alias or `OWNER/REPO/FILE.gguf` as its file in the
/// models directory (Err if it isn't pulled yet, with the command that
/// pulls it); a file name as the one pulled file of that name (with or
/// without `.gguf`). Anything else as given, for the loader's own error.
pub fn resolve(arg: &str) -> Result<String, String> {
    resolve_in(arg, models_dir().as_deref())
}

fn resolve_in(arg: &str, dir: Option<&Path>) -> Result<String, String> {
    if arg.is_empty() || Path::new(arg).exists() {
        return Ok(arg.to_string());
    }
    let Some(dir) = dir else {
        return Ok(arg.to_string());
    };
    let pulled = |f: HfFile, how: &str| {
        let d = f.dest(dir);
        match d.is_file() {
            true => Ok(d.to_string_lossy().into_owned()),
            false => Err(format!(
                "no file '{arg}', and as {how} it isn't pulled yet (no {}): axon pull {arg}",
                d.display()
            )),
        }
    };
    if let Some(a) = alias(arg) {
        return pulled(a.hf(), "a name axon pull knows");
    }
    if arg.matches('/').count() >= 2
        && let Ok(f) = HfFile::parse(arg)
    {
        return pulled(f, "a Hugging Face file");
    }
    // A pulled file by name: DIR/OWNER/REPO/[DIRS/]NAME.
    if component(arg, "file", 200).is_ok() {
        let names = [arg.to_string(), format!("{arg}.gguf")];
        let mut found = Vec::new();
        find_named(dir, 0, &names, &mut found);
        match found.len() {
            0 => {}
            1 => return Ok(found[0].to_string_lossy().into_owned()),
            _ => {
                let list: Vec<String> = found.iter().map(|p| p.display().to_string()).collect();
                return Err(format!(
                    "'{arg}' names several pulled files; give the path:\n  {}",
                    list.join("\n  ")
                ));
            }
        }
    }
    Ok(arg.to_string())
}

/// Files named one of `names` under models directory level `depth` (0:
/// the models directory, 2: a repo), in a repo or a directory of one, as
/// deep as `HfFile::parse` allows; into `found`.
fn find_named(d: &Path, depth: usize, names: &[String], found: &mut Vec<PathBuf>) {
    if depth >= 2 {
        for n in names {
            let p = d.join(n);
            if p.is_file() {
                found.push(p);
            }
        }
    }
    // A file's path has at most 6 components (`parse`): 5 directories.
    if depth < 7 {
        for sub in read_dirs(d) {
            find_named(&sub, depth + 1, names, found);
        }
    }
}

/// The subdirectories of `d` (none if it can't be read).
fn read_dirs(d: &Path) -> Vec<PathBuf> {
    let Ok(rd) = std::fs::read_dir(d) else {
        return Vec::new();
    };
    let mut v: Vec<PathBuf> = rd
        .flatten()
        .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
        .map(|e| e.path())
        .collect();
    v.sort();
    v
}

/// What a download must produce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Want {
    pub url: String,
    /// Lowercase hex.
    pub sha256: String,
    pub size: u64,
}

/// What a file's URL told about it (`parse_head`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Head {
    pub sha256: String,
    pub size: u64,
    pub commit: String,
}

/// Whether `s` is `n` hex digits.
fn is_hex(s: &str, n: usize) -> bool {
    s.len() == n && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The checksum, size and commit of a Hugging Face file from the response
/// headers of its URL (`Curl::head`: `curl -I -L`, every response's headers
/// in order: a renamed repo's 307 to its new name, Hugging Face's 302 to its
/// CDN for an LFS file, the CDN's answer). The first of Hugging Face's
/// responses that names the file (`x-linked-etag`: the LFS oid, its
/// SHA-256) gives it, with that response's `x-linked-size` and
/// `x-repo-commit`. Err for a refusal before it (401/403: a gated or
/// private repo; 404) or headers that don't say.
pub fn parse_head(text: &str) -> Result<Head, String> {
    // Each response: its status and headers (lowercase names).
    type Response = (Option<u32>, Vec<(String, String)>);
    let mut resps: Vec<Response> = Vec::new();
    for line in text.lines() {
        let line = line.trim_end_matches('\r');
        if line.starts_with("HTTP/") {
            let code = line.split_whitespace().nth(1).and_then(|c| c.parse().ok());
            resps.push((code, Vec::new()));
        } else if let (Some(r), Some((k, v))) = (resps.last_mut(), line.split_once(':')) {
            r.1.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let mut last = None;
    for (code, hs) in &resps {
        match code {
            Some(401 | 403) => {
                return Err(
                    "Hugging Face refused it (401/403): a gated or private repo; accept its \
                            license on the website and set HF_TOKEN to an access token"
                        .into(),
                );
            }
            Some(404) => return Err("Hugging Face has no such file (404)".into()),
            Some(200..=399) => {}
            Some(c) => return Err(format!("Hugging Face answered HTTP {c}")),
            None => return Err("a malformed HTTP status line".into()),
        }
        last = *code;
        let get = |k: &str| hs.iter().find(|(n, _)| n == k).map(|(_, v)| v.as_str());
        let Some(etag) = get("x-linked-etag") else {
            continue;
        };
        let sha = etag
            .trim_start_matches("W/")
            .trim_matches('"')
            .to_ascii_lowercase();
        let sha = Some(sha)
            .filter(|s| is_hex(s, 64))
            .ok_or("its SHA-256 (x-linked-etag) isn't 64 hex digits")?;
        let size = get("x-linked-size")
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&s| s > 0 && s < 1 << 40)
            .ok_or("no valid size for it (x-linked-size)")?;
        let commit = get("x-repo-commit")
            .map(|c| c.to_ascii_lowercase())
            .filter(|c| is_hex(c, 40))
            .ok_or("no commit for it (x-repo-commit)")?;
        return Ok(Head {
            sha256: sha,
            size,
            commit,
        });
    }
    match last {
        None => Err("no HTTP response".into()),
        Some(_) => Err("no SHA-256 for it (x-linked-etag): not a Git LFS file".into()),
    }
}

/// Where download bytes come from: a stream of the file from `offset` on.
pub trait Source {
    /// The bytes of `url` from `offset`.
    fn open(&self, url: &str, offset: u64) -> Result<Box<dyn Stream>, Fetch>;

    /// Wait before retry number `attempt` (1, 2, ...).
    fn pause(&self, _attempt: usize) {}
}

/// An open download.
pub trait Stream: Read {
    /// After the bytes end: Ok, or why the transfer failed.
    fn finish(self: Box<Self>) -> Result<(), Fetch>;
}

/// Why a transfer failed.
#[derive(Debug)]
pub enum Fetch {
    /// The server can't resume at the offset asked: start over from 0.
    Restart(String),
    /// Anything else (retried, resuming, a few times).
    Failed(String),
}

/// `curl`, HTTPS only, following redirects (to Hugging Face's CDN) only to
/// HTTPS; `HF_TOKEN`, if set, as a bearer token through curl's stdin (not
/// its command line), which curl (7.58+: `curl_ok`) doesn't send on to
/// another host. `-q` first: no `~/.curlrc` (its `insecure`,
/// `location-trusted` or `trace` would undo all that).
pub struct Curl;

/// The oldest curl `axon pull` sends `HF_TOKEN` through: 7.58.0 stopped
/// sending a custom Authorization header on to another host on a redirect
/// (CVE-2018-1000007).
const CURL_TOKEN_MIN: (u32, u32, u32) = (7, 58, 0);

/// The version in `curl --version`'s first line (`curl 8.5.0 (x86_64...`).
fn curl_version(text: &str) -> Option<(u32, u32, u32)> {
    let v = text.lines().next()?.strip_prefix("curl ")?;
    let v = v.split([' ', '-']).next()?;
    let mut n = v.split('.').map(|x| x.parse::<u32>().ok());
    Some((n.next()??, n.next()??, n.next().flatten().unwrap_or(0)))
}

/// Err when `HF_TOKEN` would go through a curl older than `CURL_TOKEN_MIN`
/// (checked once).
fn curl_ok() -> Result<(), String> {
    static OK: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    OK.get_or_init(|| {
        let out = std::process::Command::new("curl")
            .args(["-q", "--version"])
            .output()
            .map_err(|e| format!("curl: {e}"))?;
        let text = String::from_utf8_lossy(&out.stdout);
        match curl_version(&text) {
            Some(v) if v >= CURL_TOKEN_MIN => Ok(()),
            v => Err(format!(
                "HF_TOKEN is set, and this curl ({}) is older than 7.58, which sends it on to the \
                 host a redirect names: update curl, or unset HF_TOKEN for a public repo",
                v.map_or("version unknown".into(), |(a, b, c)| format!("{a}.{b}.{c}"))
            )),
        }
    })
    .clone()
}

impl Curl {
    fn command(extra: &[&str], url: &str) -> std::process::Command {
        let mut c = std::process::Command::new("curl");
        c.args([
            // First, or curl reads ~/.curlrc.
            "-q",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--connect-timeout",
            "30",
            "--user-agent",
            concat!("axon/", env!("CARGO_PKG_VERSION")),
            "--config",
            "-",
        ])
        .args(extra)
        .arg("--")
        .arg(url)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
        c
    }

    /// Start `c`, give it its config (the token) on stdin.
    fn spawn(mut c: std::process::Command) -> Result<std::process::Child, String> {
        let mut child = c.spawn().map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                "curl not found: axon pull needs curl on the PATH".into()
            }
            _ => format!("curl: {e}"),
        })?;
        let mut stdin = child.stdin.take().expect("piped");
        if let Ok(t) = std::env::var("HF_TOKEN")
            && !t.is_empty()
        {
            if let Err(e) = curl_ok() {
                drop(stdin);
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
            // Only a token's characters: nothing that could end the quoted
            // value in curl's config syntax.
            match t
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-.".contains(&b))
            {
                true => {
                    let _ = writeln!(stdin, "header = \"Authorization: Bearer {t}\"");
                }
                false => eprintln!(
                    "axon pull: HF_TOKEN ignored: not an access token (letters, digits, '_', '-', '.')"
                ),
            }
        }
        drop(stdin);
        Ok(child)
    }

    /// The response headers of `url`, redirects followed (`parse_head`'s
    /// input).
    pub fn head(url: &str) -> Result<String, String> {
        let child = Self::spawn(Self::command(
            &[
                "--head",
                "--location",
                "--max-redirs",
                "5",
                "--max-time",
                "60",
            ],
            url,
        ))?;
        let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "curl {url}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    }
}

/// A running curl's stdout.
struct CurlStream {
    child: std::process::Child,
    out: std::process::ChildStdout,
}

impl Read for CurlStream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.out.read(buf)
    }
}

impl Stream for CurlStream {
    fn finish(mut self: Box<Self>) -> Result<(), Fetch> {
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            let _ = e.read_to_string(&mut err);
        }
        let st = self
            .child
            .wait()
            .map_err(|e| Fetch::Failed(format!("curl: {e}")))?;
        let why = format!("curl failed ({st}): {}", err.trim());
        match st.code() {
            Some(0) => Ok(()),
            // CURLE_RANGE_ERROR: the server doesn't do byte ranges.
            Some(33) => Err(Fetch::Restart(why)),
            _ => Err(Fetch::Failed(why)),
        }
    }
}

impl Drop for CurlStream {
    fn drop(&mut self) {
        // An abandoned transfer: stop curl (no-op once it has exited).
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Source for Curl {
    fn open(&self, url: &str, offset: u64) -> Result<Box<dyn Stream>, Fetch> {
        let off = offset.to_string();
        // No --retry: curl can't take back bytes it wrote to the pipe;
        // `fetch` retries, resuming where the bytes stopped.
        let mut extra = vec![
            "--fail",
            "--location",
            // A transfer stalled under 1 KiB/s for a minute is dropped (and resumed).
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
        ];
        if offset > 0 {
            extra.extend(["--continue-at", off.as_str()]);
        }
        let mut child = Curl::spawn(Curl::command(&extra, url)).map_err(Fetch::Failed)?;
        let out = child.stdout.take().expect("piped");
        Ok(Box::new(CurlStream { child, out }))
    }

    fn pause(&self, attempt: usize) {
        std::thread::sleep(std::time::Duration::from_secs(1 << attempt.min(4)));
    }
}

/// The CLI's progress line on stderr: a carriage-return line at most 4
/// times a second on a terminal, else a line every 10 %.
struct Progress {
    what: String,
    total: u64,
    tty: bool,
    t0: std::time::Instant,
    start: u64,
    last: Option<std::time::Instant>,
    last_tenth: u64,
    verb: String,
}

/// A size for people: GB from 1 GB up, else MB.
fn gb(b: u64) -> String {
    match b >= 1_000_000_000 {
        true => format!("{:.2} GB", b as f64 / 1e9),
        false => format!("{:.1} MB", b as f64 / 1e6),
    }
}

impl Progress {
    fn new(what: &str, total: u64) -> Progress {
        use std::io::IsTerminal;
        Progress {
            what: what.into(),
            total,
            tty: std::io::stderr().is_terminal(),
            t0: std::time::Instant::now(),
            start: 0,
            last: None,
            last_tenth: 0,
            verb: String::new(),
        }
    }

    fn show(&mut self, done: u64, verb: &str, force: bool) {
        let now = std::time::Instant::now();
        if verb != self.verb {
            // Checking a partial download, then pulling: each its own rate.
            (self.verb, self.t0, self.start, self.last_tenth) = (verb.into(), now, done, 0);
            if self.tty && self.last.is_some() {
                eprintln!();
            }
        }
        let pct = done.saturating_mul(100) / self.total.max(1);
        let secs = self.t0.elapsed().as_secs_f64().max(1e-3);
        let rate = done.saturating_sub(self.start) as f64 / secs;
        let eta = match rate > 0.0 {
            true => {
                let s = (self.total.saturating_sub(done)) as f64 / rate;
                format!(", {}m{:02}s left", (s / 60.0) as u64, (s % 60.0) as u64)
            }
            false => String::new(),
        };
        let line = format!(
            "{verb} {}: {} / {} ({pct} %), {:.1} MB/s{eta}",
            self.what,
            gb(done),
            gb(self.total),
            rate / 1e6
        );
        if self.tty {
            if force
                || self
                    .last
                    .is_none_or(|t| now - t >= std::time::Duration::from_millis(250))
            {
                eprint!("\r\x1b[2K{line}");
                self.last = Some(now);
            }
        } else if pct / 10 > self.last_tenth || force {
            self.last_tenth = pct / 10;
            eprintln!("{line}");
        }
    }

    fn end(&self) {
        if self.tty {
            eprintln!();
        }
    }
}

/// Download `want` into `dest` through `src`: into `dest` + ".part"
/// (locked against a second pull of it, until it's renamed), resumed from
/// whatever it holds if `dest` + ".part.want" says it is this download (its
/// SHA-256 and size; else it starts over: an older version of the file),
/// hashed as it's written; renamed to `dest` once `want.size` bytes hash
/// to `want.sha256`. `say` hears progress (bytes so far, a verb). A
/// mismatch deletes the `.part`: the next pull starts over.
pub fn fetch(
    want: &Want,
    dest: &Path,
    src: &dyn Source,
    say: &mut dyn FnMut(u64, &str),
) -> Result<(), String> {
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let side = PathBuf::from(format!("{}.part.want", dest.display()));
    let ctx = |e: std::io::Error| format!("{}: {e}", part.display());
    let drop_part = || {
        let _ = std::fs::remove_file(&part);
        let _ = std::fs::remove_file(&side);
    };
    if let Some(d) = dest.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&part)
        .map_err(ctx)?;
    match f.try_lock() {
        Ok(()) => {}
        Err(std::fs::TryLockError::WouldBlock) => {
            return Err(format!(
                "{} is being downloaded by another axon pull",
                dest.display()
            ));
        }
        Err(std::fs::TryLockError::Error(e)) => return Err(ctx(e)),
    }
    // Another pull may have finished (renamed the `.part` away) between our
    // open and our lock: then the file we hold isn't the `.part` any more.
    if !same_file(&f, &part) {
        return Err(format!(
            "{} was just downloaded by another axon pull: run axon pull again to check it",
            dest.display()
        ));
    }
    // The `.part` is this download's only if its sidecar says so.
    let wanted = format!("{} {}\n", want.sha256, want.size);
    if std::fs::read_to_string(&side).ok().as_deref() != Some(wanted.as_str()) {
        f.set_len(0).map_err(ctx)?;
        std::fs::write(&side, &wanted).map_err(|e| format!("{}: {e}", side.display()))?;
    }
    let mut have = f.metadata().map_err(ctx)?.len();
    if have > want.size {
        f.set_len(0).map_err(ctx)?;
        have = 0;
    }
    // What the .part already holds, hashed again.
    let mut h = Sha256::new();
    if have > 0 {
        f.seek(std::io::SeekFrom::Start(0)).map_err(ctx)?;
        let mut r = (&mut f).take(have);
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = r.read(&mut buf).map_err(ctx)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
            say(h.len(), "checking");
        }
        if h.len() != have {
            return Err(format!("{}: shorter than it was", part.display()));
        }
    }
    f.seek(std::io::SeekFrom::Start(have)).map_err(ctx)?;
    let mut failures: Vec<String> = Vec::new();
    while h.len() < want.size {
        if failures.len() >= 5 {
            return Err(format!(
                "download failed ({} kept, the next pull resumes it): {}",
                part.display(),
                failures.join("; ")
            ));
        }
        if !failures.is_empty() {
            src.pause(failures.len());
        }
        let at = h.len();
        let (read_err, end) = match src.open(&want.url, at) {
            Err(e) => (None, Err(e)),
            Ok(mut s) => {
                let mut buf = vec![0u8; 1 << 20];
                let read_err = loop {
                    let n = match s.read(&mut buf) {
                        Ok(0) => break None,
                        Ok(n) => n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => break Some(e.to_string()),
                    };
                    if h.len() + n as u64 > want.size {
                        drop(s);
                        drop(f);
                        drop_part();
                        return Err(format!(
                            "the server sent more than the {} bytes expected: not this file (the \
                             partial download was deleted)",
                            want.size
                        ));
                    }
                    f.write_all(&buf[..n]).map_err(ctx)?;
                    h.update(&buf[..n]);
                    say(h.len(), "pulling");
                };
                (read_err, s.finish())
            }
        };
        match (read_err, end) {
            (None, Ok(())) if h.len() < want.size => {
                failures.push(format!("the transfer ended at byte {}", h.len()))
            }
            (None, Ok(())) => {}
            // The server can't start at `at` and sent nothing: start over.
            (_, Err(Fetch::Restart(e))) if at > 0 && h.len() == at => {
                failures.push(format!("can't resume ({e}): starting over"));
                f.set_len(0).map_err(ctx)?;
                f.seek(std::io::SeekFrom::Start(0)).map_err(ctx)?;
                h = Sha256::new();
            }
            (Some(e), _) | (None, Err(Fetch::Restart(e) | Fetch::Failed(e))) => failures.push(e),
        }
    }
    let got = h.finish_hex();
    if got != want.sha256 {
        drop(f);
        drop_part();
        return Err(format!(
            "checksum mismatch: SHA-256 {got}, expected {} (the download was deleted; pull again)",
            want.sha256
        ));
    }
    f.sync_all().map_err(ctx)?;
    // Renamed with the lock still held, so no second pull takes the `.part`
    // between the check and the rename.
    std::fs::rename(&part, dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    drop(f);
    let _ = std::fs::remove_file(&side);
    if let Some(d) = dest.parent()
        && let Ok(dir) = std::fs::File::open(d)
    {
        let _ = dir.sync_all();
    }
    Ok(())
}

/// Whether open file `f` is the file at `path` (same device and inode).
fn same_file(f: &std::fs::File, path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        match (f.metadata(), std::fs::metadata(path)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
            _ => false,
        }
    }
    #[cfg(not(unix))]
    {
        let _ = f;
        path.is_file()
    }
}

/// Whether `path` holds `want`'s bytes (size, then SHA-256).
fn verified(path: &Path, want: &Want, say: &mut dyn FnMut(u64, &str)) -> Result<bool, String> {
    let Ok(md) = std::fs::metadata(path) else {
        return Ok(false);
    };
    if md.len() != want.size {
        return Ok(false);
    }
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        say(h.len(), "checking");
    }
    Ok(h.finish_hex() == want.sha256)
}

const USAGE: &str = "usage:
  axon pull NAME | OWNER/REPO/FILE.gguf   download a GGUF from Hugging Face into the models
                                          directory, checked against its SHA-256; resumes an
                                          interrupted pull (needs curl)
  axon pull --list                        the names, and which are pulled
models directory: $AXON_MODELS_DIR, else $XDG_DATA_HOME/axon/models, else
~/.local/share/axon/models. HF_TOKEN: an access token for gated repos.";

/// `axon pull ARGS...` (`args`: what follows `pull`).
pub fn pull_cmd(args: &[String]) -> Result<(), String> {
    let dir = models_dir().ok_or("no models directory: set AXON_MODELS_DIR or HOME")?;
    match args {
        [a] if a == "--list" => {
            println!("models directory: {}", dir.display());
            for a in ALIASES {
                let pulled = a.hf().dest(&dir).is_file();
                println!(
                    "  {:<20} {:>8}  {}{}",
                    a.name,
                    gb(a.size),
                    a.what,
                    if pulled { "  [pulled]" } else { "" }
                );
            }
            Ok(())
        }
        [a] if a == "-h" || a == "--help" => {
            println!("{USAGE}");
            Ok(())
        }
        [spec] if !spec.starts_with('-') => pull_one(spec, &dir),
        _ => Err(USAGE.into()),
    }
}

/// Pull `spec` (a name or a Hugging Face file) into `dir`.
fn pull_one(spec: &str, dir: &Path) -> Result<(), String> {
    let (f, want) = match alias(spec) {
        Some(a) => {
            let f = a.hf();
            let w = Want {
                url: f.url(a.rev),
                sha256: a.sha256.into(),
                size: a.size,
            };
            (f, w)
        }
        None => {
            let f = HfFile::parse(spec)?;
            let head = Curl::head(&f.url("main"))
                .and_then(|h| parse_head(&h))
                .map_err(|e| format!("{}: {e}", f.repo_id()))?;
            let w = Want {
                // The commit it was checked at: the bytes can't move under us.
                url: f.url(&head.commit),
                sha256: head.sha256,
                size: head.size,
            };
            (f, w)
        }
    };
    let dest = f.dest(dir);
    let mut p = Progress::new(f.file_name(), want.size);
    let mut say = |n: u64, verb: &str| p.show(n, verb, false);
    if verified(&dest, &want, &mut say)? {
        p.end();
        println!("{}: already pulled, SHA-256 checked", dest.display());
        return Ok(());
    }
    eprintln!(
        "axon pull: {} ({}) from {}",
        f.file_name(),
        gb(want.size),
        f.repo_id()
    );
    let r = fetch(&want, &dest, &Curl, &mut say);
    p.end();
    r?;
    println!(
        "{}: pulled, SHA-256 {} checked",
        dest.display(),
        want.sha256
    );
    if alias(spec).is_some() {
        println!("run it: axon chat {spec}   (or axon serve {spec})");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pull_specs_parse() {
        let f = HfFile::parse("Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf").unwrap();
        assert_eq!(f.repo_id(), "Qwen/Qwen3-8B-GGUF");
        assert_eq!(f.file_name(), "Qwen3-8B-Q4_K_M.gguf");
        assert_eq!(
            f.url("main"),
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf"
        );
        assert_eq!(
            f.dest(Path::new("/m")),
            Path::new("/m/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
        );
        for s in [
            "hf.co/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            "huggingface.co/Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            "https://huggingface.co/Qwen/Qwen3-8B-GGUF/blob/main/Qwen3-8B-Q4_K_M.gguf",
            "https://hf.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
            "hf.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
        ] {
            assert_eq!(HfFile::parse(s).unwrap(), f, "{s}");
        }
        let d = HfFile::parse("o/r/Q4_K_M/m.gguf").unwrap();
        assert_eq!(d.path, ["Q4_K_M", "m.gguf"]);
        assert_eq!(
            d.url("abc"),
            "https://huggingface.co/o/r/resolve/abc/Q4_K_M/m.gguf"
        );
        // The repo's directories are kept: two files of one name don't collide.
        assert_eq!(d.dest(Path::new("/m")), Path::new("/m/o/r/Q4_K_M/m.gguf"));
        let e = HfFile::parse("o/r/Q5_K_M/m.gguf").unwrap();
        assert_ne!(d.dest(Path::new("/m")), e.dest(Path::new("/m")));
        assert!(HfFile::parse("o/r/M.GGUF").is_ok());
    }

    /// Hostile names never parse: traversal, absolute and empty parts,
    /// hidden files, options, separators and escapes, non-ASCII, NUL,
    /// over-long parts, non-GGUF and split files.
    #[test]
    fn pull_specs_refuse_traversal() {
        let long = "a".repeat(201);
        let bad = [
            "",
            "a",
            "a/b",
            "a/b/c",
            "a/b/c.bin",
            "../b/c.gguf",
            "a/../c.gguf",
            "a/b/../c.gguf",
            "a/b/../../c.gguf",
            "a/b/./c.gguf",
            "a/b/.c.gguf",
            "/a/b/c.gguf",
            "a//b/c.gguf",
            "a/b//c.gguf",
            "a/b/c.gguf/",
            "a\\b/c/d.gguf",
            "a/b/..\\c.gguf",
            "a/b/c%2f..%2fd.gguf",
            "a/b/%2e%2e/c.gguf",
            "-a/b/c.gguf",
            "a/-b/c.gguf",
            "a/b/-c.gguf",
            "a/b/--output=x.gguf",
            "a/b/c d.gguf",
            "a/b/c\0.gguf",
            "a/b/c\n.gguf",
            "а/b/c.gguf",
            "a/b/ç.gguf",
            "a/b/c.gguf.",
            "a/b/c..",
            "~/b/c.gguf",
            "a/b/$HOME.gguf",
            "a/b/`x`.gguf",
            "file:///etc/passwd.gguf",
            "http://huggingface.co/a/b/c.gguf",
            "https://evil.com/a/b/c.gguf",
            "https://huggingface.co/a/b/resolve/refs/pr/1/c.gguf",
            "https://huggingface.co/a/b/c.gguf",
            "https://www.huggingface.co/a/b/resolve/main/c.gguf",
            "https://huggingface.co.evil.com/a/b/resolve/main/c.gguf",
            "a/b/c-00001-of-00003.gguf",
            &format!("a/b/{long}.gguf"),
            &format!("{}/b/c.gguf", "o".repeat(97)),
            "a/b/1/2/3/4/5/6/c.gguf",
        ];
        for s in bad {
            assert!(HfFile::parse(s).is_err(), "{:?} parsed", s);
        }
        // Some of those are near-misses of a good one.
        assert!(HfFile::parse("a/b/c-00001-of-0003.gguf").is_ok());
        assert!(HfFile::parse("a/b/c.d.gguf").is_ok());
    }

    /// Random strings over the dangerous characters: whatever parses lands
    /// strictly under the models directory, each part a plain component of
    /// `component`'s alphabet, and its URL is Hugging Face's.
    #[test]
    fn pull_spec_fuzz_stays_in_the_models_dir() {
        let alphabet: Vec<char> = "ab0Z/._-\\%: ~$\0\n\u{430}gfu".chars().collect();
        let tails = [
            ".gguf",
            ".GGUF",
            "",
            "/x.gguf",
            "/../x.gguf",
            "-00001-of-00002.gguf",
        ];
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let dir = Path::new("/models");
        let mut ok = 0;
        for _ in 0..50_000 {
            let n = 1 + rnd() % 24;
            let mut s: String = (0..n)
                .map(|_| alphabet[(rnd() % alphabet.len() as u64) as usize])
                .collect();
            s.push_str(tails[(rnd() % tails.len() as u64) as usize]);
            let Ok(f) = HfFile::parse(&s) else { continue };
            ok += 1;
            let d = f.dest(dir);
            let rel = d.strip_prefix(dir).unwrap();
            let parts: Vec<_> = rel.components().collect();
            assert_eq!(parts.len(), 2 + f.path.len(), "{s:?} -> {d:?}");
            for c in parts {
                let std::path::Component::Normal(c) = c else {
                    panic!("{s:?} -> {d:?}")
                };
                component(c.to_str().unwrap(), "x", 200).unwrap();
            }
            assert!(f.url("main").starts_with("https://huggingface.co/"));
            assert!(!f.url("main").contains(".."), "{s:?}");
        }
        assert!(ok > 0, "some random specs are valid");
    }

    #[test]
    fn pull_alias_table() {
        let mut names = std::collections::HashSet::new();
        for a in ALIASES {
            assert!(names.insert(a.name), "{} twice", a.name);
            assert!(
                a.name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b"-.".contains(&b)),
                "{}",
                a.name
            );
            assert!(
                is_hex(a.sha256, 64) && a.sha256 == a.sha256.to_lowercase(),
                "{}",
                a.name
            );
            assert!(is_hex(a.rev, 40), "{}", a.name);
            assert!(a.size > 100 << 20, "{}", a.name);
            let f = a.hf();
            assert_eq!(f.file_name(), a.file);
            assert!(f.url(a.rev).contains(a.rev));
            // A name is never a file spec, nor a pulled file's name.
            assert!(HfFile::parse(a.name).is_err());
        }
        assert!((10..=14).contains(&ALIASES.len()));
        assert!(alias("qwen3-8b").is_some() && alias("nope").is_none());
    }

    #[test]
    fn pull_models_dir_rules() {
        let s = |v: &str| Some(std::ffi::OsString::from(v));
        assert_eq!(
            models_dir_from(s("/x"), s("/d"), s("/h")),
            Some("/x".into())
        );
        assert_eq!(
            models_dir_from(None, s("/d"), s("/h")),
            Some("/d/axon/models".into())
        );
        assert_eq!(
            models_dir_from(s("rel"), s(""), s("/h")),
            Some("/h/.local/share/axon/models".into())
        );
        assert_eq!(models_dir_from(None, None, s("rel")), None);
    }

    /// A scratch directory (small files), removed on drop.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Scratch {
            let d = std::env::temp_dir().join(format!("axon-pull-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Scratch(d)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn pull_resolve_names() {
        let t = Scratch::new("resolve");
        let dir = t.0.join("models");
        let a = alias("qwen3-8b").unwrap();
        // Not pulled: the command that pulls it.
        let e = resolve_in("qwen3-8b", Some(&dir)).unwrap_err();
        assert!(e.contains("axon pull qwen3-8b"), "{e}");
        let e = resolve_in("o/r/m.gguf", Some(&dir)).unwrap_err();
        assert!(e.contains("axon pull o/r/m.gguf"), "{e}");
        let p = a.hf().dest(&dir);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, b"x").unwrap();
        let ps = p.to_string_lossy().into_owned();
        assert_eq!(resolve_in("qwen3-8b", Some(&dir)).unwrap(), ps);
        assert_eq!(
            resolve_in(&format!("{}/{}", a.repo, a.file), Some(&dir)).unwrap(),
            ps
        );
        assert_eq!(resolve_in(a.file, Some(&dir)).unwrap(), ps);
        assert_eq!(resolve_in("Qwen3-8B-Q4_K_M", Some(&dir)).unwrap(), ps);
        // A path that exists wins; anything else is left for the loader.
        assert_eq!(resolve_in(&ps, None).unwrap(), ps);
        assert_eq!(resolve_in("nope.gguf", Some(&dir)).unwrap(), "nope.gguf");
        assert_eq!(resolve_in("../x", Some(&dir)).unwrap(), "../x");
        assert_eq!(resolve_in("qwen3-8b", None).unwrap(), "qwen3-8b");
        // A file in a directory of a repo is found by name too.
        let sub = HfFile::parse("o/r/Q8_0/deep-q8.gguf").unwrap().dest(&dir);
        std::fs::create_dir_all(sub.parent().unwrap()).unwrap();
        std::fs::write(&sub, b"z").unwrap();
        assert_eq!(
            resolve_in("deep-q8", Some(&dir)).unwrap(),
            sub.to_string_lossy()
        );
        // Two pulled files of one name: ambiguous.
        let q = dir.join("other/repo").join(a.file);
        std::fs::create_dir_all(q.parent().unwrap()).unwrap();
        std::fs::write(&q, b"y").unwrap();
        assert!(
            resolve_in(a.file, Some(&dir))
                .unwrap_err()
                .contains("several")
        );
    }

    #[test]
    fn pull_head_parses() {
        let ok = "HTTP/2 302 \r\ncontent-type: text/plain\r\nx-repo-commit: 7C41481F57CB95916B40956AB2F0B139B296D974\r\n\
                  x-linked-size: 5027783488\r\nx-linked-etag: \"d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785\"\r\n\
                  location: https://cdn.example/x\r\n\r\n";
        assert_eq!(
            parse_head(ok).unwrap(),
            Head {
                sha256: "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785".into(),
                size: 5_027_783_488,
                commit: "7c41481f57cb95916b40956ab2f0b139b296d974".into(),
            }
        );
        assert!(
            parse_head("HTTP/2 401\r\n\r\n")
                .unwrap_err()
                .contains("HF_TOKEN")
        );
        assert!(
            parse_head("HTTP/1.1 404 Not Found\r\n\r\n")
                .unwrap_err()
                .contains("404")
        );
        assert!(parse_head("HTTP/2 500\r\n").unwrap_err().contains("500"));
        assert!(parse_head("").is_err());
        let bad = [
            ok.replace("d98cdcbd", "d98cdcb"),
            ok.replace("d98cdcbd", "x98cdcbd"),
            ok.replace("5027783488", "-1"),
            ok.replace("5027783488", "99999999999999999999"),
            ok.replace("5027783488", "0"),
            ok.replace("7C41481F", "7C41481"),
            ok.replace("x-linked-etag", "etag"),
        ];
        for b in bad {
            assert!(parse_head(&b).is_err(), "{b}");
        }
        // The first response that names the file wins (the CDN's after it
        // can't override it).
        let two = format!("{ok}HTTP/2 200\r\nx-linked-size: 1\r\n\r\n");
        assert_eq!(parse_head(&two).unwrap().size, 5_027_783_488);
        // A renamed repo: its 307 first, then the file's 302, then the CDN.
        let moved = format!(
            "HTTP/2 307 \r\nlocation: /o/r-moved/resolve/main/m.gguf\r\n\r\n{ok}HTTP/2 200 \r\n\
             content-length: 5027783488\r\n\r\n"
        );
        assert_eq!(parse_head(&moved).unwrap(), parse_head(ok).unwrap());
        // ... but a refusal on the way is one.
        let gated = "HTTP/2 307 \r\nlocation: /x\r\n\r\nHTTP/2 401 \r\n\r\n";
        assert!(parse_head(gated).unwrap_err().contains("HF_TOKEN"));
        // A 200 without the LFS headers: not an LFS file.
        let plain = "HTTP/2 200 \r\ncontent-length: 12\r\n\r\n";
        assert!(
            parse_head(plain)
                .unwrap_err()
                .contains("not a Git LFS file")
        );
    }

    /// A file-backed `Source`: serves `data` from the offset asked, failing
    /// after `cut` bytes on the first `fail` opens, or refusing to resume.
    struct Fake {
        data: Vec<u8>,
        cut: usize,
        fail: std::cell::Cell<usize>,
        no_resume: bool,
        opens: std::cell::RefCell<Vec<u64>>,
    }

    struct FakeStream {
        data: Vec<u8>,
        at: usize,
        stop: usize,
        err: bool,
    }

    impl Read for FakeStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if self.at >= self.stop {
                return match self.err {
                    true => Err(std::io::Error::other("connection reset")),
                    false => Ok(0),
                };
            }
            let n = buf.len().min(self.stop - self.at).min(7919);
            buf[..n].copy_from_slice(&self.data[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    impl Stream for FakeStream {
        fn finish(self: Box<Self>) -> Result<(), Fetch> {
            Ok(())
        }
    }

    impl Source for Fake {
        fn open(&self, _url: &str, offset: u64) -> Result<Box<dyn Stream>, Fetch> {
            self.opens.borrow_mut().push(offset);
            if offset > 0 && self.no_resume {
                return Err(Fetch::Restart("no ranges".into()));
            }
            let at = offset as usize;
            let failing = self.fail.get() > 0;
            if failing {
                self.fail.set(self.fail.get() - 1);
            }
            let stop = match failing {
                true => (at + self.cut).min(self.data.len()),
                false => self.data.len(),
            };
            Ok(Box::new(FakeStream {
                data: self.data.clone(),
                at,
                stop,
                err: failing,
            }))
        }
    }

    fn fake(data: &[u8], cut: usize, fail: usize) -> Fake {
        Fake {
            data: data.to_vec(),
            cut,
            fail: std::cell::Cell::new(fail),
            no_resume: false,
            opens: std::cell::RefCell::new(Vec::new()),
        }
    }

    /// `fetch`: a clean download lands verified; a dropped connection
    /// resumes where it stopped; a `.part` from an earlier run is resumed;
    /// a server that can't resume starts over; a damaged `.part` or a wrong
    /// checksum is a mismatch that deletes it; too many bytes or a `.part`
    /// longer than the file are caught; five failures in a row keep the
    /// `.part` for the next pull.
    #[test]
    fn pull_fetch_resumes_and_verifies() {
        let t = Scratch::new("fetch");
        let data: Vec<u8> = (0..300_000u32)
            .map(|i| (i.wrapping_mul(2654435761) >> 13) as u8)
            .collect();
        let want = Want {
            url: "https://huggingface.co/o/r/resolve/x/m.gguf".into(),
            sha256: crate::sha256::digest_hex(&data),
            size: data.len() as u64,
        };
        let dest = t.0.join("o/r/m.gguf");
        let part = t.0.join("o/r/m.gguf.part");
        let side = t.0.join("o/r/m.gguf.part.want");
        let mut quiet = |_: u64, _: &str| {};
        let reset = || {
            let _ = std::fs::remove_file(&dest);
            let _ = std::fs::remove_file(&part);
            let _ = std::fs::remove_file(&side);
        };
        // A `.part` an earlier pull of this file left, with its sidecar.
        let seed = |bytes: &[u8]| {
            std::fs::write(&part, bytes).unwrap();
            std::fs::write(&side, format!("{} {}\n", want.sha256, want.size)).unwrap();
        };

        let src = fake(&data, 0, 0);
        fetch(&want, &dest, &src, &mut quiet).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        assert!(!part.exists() && !side.exists());
        assert_eq!(*src.opens.borrow(), [0]);

        reset();
        let src = fake(&data, 100_000, 2);
        fetch(&want, &dest, &src, &mut quiet).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);
        assert_eq!(*src.opens.borrow(), [0, 100_000, 200_000]);

        reset();
        seed(&data[..123_457]);
        let src = fake(&data, 0, 0);
        let mut checked = 0;
        fetch(&want, &dest, &src, &mut |n, v| {
            if v == "checking" {
                checked = n
            }
        })
        .unwrap();
        assert_eq!(checked, 123_457);
        assert_eq!(*src.opens.borrow(), [123_457]);
        assert_eq!(std::fs::read(&dest).unwrap(), data);

        reset();
        seed(&data[..5000]);
        let mut src = fake(&data, 0, 0);
        src.no_resume = true;
        fetch(&want, &dest, &src, &mut quiet).unwrap();
        assert_eq!(*src.opens.borrow(), [5000, 0]);
        assert_eq!(std::fs::read(&dest).unwrap(), data);

        reset();
        let mut bad = data[..5000].to_vec();
        bad[17] ^= 1;
        seed(&bad);
        let e = fetch(&want, &dest, &fake(&data, 0, 0), &mut quiet).unwrap_err();
        assert!(e.contains("checksum mismatch"), "{e}");
        assert!(!part.exists() && !dest.exists() && !side.exists());

        reset();
        let mut longer = data.clone();
        longer.push(0);
        let e = fetch(&want, &dest, &fake(&longer, 0, 0), &mut quiet).unwrap_err();
        assert!(e.contains("more than"), "{e}");
        assert!(!part.exists() && !dest.exists() && !side.exists());

        reset();
        seed(&longer);
        fetch(&want, &dest, &fake(&data, 0, 0), &mut quiet).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);

        reset();
        let e = fetch(&want, &dest, &fake(&data, 1000, 99), &mut quiet).unwrap_err();
        assert!(e.contains("kept, the next pull resumes it"), "{e}");
        assert_eq!(std::fs::metadata(&part).unwrap().len(), 5000);
        assert!(side.is_file());
        // ... and the next pull does.
        fetch(&want, &dest, &fake(&data, 0, 0), &mut quiet).unwrap();
        assert_eq!(std::fs::read(&dest).unwrap(), data);

        // A `.part` of another version of the file (its sidecar names another
        // SHA-256 or size), or with no sidecar: started over, not resumed.
        for stale in [
            Some(format!("{} {}\n", "0".repeat(64), want.size)),
            Some(format!("{} {}\n", want.sha256, want.size + 1)),
            None,
        ] {
            reset();
            std::fs::write(&part, &data[..77_777]).unwrap();
            if let Some(l) = &stale {
                std::fs::write(&side, l).unwrap();
            }
            let src = fake(&data, 0, 0);
            fetch(&want, &dest, &src, &mut quiet).unwrap();
            assert_eq!(*src.opens.borrow(), [0], "{stale:?}");
            assert_eq!(std::fs::read(&dest).unwrap(), data);
            assert!(!part.exists() && !side.exists());
        }

        // Already there and intact: `verified` says so without a download.
        assert!(verified(&dest, &want, &mut quiet).unwrap());
        let mut w2 = want.clone();
        w2.sha256 = "0".repeat(64);
        assert!(!verified(&dest, &w2, &mut quiet).unwrap());
    }

    /// `same_file`: an open `.part` renamed away (another pull finished it)
    /// is no longer the file at its path, nor is a new file there.
    #[test]
    fn pull_same_file_sees_a_rename() {
        let t = Scratch::new("same");
        let (a, b) = (t.0.join("m.part"), t.0.join("m.gguf"));
        std::fs::write(&a, b"x").unwrap();
        let f = std::fs::File::open(&a).unwrap();
        assert!(same_file(&f, &a));
        std::fs::rename(&a, &b).unwrap();
        assert!(!same_file(&f, &a), "gone");
        std::fs::write(&a, b"y").unwrap();
        assert!(!same_file(&f, &a), "another file");
    }

    /// curl skips ~/.curlrc only when `-q` is its first argument; the URL
    /// comes after `--`.
    #[test]
    fn pull_curl_ignores_curlrc() {
        let c = Curl::command(
            &["--head"],
            "https://huggingface.co/o/r/resolve/main/m.gguf",
        );
        let args: Vec<&str> = c.get_args().map(|a| a.to_str().unwrap()).collect();
        assert_eq!(args[0], "-q", "{args:?}");
        assert_eq!(args.iter().filter(|a| **a == "-q").count(), 1);
        let n = args.len();
        assert_eq!(
            args[n - 2..],
            ["--", "https://huggingface.co/o/r/resolve/main/m.gguf"]
        );
        assert!(!args.contains(&"--location-trusted") && !args.contains(&"--insecure"));
    }

    /// `curl --version`'s first line, and the floor for sending HF_TOKEN.
    #[test]
    fn pull_curl_version() {
        let v = |s: &str| curl_version(s);
        assert_eq!(
            v("curl 8.5.0 (x86_64-pc-linux-gnu) libcurl/8.5.0\nRelease-Date"),
            Some((8, 5, 0))
        );
        assert_eq!(
            v("curl 7.55.1 (Windows) libcurl/7.55.1 WinSSL"),
            Some((7, 55, 1))
        );
        assert_eq!(v("curl 8.12.0-DEV (x86_64)"), Some((8, 12, 0)));
        assert_eq!(v("curl 7.58 (x)"), Some((7, 58, 0)));
        assert_eq!(v("nope"), None);
        assert_eq!(v(""), None);
        assert!(v("curl 7.55.1 (Windows)").unwrap() < CURL_TOKEN_MIN);
        assert!(v("curl 7.58.0 (Linux)").unwrap() >= CURL_TOKEN_MIN);
    }

    /// Two pulls of one file at once: the second is refused, not interleaved.
    #[test]
    fn pull_part_is_locked() {
        let t = Scratch::new("lock");
        let dest = t.0.join("m.gguf");
        let part = t.0.join("m.gguf.part");
        let f = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&part)
            .unwrap();
        f.lock().unwrap();
        let want = Want {
            url: "https://huggingface.co/o/r/resolve/x/m.gguf".into(),
            sha256: "0".repeat(64),
            size: 10,
        };
        let e = fetch(&want, &dest, &fake(&[0; 10], 0, 0), &mut |_, _| {}).unwrap_err();
        assert!(e.contains("another axon pull"), "{e}");
    }
}
