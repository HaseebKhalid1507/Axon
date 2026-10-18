//! A small HTTP/1.1 server side for `axon serve`: request parsing and
//! response writing over any `Read + Write` stream (a `TcpStream`, or a
//! buffer in the tests). No TLS, no HTTP/2: run it behind a reverse proxy
//! for those.
//!
//! What it does:
//!
//! * Request line + headers up to `MAX_HEAD` bytes and `MAX_HEADERS` lines
//!   (431 past either), CRLF or bare LF line ends, leading empty lines
//!   skipped. Obsolete line folding, a malformed request line or header, a
//!   target that isn't origin-form or absolute-form (`http://host/path`,
//!   reduced to its path): 400. HTTP/1.0 and 1.1 only (505).
//! * Bodies by `Content-Length` (equal duplicates allowed, anything else
//!   400) or `Transfer-Encoding: chunked` (extensions and trailers read and
//!   ignored); both at once is 400 (request smuggling), another transfer
//!   coding 501. At most `MAX_BODY` bytes (413). `Expect: 100-continue` is
//!   answered before the body is read; another expectation is 417.
//! * Keep-alive: HTTP/1.1 unless `Connection: close`, HTTP/1.0 only with
//!   `Connection: keep-alive`. Bytes past a request stay buffered for the
//!   next one, so pipelined requests are answered in order.
//! * A request that starts arriving must be complete within `head_timeout`
//!   (slow-loris; 408); the socket's own read timeout bounds each wait.
//! * Responses: whole (`Content-Length`; headers only for HEAD), or chunked
//!   for streams (SSE); HTTP/1.0 streams are delimited by closing the
//!   connection instead. Every response carries `Date` and the connection's
//!   `extra` headers (CORS).

use std::io::{self, Read, Write};
use std::time::{Duration, Instant};

/// Most bytes of request line + headers.
pub const MAX_HEAD: usize = 16 * 1024;
/// Most header lines.
pub const MAX_HEADERS: usize = 100;
/// Most body bytes (a 32k-token prompt is ~150 KiB of JSON).
pub const MAX_BODY: usize = 1 << 20;

/// A parsed request.
#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// The target's path, without the query string.
    pub path: String,
    /// HTTP/1.x minor version: 0 or 1.
    pub minor: u8,
    /// Names lowercased, values trimmed, in order.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// The connection may carry another request after this one.
    pub keep_alive: bool,
}

impl Request {
    /// The first header named `name` (lowercase).
    #[cfg(test)]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// A request that can't be served: the status to answer with (the
/// connection is closed after it).
#[derive(Debug, PartialEq)]
pub struct HttpError {
    pub status: u16,
    pub msg: &'static str,
}

fn err<T>(status: u16, msg: &'static str) -> Result<T, HttpError> {
    Err(HttpError { status, msg })
}

/// Tells whether the other end has gone away, without blocking or
/// consuming input.
pub trait Peer {
    fn closed(&self) -> bool;
}

impl Peer for std::net::TcpStream {
    fn closed(&self) -> bool {
        if self.set_nonblocking(true).is_err() {
            return true;
        }
        let r = self.peek(&mut [0u8; 1]);
        let _ = self.set_nonblocking(false);
        match r {
            Ok(0) => true,
            Ok(_) => false,
            Err(e) => e.kind() != io::ErrorKind::WouldBlock,
        }
    }
}

/// `Conn::gate`.
pub type Gate = Box<dyn Fn(&Request) -> bool + Send>;

/// One connection: its stream and the bytes read but not yet used.
pub struct Conn<S> {
    pub s: S,
    buf: Vec<u8>,
    /// Longest a request may take to arrive once its first byte has.
    pub head_timeout: Duration,
    /// The response being streamed is chunked (else close-delimited).
    chunked: bool,
    /// Header lines (`Name: value\r\n`) every response gets.
    pub extra: String,
    /// Answering a HEAD request: `respond` sends the headers only.
    pub head_only: bool,
    /// Whether a request that carries a body may send it (--api-key: false
    /// answers 401 right after the head, before a 413, the body never
    /// buffered and the connection then closed; `100 Continue` is never
    /// sent for it).
    pub gate: Option<Gate>,
    /// The path and method of the request `gate` just refused (its route
    /// label; a HEAD's 401 has no body).
    pub refused: Option<(String, String)>,
    /// The path of the request whose head was read last (None: no head
    /// read yet, or it didn't parse): an error answered before routing
    /// (401, 413, a broken body) takes that route's API's shape.
    pub path: Option<String>,
    /// The status of the last response started (0: none yet).
    pub status: u16,
    /// Bytes written so far.
    pub written: u64,
}

/// The status line's reason phrase.
pub fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        417 => "Expectation Failed",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

fn is_tchar(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

/// End of the head (index past its empty line) in `b`, if it's all there.
fn head_end(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    while let Some(p) = b[i..].iter().position(|&c| c == b'\n') {
        let nl = i + p;
        // The line after this newline is empty: "\n\n" or "\n\r\n".
        match b.get(nl + 1) {
            Some(b'\n') => return Some(nl + 2),
            Some(b'\r') if b.get(nl + 2) == Some(&b'\n') => return Some(nl + 3),
            _ => i = nl + 1,
        }
    }
    None
}

/// What the head says: the request without its body, and how the body comes.
enum BodyLen {
    Fixed(usize),
    Chunked,
}

fn parse_head(head: &[u8]) -> Result<(Request, BodyLen, bool), HttpError> {
    let head = std::str::from_utf8(head).or(err(400, "request head is not UTF-8"))?;
    let mut lines = head.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l));
    let line = lines.next().unwrap_or("");
    let mut parts = line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return err(400, "malformed request line");
    };
    if method.is_empty() || !method.bytes().all(is_tchar) {
        return err(400, "malformed method");
    }
    let minor = match version {
        "HTTP/1.1" => 1,
        "HTTP/1.0" => 0,
        v if v.starts_with("HTTP/") => return err(505, "only HTTP/1.0 and HTTP/1.1"),
        _ => return err(400, "malformed request line"),
    };
    if target.bytes().any(|b| b <= b' ' || b == 0x7f) {
        return err(400, "malformed request target");
    }
    // Absolute-form (proxies send it): the path after scheme://authority.
    let target = match ["http://", "https://"].iter().find_map(|p| {
        target
            .get(..p.len())
            .filter(|t| t.eq_ignore_ascii_case(p))
            .map(|_| p.len())
    }) {
        Some(n) => match target[n..].find(['/', '?', '#']) {
            Some(i) if target[n + i..].starts_with('/') => &target[n + i..],
            _ => "/",
        },
        None => target,
    };
    if !target.starts_with('/') {
        return err(400, "malformed request target");
    }
    let path = target.split(['?', '#']).next().unwrap_or("").to_string();

    let mut headers = Vec::new();
    for l in lines {
        if l.is_empty() {
            break;
        }
        if l.starts_with([' ', '\t']) {
            return err(400, "obsolete header line folding");
        }
        let Some((name, value)) = l.split_once(':') else {
            return err(400, "malformed header");
        };
        if name.is_empty() || !name.bytes().all(is_tchar) {
            return err(400, "malformed header name");
        }
        if value.bytes().any(|b| (b < b' ' && b != b'\t') || b == 0x7f) {
            return err(400, "control character in a header");
        }
        if headers.len() == MAX_HEADERS {
            return err(431, "too many headers");
        }
        headers.push((
            name.to_ascii_lowercase(),
            value.trim_matches([' ', '\t']).to_string(),
        ));
    }

    let mut req = Request {
        method: method.to_string(),
        path,
        minor,
        headers,
        body: Vec::new(),
        keep_alive: minor == 1,
    };
    let mut cl: Option<usize> = None;
    let mut te = None;
    let mut cont = false;
    for (n, v) in &req.headers {
        match n.as_str() {
            "content-length" => {
                if v.is_empty() || !v.bytes().all(|b| b.is_ascii_digit()) {
                    return err(400, "malformed Content-Length");
                }
                let n = v.parse::<usize>().unwrap_or(usize::MAX);
                if cl.is_some_and(|c| c != n) {
                    return err(400, "conflicting Content-Length headers");
                }
                cl = Some(n);
            }
            "transfer-encoding" => {
                if te.is_some() || !v.eq_ignore_ascii_case("chunked") {
                    return err(501, "only the chunked transfer coding is supported");
                }
                te = Some(());
            }
            "connection" => {
                for t in v.split(',').map(str::trim) {
                    if t.eq_ignore_ascii_case("close") {
                        req.keep_alive = false;
                    } else if t.eq_ignore_ascii_case("keep-alive") && minor == 0 {
                        req.keep_alive = true;
                    }
                }
            }
            "expect" => {
                if !v.eq_ignore_ascii_case("100-continue") {
                    return err(417, "unsupported expectation");
                }
                cont = minor == 1;
            }
            _ => {}
        }
    }
    let body = match (te, cl) {
        (Some(_), Some(_)) => return err(400, "both Transfer-Encoding and Content-Length"),
        (Some(_), None) if minor == 0 => return err(400, "chunked request in HTTP/1.0"),
        (Some(_), None) => BodyLen::Chunked,
        // Past MAX_BODY: 413 in `read_request`, after `Conn::gate`.
        (None, n) => BodyLen::Fixed(n.unwrap_or(0)),
    };
    let cont = cont && !matches!(body, BodyLen::Fixed(0));
    Ok((req, body, cont))
}

impl<S: Read + Write> Conn<S> {
    pub fn new(s: S) -> Conn<S> {
        Conn {
            s,
            buf: Vec::new(),
            head_timeout: Duration::from_secs(30),
            chunked: false,
            extra: String::new(),
            head_only: false,
            gate: None,
            refused: None,
            path: None,
            status: 0,
            written: 0,
        }
    }

    /// Read more bytes into the buffer; 0 at the end of the stream.
    /// `since`: when the request started arriving, for `head_timeout`.
    fn fill(&mut self, since: Option<Instant>) -> Result<usize, HttpError> {
        if since.is_some_and(|t| t.elapsed() > self.head_timeout) {
            return err(408, "request took too long to arrive");
        }
        let mut tmp = [0u8; 8192];
        loop {
            match self.s.read(&mut tmp) {
                Ok(n) => {
                    self.buf.extend_from_slice(&tmp[..n]);
                    return Ok(n);
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e)
                    if matches!(
                        e.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                    ) =>
                {
                    return err(408, "request took too long to arrive");
                }
                // Reset, etc.: the request can't be completed; nobody to answer.
                Err(_) => return Ok(0),
            }
        }
    }

    /// The next request. Ok(None): the connection ended (or went idle past
    /// the socket's read timeout) between requests. An error should be
    /// answered with its status, then the connection closed.
    pub fn read_request(&mut self) -> Result<Option<Request>, HttpError> {
        // An error below belongs to this request, not the one before.
        self.path = None;
        // Empty lines before a request are allowed (RFC 9112 2.2).
        let mut since = None;
        let end = loop {
            let skip = self
                .buf
                .iter()
                .position(|&b| b != b'\r' && b != b'\n')
                .unwrap_or(self.buf.len());
            self.buf.drain(..skip);
            if !self.buf.is_empty() {
                since.get_or_insert_with(Instant::now);
                if let Some(e) = head_end(&self.buf) {
                    if e > MAX_HEAD {
                        return err(431, "request head too large");
                    }
                    break e;
                }
                if self.buf.len() > MAX_HEAD {
                    return err(431, "request head too large");
                }
                // Not HTTP at all (a TLS ClientHello, binary junk): say so
                // now instead of waiting `head_timeout` for a line end
                // that may never come. Same status as `parse_head` gives.
                let m = self.buf.iter().position(|&b| b == b' ');
                if !self.buf[..m.unwrap_or(self.buf.len())]
                    .iter()
                    .all(|&b| is_tchar(b))
                {
                    return err(400, "malformed request line");
                }
            }
            match self.fill(since) {
                Ok(0) if self.buf.is_empty() => return Ok(None),
                Ok(0) => return err(400, "connection closed mid-request"),
                Ok(_) => {}
                Err(_) if self.buf.is_empty() => return Ok(None),
                Err(e) => return Err(e),
            }
        };
        let (mut req, body, cont) = parse_head(&self.buf[..end])?;
        self.path = Some(req.path.clone());
        self.buf.drain(..end);
        let has_body = !matches!(body, BodyLen::Fixed(0));
        if has_body && self.gate.as_ref().is_some_and(|g| !g(&req)) {
            self.refused = Some((req.path, req.method));
            return err(401, "unauthorized");
        }
        if let BodyLen::Fixed(n) = body
            && n > MAX_BODY
        {
            return err(413, "request body too large");
        }
        if cont && self.buf.is_empty() {
            self.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
                .or(err(400, "connection closed mid-request"))?;
        }
        let since = since.unwrap_or_else(Instant::now);
        match body {
            BodyLen::Fixed(n) => {
                while self.buf.len() < n {
                    if self.fill(Some(since))? == 0 {
                        return err(400, "connection closed mid-body");
                    }
                }
                req.body = self.buf.drain(..n).collect();
            }
            BodyLen::Chunked => req.body = self.read_chunked(since)?,
        }
        Ok(Some(req))
    }

    /// One line (without its line end) of a chunked body.
    fn line(&mut self, since: Instant) -> Result<String, HttpError> {
        loop {
            if let Some(p) = self.buf.iter().position(|&b| b == b'\n') {
                let l: Vec<u8> = self.buf.drain(..=p).collect();
                let l = &l[..p];
                let l = l.strip_suffix(b"\r").unwrap_or(l);
                return String::from_utf8(l.to_vec()).or(err(400, "malformed chunk"));
            }
            if self.buf.len() > 4096 {
                return err(400, "chunk header line too long");
            }
            if self.fill(Some(since))? == 0 {
                return err(400, "connection closed mid-body");
            }
        }
    }

    fn read_chunked(&mut self, since: Instant) -> Result<Vec<u8>, HttpError> {
        let mut body = Vec::new();
        loop {
            let l = self.line(since)?;
            let size = l.split(';').next().unwrap_or("").trim_matches([' ', '\t']);
            if size.is_empty() || size.len() > 8 || !size.bytes().all(|b| b.is_ascii_hexdigit()) {
                return err(400, "malformed chunk size");
            }
            let n = usize::from_str_radix(size, 16).or(err(400, "malformed chunk size"))?;
            if n == 0 {
                // Trailers, up to the empty line.
                let mut total = 0;
                loop {
                    let t = self.line(since)?;
                    if t.is_empty() {
                        return Ok(body);
                    }
                    total += t.len();
                    if total > MAX_HEAD {
                        return err(431, "trailers too large");
                    }
                }
            }
            if body.len() + n > MAX_BODY {
                return err(413, "request body too large");
            }
            while self.buf.len() < n + 2 {
                if self.fill(Some(since))? == 0 {
                    return err(400, "connection closed mid-body");
                }
            }
            body.extend_from_slice(&self.buf[..n]);
            let crlf = match &self.buf[n..n + 2] {
                b"\r\n" => 2,
                [b'\n', _] => 1,
                _ => return err(400, "malformed chunk"),
            };
            self.buf.drain(..n + crlf);
        }
    }

    fn write_all(&mut self, b: &[u8]) -> io::Result<()> {
        self.s.write_all(b)?;
        self.written += b.len() as u64;
        self.s.flush()
    }

    /// A whole response, in one write.
    pub fn respond(
        &mut self,
        status: u16,
        ctype: &str,
        body: &[u8],
        keep_alive: bool,
        extra: &[(&str, &str)],
    ) -> io::Result<()> {
        self.status = status;
        let mut out = format!(
            "HTTP/1.1 {status} {}\r\nDate: {}\r\n{}",
            reason(status),
            http_date(unix_secs()),
            self.extra
        );
        if status != 204 {
            out.push_str(&format!(
                "Content-Type: {ctype}\r\nContent-Length: {}\r\n",
                body.len()
            ));
        }
        for (n, v) in extra {
            out.push_str(&format!("{n}: {v}\r\n"));
        }
        if !keep_alive {
            out.push_str("Connection: close\r\n");
        }
        out.push_str("\r\n");
        let mut b = out.into_bytes();
        if !self.head_only && status != 204 {
            b.extend_from_slice(body);
        }
        self.write_all(&b)
    }

    /// Start a streamed response: chunked for HTTP/1.1, else delimited by
    /// closing the connection (then `keep_alive` must be false).
    pub fn start_stream(
        &mut self,
        status: u16,
        ctype: &str,
        minor: u8,
        keep_alive: bool,
    ) -> io::Result<()> {
        self.status = status;
        self.chunked = minor == 1;
        let mut out = format!(
            "HTTP/1.1 {status} {}\r\nDate: {}\r\n{}Content-Type: {ctype}\r\nCache-Control: no-cache\r\n",
            reason(status),
            http_date(unix_secs()),
            self.extra
        );
        if self.chunked {
            out.push_str("Transfer-Encoding: chunked\r\n");
        }
        if !keep_alive || !self.chunked {
            out.push_str("Connection: close\r\n");
        }
        out.push_str("\r\n");
        self.write_all(out.as_bytes())
    }

    /// Part of a streamed response's body, sent now (one write).
    pub fn send(&mut self, data: &[u8]) -> io::Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        if !self.chunked {
            return self.write_all(data);
        }
        let mut b = format!("{:x}\r\n", data.len()).into_bytes();
        b.reserve(data.len() + 2);
        b.extend_from_slice(data);
        b.extend_from_slice(b"\r\n");
        self.write_all(&b)
    }

    /// The end of a streamed response.
    pub fn end_stream(&mut self) -> io::Result<()> {
        if self.chunked {
            self.write_all(b"0\r\n\r\n")
        } else {
            Ok(())
        }
    }
}

fn unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// An IMF-fixdate (`Sun, 06 Nov 1994 08:49:37 GMT`) for `Date`.
pub fn http_date(t: u64) -> String {
    let (days, s) = (t / 86400, t % 86400);
    // Civil from days (Howard Hinnant's algorithm), 1970-01-01 a Thursday.
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    const WD: [&str; 7] = ["Thu", "Fri", "Sat", "Sun", "Mon", "Tue", "Wed"];
    const MON: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    format!(
        "{}, {d:02} {} {y} {:02}:{:02}:{:02} GMT",
        WD[(days % 7) as usize],
        MON[(m - 1) as usize],
        s / 3600,
        s / 60 % 60,
        s % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The response text with its `Date` line removed.
    fn undated(b: &[u8]) -> String {
        String::from_utf8(b.to_vec())
            .unwrap()
            .split("\r\n")
            .filter(|l| !l.starts_with("Date: "))
            .collect::<Vec<_>>()
            .join("\r\n")
    }

    #[test]
    fn dates() {
        assert_eq!(http_date(784_111_777), "Sun, 06 Nov 1994 08:49:37 GMT");
        assert_eq!(http_date(0), "Thu, 01 Jan 1970 00:00:00 GMT");
        assert_eq!(http_date(951_782_400), "Tue, 29 Feb 2000 00:00:00 GMT");
        assert_eq!(http_date(1_792_044_266), "Thu, 15 Oct 2026 06:04:26 GMT");
    }

    #[test]
    fn absolute_form_targets() {
        for (t, want) in [
            ("http://h:8080/v1/models?x=1", "/v1/models"),
            ("HTTPS://h/a/b", "/a/b"),
            ("http://h", "/"),
            ("http://h?q", "/"),
        ] {
            let raw = format!("GET {t} HTTP/1.1\r\n\r\n");
            assert_eq!(one(raw.as_bytes()).unwrap().unwrap().path, want, "{t}");
        }
        assert_eq!(status(b"GET ftp://h/x HTTP/1.1\r\n\r\n"), 400);
        assert_eq!(status(b"OPTIONS * HTTP/1.1\r\n\r\n"), 400);
    }

    /// A stream that hands out `input` `step` bytes per read and records
    /// what is written.
    pub struct Mock {
        input: Vec<u8>,
        at: usize,
        step: usize,
        pub out: Vec<u8>,
    }

    impl Mock {
        pub fn new(input: &[u8], step: usize) -> Mock {
            Mock {
                input: input.to_vec(),
                at: 0,
                step,
                out: Vec::new(),
            }
        }
    }

    impl Read for Mock {
        fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
            let n = self.step.min(b.len()).min(self.input.len() - self.at);
            b[..n].copy_from_slice(&self.input[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    impl Write for Mock {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.out.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    fn all(input: &[u8], step: usize) -> Vec<Result<Option<Request>, HttpError>> {
        let mut c = Conn::new(Mock::new(input, step));
        let mut v = Vec::new();
        loop {
            let r = c.read_request();
            let stop = !matches!(r, Ok(Some(_)));
            v.push(r);
            if stop {
                return v;
            }
        }
    }

    fn one(input: &[u8]) -> Result<Option<Request>, HttpError> {
        Conn::new(Mock::new(input, 7)).read_request()
    }

    fn status(input: &[u8]) -> u16 {
        match one(input) {
            Err(e) => e.status,
            Ok(Some(_)) => 200,
            Ok(None) => 0,
        }
    }

    /// Bytes that can't begin an HTTP request (a TLS ClientHello, junk)
    /// are a 400 as soon as they arrive, not after `head_timeout` waiting
    /// for a line end. A valid head arriving slowly is still waited for.
    #[test]
    fn non_http_bytes_rejected_at_once() {
        let hello = [
            0x16u8, 0x03, 0x01, 0x00, 0xa5, 0x01, 0x00, 0x00, 0xa1, 0x03, 0x03,
        ];
        for input in [&hello[..], b"\x00\x01\x02", b"GE\x7fT", b"\tGET / HTTP/1.1"] {
            for step in [1, 4096] {
                let e = Conn::new(Mock::new(input, step))
                    .read_request()
                    .unwrap_err();
                assert_eq!(
                    (e.status, e.msg),
                    (400, "malformed request line"),
                    "{input:?}"
                );
            }
        }
        for input in [&b"POST /v1/completions HTTP/1.1\r\nHo"[..], b"GET", b"G"] {
            let e = Conn::new(Mock::new(input, 1)).read_request().unwrap_err();
            assert_eq!(e.msg, "connection closed mid-request", "{input:?}");
        }
    }

    #[test]
    fn parses_a_post_with_a_body_in_any_read_size() {
        let raw = b"POST /v1/completions?x=1 HTTP/1.1\r\nHost: a\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n{\"a\":\"bc\"}\n";
        for step in [1, 2, 5, 64, 8192] {
            let r = one(raw).unwrap().unwrap();
            let _ = step;
            let mut c = Conn::new(Mock::new(raw, step));
            let r2 = c.read_request().unwrap().unwrap();
            for r in [r, r2] {
                assert_eq!(r.method, "POST");
                assert_eq!(r.path, "/v1/completions");
                assert_eq!(r.minor, 1);
                assert!(r.keep_alive);
                assert_eq!(r.header("content-type"), Some("application/json"));
                assert_eq!(r.body, b"{\"a\":\"bc\"}\n");
            }
            assert!(matches!(c.read_request(), Ok(None)));
        }
    }

    #[test]
    fn bare_lf_and_leading_blank_lines() {
        let r = one(b"\r\n\nGET /v1/models HTTP/1.0\nConnection: keep-alive\n\n")
            .unwrap()
            .unwrap();
        assert_eq!(
            (r.path.as_str(), r.minor, r.keep_alive),
            ("/v1/models", 0, true)
        );
        let r = one(b"GET / HTTP/1.0\r\n\r\n").unwrap().unwrap();
        assert!(!r.keep_alive);
        let r = one(b"GET / HTTP/1.1\r\nConnection: Close\r\n\r\n")
            .unwrap()
            .unwrap();
        assert!(!r.keep_alive);
    }

    #[test]
    fn pipelined_requests_come_out_in_order() {
        let raw = b"POST /a HTTP/1.1\r\nContent-Length: 3\r\n\r\nabcGET /b HTTP/1.1\r\n\r\nPOST /c HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nxyz\r\n0\r\n\r\nGET /d HTTP/1.1\r\nConnection: close\r\n\r\n";
        for step in [1, 3, 4096] {
            let v = all(raw, step);
            assert_eq!(v.len(), 5, "step {step}");
            let got: Vec<(String, Vec<u8>, bool)> = v[..4]
                .iter()
                .map(|r| {
                    let r = r.as_ref().unwrap().as_ref().unwrap();
                    (r.path.clone(), r.body.clone(), r.keep_alive)
                })
                .collect();
            assert_eq!(
                got,
                vec![
                    ("/a".into(), b"abc".to_vec(), true),
                    ("/b".into(), vec![], true),
                    ("/c".into(), b"xyz".to_vec(), true),
                    ("/d".into(), vec![], false),
                ]
            );
            assert!(matches!(v[4], Ok(None)));
        }
    }

    #[test]
    fn chunked_bodies() {
        let raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: Chunked\r\n\r\n4;ext=1\r\nWiki\r\n5\r\npedia\r\nE\r\n in\r\n\r\nchunks.\r\n0\r\nX-Trailer: 1\r\n\r\n";
        let r = one(raw).unwrap().unwrap();
        assert_eq!(r.body, b"Wikipedia in\r\n\r\nchunks.");
        assert_eq!(
            status(b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n"),
            400
        );
        assert_eq!(
            status(b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabcd\r\n0\r\n\r\n"),
            400
        );
        assert_eq!(
            status(b"POST /x HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n"),
            501
        );
        assert_eq!(
            status(b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\n"),
            400
        );
        assert_eq!(
            status(b"POST /x HTTP/1.0\r\nTransfer-Encoding: chunked\r\n\r\n"),
            400
        );
        // Past MAX_BODY in total: refused before it is all read.
        let mut big = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        for _ in 0..3 {
            big.extend_from_slice(format!("{:x}\r\n", 3 << 20).as_bytes());
            big.extend(std::iter::repeat_n(b'a', 3 << 20));
            big.extend_from_slice(b"\r\n");
        }
        assert_eq!(status(&big), 413);
    }

    #[test]
    fn malformed_requests() {
        for (raw, want) in [
            (&b"GET /\r\n\r\n"[..], 400),
            (b"GET  / HTTP/1.1\r\n\r\n", 400),
            (b"GET / HTTP/2.0\r\n\r\n", 505),
            (b"GET / FOO/1.1\r\n\r\n", 400),
            (b"GET x/ HTTP/1.1\r\n\r\n", 400),
            (b"G(T / HTTP/1.1\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\nNo-Colon\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\nBad Name: 1\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\nA: 1\r\n folded\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\nA: \x01\r\n\r\n", 400),
            (b"GET / HTTP/1.1\r\nA: \xff\r\n\r\n", 400),
            (b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n", 400),
            (b"POST / HTTP/1.1\r\nContent-Length: 1, 1\r\n\r\n", 400),
            (
                b"POST / HTTP/1.1\r\nContent-Length: 2\r\nContent-Length: 3\r\n\r\nabc",
                400,
            ),
            (
                b"POST / HTTP/1.1\r\nContent-Length: 3\r\nContent-Length: 3\r\n\r\nabc",
                200,
            ),
            (
                b"POST / HTTP/1.1\r\nContent-Length: 99999999999999999999999\r\n\r\n",
                413,
            ),
            (b"POST / HTTP/1.1\r\nContent-Length: 9000000\r\n\r\n", 413),
            (b"POST / HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", 400),
            (b"POST / HTTP/1.1\r\nExpect: magic\r\n\r\n", 417),
            (b"GET / HTTP/1.1\r\nHost: a", 400),
            (b"", 0),
            (b"\r\n\r\n", 0),
        ] {
            assert_eq!(status(raw), want, "{:?}", String::from_utf8_lossy(raw));
        }
    }

    #[test]
    fn oversized_heads() {
        let mut raw = b"GET / HTTP/1.1\r\n".to_vec();
        while raw.len() <= MAX_HEAD {
            raw.extend_from_slice(
                b"X-Pad: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\r\n",
            );
        }
        raw.extend_from_slice(b"\r\n");
        assert_eq!(status(&raw), 431);
        // Never terminated: refused once past the limit, not buffered forever.
        let mut raw = b"GET / HTTP/1.1\r\nX: ".to_vec();
        raw.extend(std::iter::repeat_n(b'a', 1 << 20));
        assert_eq!(status(&raw), 431);
        let mut raw = b"GET / HTTP/1.1\r\n".to_vec();
        for i in 0..=MAX_HEADERS {
            raw.extend_from_slice(format!("H{i}: 1\r\n").as_bytes());
        }
        raw.extend_from_slice(b"\r\n");
        assert_eq!(status(&raw), 431);
    }

    #[test]
    fn expect_continue_is_answered_before_the_body() {
        let raw = b"POST /x HTTP/1.1\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\nhi";
        // Head and body in separate reads: the 100 goes out in between.
        let mut c = Conn::new(Mock::new(raw, raw.len() - 2));
        let r = c.read_request().unwrap().unwrap();
        assert_eq!(r.body, b"hi");
        assert_eq!(c.s.out, b"HTTP/1.1 100 Continue\r\n\r\n");
        // Body already there (or none): no 100.
        let mut c = Conn::new(Mock::new(raw, 4096));
        c.read_request().unwrap().unwrap();
        assert!(c.s.out.is_empty());
    }

    #[test]
    fn head_timeout() {
        let mut c = Conn::new(Mock::new(b"GET / HTTP/1.1\r\n", 1));
        c.head_timeout = Duration::ZERO;
        std::thread::sleep(Duration::from_millis(2));
        assert_eq!(c.read_request().unwrap_err().status, 408);
    }

    #[test]
    fn responses() {
        let mut c = Conn::new(Mock::new(b"", 1));
        c.respond(404, "application/json", b"{}", false, &[("Allow", "GET")])
            .unwrap();
        assert!(String::from_utf8_lossy(&c.s.out).contains("\r\nDate: "));
        assert_eq!(
            undated(&std::mem::take(&mut c.s.out)),
            "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nContent-Length: 2\r\nAllow: GET\r\nConnection: close\r\n\r\n{}"
        );
        c.start_stream(200, "text/event-stream", 1, true).unwrap();
        c.send(b"data: x\n\n").unwrap();
        c.send(b"").unwrap();
        c.end_stream().unwrap();
        assert_eq!(
            undated(&std::mem::take(&mut c.s.out)),
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n9\r\ndata: x\n\n\r\n0\r\n\r\n"
        );
        // HEAD: the same headers, no body; 204: no body or content headers;
        // `extra` on everything.
        c.extra = "Access-Control-Allow-Origin: *\r\n".into();
        c.head_only = true;
        c.respond(200, "application/json", b"{}", true, &[])
            .unwrap();
        assert_eq!(
            undated(&std::mem::take(&mut c.s.out)),
            "HTTP/1.1 200 OK\r\nAccess-Control-Allow-Origin: *\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n"
        );
        c.head_only = false;
        c.respond(204, "", b"", true, &[("Allow", "GET")]).unwrap();
        assert_eq!(
            undated(&std::mem::take(&mut c.s.out)),
            "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\nAllow: GET\r\n\r\n"
        );
        c.extra.clear();
        c.start_stream(200, "text/event-stream", 0, false).unwrap();
        c.send(b"data: x\n\n").unwrap();
        c.end_stream().unwrap();
        assert_eq!(
            undated(&std::mem::take(&mut c.s.out)),
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\ndata: x\n\n"
        );
    }
}
