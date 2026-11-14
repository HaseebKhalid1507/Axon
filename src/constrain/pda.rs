//! The automaton: a byte-level graph with calls and returns (a pushdown
//! automaton), deterministic by construction.
//!
//! Nodes are grammar positions. An edge consumes one byte of a set, or one
//! special token (a symbol of its own, never bytes: `Sym::Special`), or is a
//! call: push a return node and enter a sub-grammar. A node marked `ret`
//! returns (pops) when none of its edges takes the symbol, so a sub-grammar
//! ends by lookahead (a number ends at the first byte that isn't part of
//! it) and a token's last byte never pops: the state after a token is where
//! its last byte left it. `Builder::finish` checks that every node's
//! alternatives start with disjoint symbols (so a step has at most one way
//! to go), that every call enters a sub-grammar that consumes before it can
//! return (no nullable calls, no left recursion), and that every node can
//! still finish (so a mask is never empty).
//!
//! `step` is generic over the stack so that the live state (`Vec<NodeId>`)
//! and the mask walk's virtual stack (mask.rs: the live stack's top known,
//! everything below it unknown) run the very same transitions: that is
//! what makes a cached mask equal the token-by-token check.

/// A grammar position.
pub type NodeId = u32;

/// Most stack frames when a JSON container (object or array) opens: its
/// own frame included, so at most this many containers are open at once
/// when the grammar holds one frame per container (the free JSON grammar
/// does). `tools::Reader` refuses nesting past 128, so every call the
/// automaton lets through parses.
pub const MAX_DEPTH: usize = 64;

/// A set of bytes.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct ByteSet([u64; 4]);

impl ByteSet {
    pub fn of(bytes: &[u8]) -> ByteSet {
        let mut s = ByteSet::default();
        for &b in bytes {
            s.add(b);
        }
        s
    }

    /// lo..=hi.
    pub fn range(lo: u8, hi: u8) -> ByteSet {
        let mut s = ByteSet::default();
        for b in lo..=hi {
            s.add(b);
        }
        s
    }

    pub fn add(&mut self, b: u8) {
        self.0[(b >> 6) as usize] |= 1 << (b & 63);
    }

    pub fn has(&self, b: u8) -> bool {
        (self.0[(b >> 6) as usize] >> (b & 63)) & 1 == 1
    }

    pub fn union(self, o: ByteSet) -> ByteSet {
        ByteSet(std::array::from_fn(|i| self.0[i] | o.0[i]))
    }

    pub fn minus(self, o: ByteSet) -> ByteSet {
        ByteSet(std::array::from_fn(|i| self.0[i] & !o.0[i]))
    }

    pub fn intersects(&self, o: &ByteSet) -> bool {
        (0..4).any(|i| self.0[i] & o.0[i] != 0)
    }
}

/// One input symbol: a byte of a token's text, or a special token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sym {
    Byte(u8),
    Special(u32),
}

#[derive(Clone, Debug)]
pub enum Edge {
    /// Take one byte of `set`, go to `to`. `opens`: it opens a JSON
    /// container, allowed only while the stack (its frame included) holds
    /// at most `MAX_DEPTH` frames.
    Bytes {
        set: ByteSet,
        to: NodeId,
        opens: bool,
    },
    /// Take special token `id`, go to `to`.
    Special { id: u32, to: NodeId },
    /// Push `ret`, enter `sub` (taking nothing).
    Call { sub: NodeId, ret: NodeId },
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub edges: Vec<Edge>,
    /// None of `edges` takes the symbol: return to the caller (pop).
    pub ret: bool,
    /// The grammar is complete here (with an empty stack).
    pub accept: bool,
    /// Inside a JSON string (its body or an escape): what a reader of the
    /// text needs to tell a `</tool_call>` in a string value from a real
    /// one (the maintainer's ruling 5 on the spec's §6: PR B's Parser).
    pub in_str: bool,
}

/// What a step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Ok,
    Reject,
    /// It depends on stack frames the stack doesn't know (mask.rs: below
    /// the live stack's top).
    Unknown,
}

/// What a pop found.
pub enum Pop {
    Node(NodeId),
    Empty,
    Unknown,
}

/// A stack `Graph::step` runs on.
pub trait Stack {
    fn push(&mut self, ret: NodeId);
    fn pop(&mut self) -> Pop;
    /// Whether a container may open now, its frame already pushed (the
    /// stack holds at most `MAX_DEPTH` frames); None: not known.
    fn may_open(&self) -> Option<bool>;
}

impl Stack for Vec<NodeId> {
    fn push(&mut self, ret: NodeId) {
        Vec::push(self, ret);
    }

    fn pop(&mut self) -> Pop {
        match Vec::pop(self) {
            Some(n) => Pop::Node(n),
            None => Pop::Empty,
        }
    }

    fn may_open(&self) -> Option<bool> {
        Some(self.len() <= MAX_DEPTH)
    }
}

/// A finished graph (`Builder::finish`).
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    /// Per node: the bytes and the specials (sorted) it can take first,
    /// through calls (never through a return: sub-grammars aren't
    /// nullable).
    first: Vec<ByteSet>,
    first_sp: Vec<Vec<u32>>,
}

impl Graph {
    /// Whether `n`'s sub-grammar can start with `sym`.
    fn starts(&self, n: NodeId, sym: Sym) -> bool {
        match sym {
            Sym::Byte(b) => self.first[n as usize].has(b),
            Sym::Special(id) => self.first_sp[n as usize].binary_search(&id).is_ok(),
        }
    }

    /// Take `sym` at `*node` with stack `st`: Ok (and `*node` is where it
    /// went), Reject, or Unknown (`st` couldn't tell). After Reject or
    /// Unknown, `*node` and `st` are unspecified.
    pub fn step<S: Stack>(&self, node: &mut NodeId, st: &mut S, sym: Sym) -> Step {
        let mut n = *node;
        loop {
            let nd = &self.nodes[n as usize];
            let mut call = None;
            for e in &nd.edges {
                match *e {
                    Edge::Bytes { set, to, opens } => {
                        if let Sym::Byte(b) = sym
                            && set.has(b)
                        {
                            if opens {
                                match st.may_open() {
                                    Some(true) => {}
                                    Some(false) => return Step::Reject,
                                    None => return Step::Unknown,
                                }
                            }
                            *node = to;
                            return Step::Ok;
                        }
                    }
                    Edge::Special { id, to } => {
                        if sym == Sym::Special(id) {
                            *node = to;
                            return Step::Ok;
                        }
                    }
                    Edge::Call { sub, ret } => {
                        if self.starts(sub, sym) {
                            call = Some((sub, ret));
                            break;
                        }
                    }
                }
            }
            if let Some((sub, ret)) = call {
                st.push(ret);
                n = sub;
                continue;
            }
            if !nd.ret {
                return Step::Reject;
            }
            match st.pop() {
                Pop::Node(r) => n = r,
                Pop::Empty => return Step::Reject,
                Pop::Unknown => return Step::Unknown,
            }
        }
    }

    /// Whether the input may end at (`node`, `stack`): finished frames
    /// return (as at a symbol nothing takes) until an accepting node with
    /// an empty stack.
    #[cfg_attr(not(test), allow(dead_code))] // the tests' oracle
    pub fn may_end(&self, mut node: NodeId, stack: &[NodeId]) -> bool {
        let mut st = stack.to_vec();
        loop {
            let nd = &self.nodes[node as usize];
            if nd.accept && st.is_empty() {
                return true;
            }
            if !nd.ret {
                return false;
            }
            match st.pop() {
                Some(r) => node = r,
                None => return false,
            }
        }
    }
}

/// Builds a `Graph`.
#[derive(Default)]
pub struct Builder {
    nodes: Vec<Node>,
}

/// JSON whitespace.
pub fn ws() -> ByteSet {
    ByteSet::of(b" \t\n\r")
}

impl Builder {
    pub fn node(&mut self) -> NodeId {
        self.nodes.push(Node::default());
        (self.nodes.len() - 1) as NodeId
    }

    pub fn bytes(&mut self, from: NodeId, set: ByteSet, to: NodeId) {
        self.edge(
            from,
            Edge::Bytes {
                set,
                to,
                opens: false,
            },
        );
    }

    /// `bytes` for a byte that opens a container (`Edge::Bytes::opens`).
    pub fn opens(&mut self, from: NodeId, set: ByteSet, to: NodeId) {
        self.edge(
            from,
            Edge::Bytes {
                set,
                to,
                opens: true,
            },
        );
    }

    pub fn special(&mut self, from: NodeId, id: u32, to: NodeId) {
        self.edge(from, Edge::Special { id, to });
    }

    pub fn call(&mut self, from: NodeId, sub: NodeId, ret: NodeId) {
        self.edge(from, Edge::Call { sub, ret });
    }

    fn edge(&mut self, from: NodeId, e: Edge) {
        self.nodes[from as usize].edges.push(e);
    }

    /// Copy `src`'s edges into `dst` (a tail call: `dst` takes what `src`
    /// starts with and goes on as `src` would).
    pub fn merge(&mut self, dst: NodeId, src: NodeId) {
        let e = self.nodes[src as usize].edges.clone();
        self.nodes[dst as usize].edges.extend(e);
    }

    pub fn set_ret(&mut self, n: NodeId) {
        self.nodes[n as usize].ret = true;
    }

    pub fn set_accept(&mut self, n: NodeId) {
        self.nodes[n as usize].accept = true;
    }

    pub fn set_in_str(&mut self, n: NodeId) {
        self.nodes[n as usize].in_str = true;
    }

    /// Whitespace loops at `n`.
    pub fn ws_loop(&mut self, n: NodeId) {
        self.bytes(n, ws(), n);
    }

    /// A chain taking `lit` byte by byte from `from`; its end node.
    pub fn lit(&mut self, from: NodeId, lit: &[u8]) -> NodeId {
        let mut at = from;
        for &b in lit {
            let next = self.node();
            self.bytes(at, ByteSet::of(&[b]), next);
            at = next;
        }
        at
    }

    /// The graph, checked (module doc): Err names the first problem.
    pub fn finish(self) -> Result<Graph, String> {
        let nodes = self.nodes;
        let n = nodes.len();
        for (i, nd) in nodes.iter().enumerate() {
            for e in &nd.edges {
                let (a, b) = match *e {
                    Edge::Bytes { to, .. } | Edge::Special { to, .. } => (to, to),
                    Edge::Call { sub, ret } => (sub, ret),
                };
                if a as usize >= n || b as usize >= n {
                    return Err(format!("node {i}: an edge to a node that doesn't exist"));
                }
            }
        }
        // First sets, to a fixpoint (calls may be recursive).
        let mut first = vec![ByteSet::default(); n];
        let mut first_sp: Vec<Vec<u32>> = vec![Vec::new(); n];
        loop {
            let mut changed = false;
            for i in 0..n {
                let mut f = first[i];
                let mut sp = first_sp[i].clone();
                for e in &nodes[i].edges {
                    match *e {
                        Edge::Bytes { set, .. } => f = f.union(set),
                        Edge::Special { id, .. } => sp.push(id),
                        Edge::Call { sub, .. } => {
                            f = f.union(first[sub as usize]);
                            sp.extend_from_slice(&first_sp[sub as usize]);
                        }
                    }
                }
                sp.sort_unstable();
                sp.dedup();
                if f != first[i] || sp != first_sp[i] {
                    first[i] = f;
                    first_sp[i] = sp;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        for (i, nd) in nodes.iter().enumerate() {
            // Sub-grammars consume before they can end.
            for e in &nd.edges {
                if let Edge::Call { sub, .. } = *e {
                    let s = &nodes[sub as usize];
                    if s.ret || s.accept {
                        return Err(format!("node {i}: a call into a nullable node {sub}"));
                    }
                }
            }
            // Disjoint alternatives.
            let mut seen = ByteSet::default();
            let mut seen_sp: Vec<u32> = Vec::new();
            for e in &nd.edges {
                let (f, sp): (ByteSet, &[u32]) = match e {
                    Edge::Bytes { set, .. } => (*set, &[]),
                    Edge::Special { id, .. } => (ByteSet::default(), std::slice::from_ref(id)),
                    Edge::Call { sub, .. } => (first[*sub as usize], &first_sp[*sub as usize]),
                };
                if f.intersects(&seen) || sp.iter().any(|x| seen_sp.contains(x)) {
                    return Err(format!("node {i}: two alternatives start alike"));
                }
                seen = seen.union(f);
                seen_sp.extend_from_slice(sp);
            }
        }
        // No left recursion: following calls without taking anything never
        // comes back to where it started (else `step` would push forever).
        for start in 0..n {
            let mut stack = vec![start];
            let mut seen = vec![false; n];
            while let Some(x) = stack.pop() {
                for e in &nodes[x].edges {
                    if let Edge::Call { sub, .. } = *e {
                        let s = sub as usize;
                        if s == start {
                            return Err(format!("node {start}: left recursion"));
                        }
                        if !seen[s] {
                            seen[s] = true;
                            stack.push(s);
                        }
                    }
                }
            }
        }
        // Every node can finish: reach a return or an accepting node.
        let mut fin = vec![false; n];
        loop {
            let mut changed = false;
            for i in 0..n {
                if fin[i] {
                    continue;
                }
                let nd = &nodes[i];
                let f = nd.ret
                    || nd.accept
                    || nd.edges.iter().any(|e| match *e {
                        Edge::Bytes { to, .. } | Edge::Special { to, .. } => fin[to as usize],
                        Edge::Call { sub, ret } => fin[sub as usize] && fin[ret as usize],
                    });
                if f {
                    fin[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        if let Some(i) = fin.iter().position(|&f| !f) {
            return Err(format!("node {i} can't finish"));
        }
        Ok(Graph {
            nodes,
            first,
            first_sp,
        })
    }
}

/// The JSON library's entry nodes (RFC 8259, with bytes >= 0x80 taken as
/// string bytes without checking UTF-8, as `tools::parse_json`'s lossy
/// callers read them; at most `MAX_DEPTH` stack frames when a container
/// opens). Each is a sub-grammar to call: it consumes before it returns.
#[derive(Clone, Copy, Debug)]
pub struct Json {
    /// Any value (PR C: json_object's top; the tests' oracle).
    #[cfg_attr(not(test), allow(dead_code))]
    pub value: NodeId,
    pub string: NodeId,
    pub object: NodeId,
}

/// Add the JSON library to `b`.
pub fn json(b: &mut Builder) -> Json {
    let quote = ByteSet::of(b"\"");
    let digit = ByteSet::range(b'0', b'9');
    let hex = digit
        .union(ByteSet::range(b'a', b'f'))
        .union(ByteSet::range(b'A', b'F'));
    let e = ByteSet::of(b"eE");

    // Strings: no raw control bytes, escapes as RFC 8259 has them.
    let string = b.node();
    let body = b.node();
    let end = b.node();
    let esc = b.node();
    b.bytes(string, quote, body);
    let plain = ByteSet::range(0x20, 0xff).minus(ByteSet::of(b"\"\\"));
    b.bytes(body, plain, body);
    b.bytes(body, quote, end);
    b.bytes(body, ByteSet::of(b"\\"), esc);
    b.bytes(esc, ByteSet::of(b"\"\\/bfnrt"), body);
    let mut u = b.node();
    b.bytes(esc, ByteSet::of(b"u"), u);
    b.set_in_str(body);
    b.set_in_str(esc);
    for i in 0..4 {
        b.set_in_str(u);
        let next = if i == 3 { body } else { b.node() };
        b.bytes(u, hex, next);
        u = next;
    }
    b.set_ret(end);

    // Numbers: -?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?, ending at
    // the first byte that isn't part of one.
    let number = b.node();
    let minus = b.node();
    let zero = b.node();
    let int = b.node();
    let frac0 = b.node();
    let frac = b.node();
    let exp0 = b.node();
    let exp1 = b.node();
    let exp = b.node();
    let nz = ByteSet::range(b'1', b'9');
    b.bytes(number, ByteSet::of(b"-"), minus);
    for n in [number, minus] {
        b.bytes(n, ByteSet::of(b"0"), zero);
        b.bytes(n, nz, int);
    }
    b.bytes(int, digit, int);
    for n in [zero, int] {
        b.bytes(n, ByteSet::of(b"."), frac0);
        b.bytes(n, e, exp0);
        b.set_ret(n);
    }
    b.bytes(frac0, digit, frac);
    b.bytes(frac, digit, frac);
    b.bytes(frac, e, exp0);
    b.set_ret(frac);
    b.bytes(exp0, ByteSet::of(b"+-"), exp1);
    b.bytes(exp0, digit, exp);
    b.bytes(exp1, digit, exp);
    b.bytes(exp, digit, exp);
    b.set_ret(exp);

    // true, false, null.
    let lits: Vec<NodeId> = [&b"true"[..], b"false", b"null"]
        .iter()
        .map(|w| {
            let s = b.node();
            let end = b.lit(s, w);
            b.set_ret(end);
            s
        })
        .collect();

    // Objects and arrays; `value` is filled in below (recursion).
    let value = b.node();
    let object = b.node();
    let o_open = b.node();
    let o_colon = b.node();
    let o_val = b.node();
    let o_after = b.node();
    let o_key = b.node();
    let o_end = b.node();
    b.opens(object, ByteSet::of(b"{"), o_open);
    b.ws_loop(o_open);
    b.bytes(o_open, ByteSet::of(b"}"), o_end);
    b.call(o_open, string, o_colon);
    b.ws_loop(o_colon);
    b.bytes(o_colon, ByteSet::of(b":"), o_val);
    b.ws_loop(o_val);
    b.call(o_val, value, o_after);
    b.ws_loop(o_after);
    b.bytes(o_after, ByteSet::of(b","), o_key);
    b.bytes(o_after, ByteSet::of(b"}"), o_end);
    b.ws_loop(o_key);
    b.call(o_key, string, o_colon);
    b.set_ret(o_end);

    let array = b.node();
    let a_open = b.node();
    let a_after = b.node();
    let a_val = b.node();
    let a_end = b.node();
    b.opens(array, ByteSet::of(b"["), a_open);
    b.ws_loop(a_open);
    b.bytes(a_open, ByteSet::of(b"]"), a_end);
    b.call(a_open, value, a_after);
    b.ws_loop(a_after);
    b.bytes(a_after, ByteSet::of(b","), a_val);
    b.bytes(a_after, ByteSet::of(b"]"), a_end);
    b.ws_loop(a_val);
    b.call(a_val, value, a_after);
    b.set_ret(a_end);

    // A value: each kind's first byte straight into its body (tail calls:
    // the kind returns to whoever called `value`).
    for k in [string, number, object, array] {
        b.merge(value, k);
    }
    for l in lits {
        b.merge(value, l);
    }
    Json {
        value,
        string,
        object,
    }
}
