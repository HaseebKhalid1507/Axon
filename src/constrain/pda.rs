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
//!
//! The JSON library comes in two styles (`Style`). `Style::default()` is
//! RFC 8259's JSON (the oracle tests). `Style::HELD`, what a held tool call
//! gets, keeps a model from running on inside a call (#272): whitespace
//! between JSON tokens bounded as llama.cpp's grammars bound it
//! (`Space::Bounded`: a whitespace position is unrolled into a few copies of
//! its node, one per count of what was taken, `Builder::expand_spaces`, so
//! the automaton stays a plain graph and a mask is still a function of the
//! node), and an array's items all of its first item's kind (no
//! `[{...}, false, false, ...]`).
//!
//! `Graph::finisher` is a graph's shortest way out: from any state, only
//! the symbols on a shortest path to the grammar's end. A constraint
//! switches to it to close a call that has stalled (mod.rs).

/// A grammar position.
pub type NodeId = u32;

/// Most stack frames when a JSON container (object or array) opens: its
/// own frame included, so at most this many containers are open at once
/// when the grammar holds one frame per container (the free JSON grammar
/// does). `tools::Reader` refuses nesting past 128, so every call the
/// automaton lets through parses.
pub const MAX_DEPTH: usize = 64;

/// Whitespace between JSON tokens (`Builder::ws_loop`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Space {
    /// Any run of JSON whitespace (space, tab, LF, CR), as RFC 8259 has it.
    #[default]
    Free,
    /// llama.cpp's `space` rule (common/json-schema-to-grammar.cpp,
    /// `SPACE_RULE` at 4413a19: `| " " | "\n"{1,2} [ \t]{0,20}`): nothing,
    /// one space, or one or two newlines then at most `INDENT_MAX` spaces or
    /// tabs. A valid call written on one line or pretty-printed fits; a
    /// model can't run on in whitespace inside a call (#272: 25k tokens of
    /// it until codex gave up).
    Bounded,
}

/// `Space::Bounded`: at most this many spaces or tabs after the newlines.
pub const INDENT_MAX: usize = 20;

/// How a grammar's JSON is held (`Builder::new`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Style {
    pub space: Space,
    /// An array's items are all of its first item's kind: string, number,
    /// boolean, null, object or array. In the acceptance runs' calls (5.7k
    /// across codex, opencode and pi, constrained or not, llama-server's
    /// too) no array mixed kinds but the 7 runaways of #272, which wrote
    /// `false` after `{...}` 1900 times.
    pub one_kind_arrays: bool,
}

impl Style {
    /// A held tool call's (module doc).
    pub const HELD: Style = Style {
        space: Space::Bounded,
        one_kind_arrays: true,
    };
}

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
    /// The graph as bytes, canonically: per node in order its flags, then
    /// its edges in order (kind, byte set, specials, targets), little
    /// endian. What the identity hashes (`constrain::hermes_field`), so a
    /// change to what the automaton takes changes the hash with no one
    /// having to remember to.
    pub fn canon(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.nodes.len() as u32).to_le_bytes());
        for nd in &self.nodes {
            out.push(nd.ret as u8 | (nd.accept as u8) << 1 | (nd.in_str as u8) << 2);
            out.extend_from_slice(&(nd.edges.len() as u32).to_le_bytes());
            for e in &nd.edges {
                match *e {
                    Edge::Bytes { set, to, opens } => {
                        out.push(if opens { 1 } else { 0 });
                        for w in set.0 {
                            out.extend_from_slice(&w.to_le_bytes());
                        }
                        out.extend_from_slice(&to.to_le_bytes());
                    }
                    Edge::Special { id, to } => {
                        out.push(2);
                        out.extend_from_slice(&id.to_le_bytes());
                        out.extend_from_slice(&to.to_le_bytes());
                    }
                    Edge::Call { sub, ret } => {
                        out.push(3);
                        out.extend_from_slice(&sub.to_le_bytes());
                        out.extend_from_slice(&ret.to_le_bytes());
                    }
                }
            }
        }
    }

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

impl Graph {
    /// The shortest way out of this graph: the same nodes (so a state here
    /// is a state there), each keeping only the edges on a shortest path to
    /// where its sub-grammar ends (bytes and specials count one each; a
    /// call, its sub-grammar's shortest plus its return node's), and a node
    /// that may return or accept keeping none (it returns, or the input
    /// ends). From any state it takes only what closes the open strings,
    /// containers and frame soonest: no whitespace, no new member, a value
    /// position the shortest value (a digit). Never None: every node of a
    /// finished graph can finish.
    pub fn finisher(&self) -> Graph {
        const INF: u64 = u64::MAX / 4;
        let n = self.nodes.len();
        let cost = |d: &[u64], e: &Edge| match *e {
            Edge::Bytes { to, .. } | Edge::Special { to, .. } => 1 + d[to as usize],
            Edge::Call { sub, ret } => d[sub as usize] + d[ret as usize],
        };
        let mut d = vec![INF; n];
        loop {
            let mut changed = false;
            for i in 0..n {
                let nd = &self.nodes[i];
                let mut best = if nd.ret || nd.accept { 0 } else { INF };
                for e in &nd.edges {
                    best = best.min(cost(&d, e));
                }
                if best < d[i] {
                    d[i] = best;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let nodes = (self.nodes.iter().enumerate())
            .map(|(i, nd)| Node {
                edges: match nd.ret || nd.accept {
                    true => Vec::new(),
                    false => (nd.edges.iter())
                        .filter(|e| cost(&d, e) == d[i])
                        .cloned()
                        .collect(),
                },
                ..nd.clone()
            })
            .collect();
        Builder {
            nodes,
            ..Builder::default()
        }
        .finish()
        .expect("a subgraph of a finished graph, every node keeping a shortest way out")
    }
}

/// Builds a `Graph`.
#[derive(Default)]
pub struct Builder {
    nodes: Vec<Node>,
    style: Style,
    /// `Space::Bounded`: whitespace positions not yet unrolled
    /// (`expand_spaces`).
    spaces: Vec<NodeId>,
}

/// JSON whitespace.
pub fn ws() -> ByteSet {
    ByteSet::of(b" \t\n\r")
}

impl Builder {
    /// A builder whose JSON (`json`, `ws_loop`) is `style`'s.
    pub fn new(style: Style) -> Builder {
        Builder {
            style,
            ..Builder::default()
        }
    }

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

    /// Whitespace at `n`, as the builder's `Space` has it: a loop (`Free`),
    /// or (`Bounded`) unrolled by `expand_spaces` once `n`'s other edges are
    /// all in.
    pub fn ws_loop(&mut self, n: NodeId) {
        match self.style.space {
            Space::Free => self.bytes(n, ws(), n),
            Space::Bounded => self.spaces.push(n),
        }
    }

    /// Unroll the bounded whitespace positions so far (`Space::Bounded`):
    /// each node `n` gets 2 + `INDENT_MAX` + 1 copies (its edges and flags,
    /// so a copy takes what `n` takes), reached by whitespace: one space
    /// (then nothing more), a newline, a second newline, then the k-th space
    /// or tab. Call it when every such node's other edges are in (`finish`
    /// does, for any left).
    pub fn expand_spaces(&mut self) {
        let st = ByteSet::of(b" \t");
        for n in std::mem::take(&mut self.spaces) {
            let proto = self.nodes[n as usize].clone();
            let mut copy = || {
                self.nodes.push(proto.clone());
                (self.nodes.len() - 1) as NodeId
            };
            let one = copy();
            let nl1 = copy();
            let nl2 = copy();
            let ind: Vec<NodeId> = (0..INDENT_MAX).map(|_| copy()).collect();
            self.bytes(n, ByteSet::of(b" "), one);
            self.bytes(n, ByteSet::of(b"\n"), nl1);
            self.bytes(nl1, ByteSet::of(b"\n"), nl2);
            self.bytes(nl1, st, ind[0]);
            self.bytes(nl2, st, ind[0]);
            for k in 1..INDENT_MAX {
                self.bytes(ind[k - 1], st, ind[k]);
            }
        }
    }

    /// A JSON string that is one of `names` (each its string literal's
    /// body, as escaped in the text the model saw: `"` and `\` not taken
    /// raw), from `from` (taking its opening quote) to `to` (after its
    /// closing one). A trie, so names sharing a prefix share its nodes, and
    /// every node's alternatives start with different bytes. Its nodes are
    /// string nodes (`Node::in_str`). Duplicates count once; no names:
    /// nothing is taken (the caller's grammar must not need it).
    pub fn names(&mut self, from: NodeId, names: &[Vec<u8>], to: NodeId) {
        let quote = ByteSet::of(b"\"");
        let root = self.node();
        self.set_in_str(root);
        self.bytes(from, quote, root);
        let mut kids: std::collections::HashMap<(NodeId, u8), NodeId> = Default::default();
        let mut ends: Vec<NodeId> = Vec::new();
        for name in names {
            let mut at = root;
            for &b in name {
                at = match kids.get(&(at, b)) {
                    Some(&x) => x,
                    None => {
                        let x = self.node();
                        self.set_in_str(x);
                        self.bytes(at, ByteSet::of(&[b]), x);
                        kids.insert((at, b), x);
                        x
                    }
                };
            }
            if !ends.contains(&at) {
                self.bytes(at, quote, to);
                ends.push(at);
            }
        }
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
    pub fn finish(mut self) -> Result<Graph, String> {
        self.expand_spaces();
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
    pub object: NodeId,
}

/// Add the JSON library to `b`, its whitespace the builder's `Space`
/// (unrolled before it returns: its nodes are all in).
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
    let a_end = b.node();
    b.opens(array, ByteSet::of(b"["), a_open);
    b.ws_loop(a_open);
    b.bytes(a_open, ByteSet::of(b"]"), a_end);
    // The items: any value each, or (`Style::one_kind_arrays`) the first's
    // kind after it: an `a_after`/`a_val` pair per kind, entered from
    // `a_open` by the first item's first byte.
    let boolean = b.node();
    b.merge(boolean, lits[0]);
    b.merge(boolean, lits[1]);
    let kinds: Vec<NodeId> = if b.style.one_kind_arrays {
        vec![string, number, object, array, boolean, lits[2]]
    } else {
        vec![value]
    };
    for k in kinds {
        let a_after = b.node();
        let a_val = b.node();
        b.call(a_open, k, a_after);
        b.ws_loop(a_after);
        b.bytes(a_after, ByteSet::of(b","), a_val);
        b.bytes(a_after, ByteSet::of(b"]"), a_end);
        b.ws_loop(a_val);
        b.call(a_val, k, a_after);
    }
    b.set_ret(a_end);

    // A value: each kind's first byte straight into its body (tail calls:
    // the kind returns to whoever called `value`).
    for k in [string, number, object, array] {
        b.merge(value, k);
    }
    for l in lits {
        b.merge(value, l);
    }
    b.expand_spaces();
    Json { value, object }
}
