#!/usr/bin/env python3
"""harness_accept.py: the coding-agent acceptance test (docs/log/2026-11-13-harness-accept.md).

Three real harnesses (opencode, codex, pi) drive a model on a fixed set of small coding tasks, each with an automatic
pass/fail check, against `axon serve` and against llama-server (tuned for its side: flash attention, every layer on
the GPU, the same 32k context per request and KV type, a unified KV pool, larger batches, `--jinja` tool calling,
gpt-oss's experts on the GPU as far as they fit). Stdlib only, one file (axon-podjob stages it as is).

    harness_accept.py run --model qwen2.5-7b --engines axon,llama --harness pi --axon-bin AXON --llama-bin LLAMA \
        --out DIR [--tasks all|id,id] [--rounds N] [--turn-timeout S]
    harness_accept.py table LOG_OR_JSONL...       results table + engine-side failure counts (markdown)
    harness_accept.py selfcheck [--no-sandbox]    every task's check fails before and passes after its reference
                                                  solution; the proxy's parsing against a fake upstream (no model)
    harness_accept.py list                        the tasks

Per engine, one server process (restarted only after a crash) sits behind an in-process recording proxy. Per task the
proxy's records give: requests, HTTP errors (message kept, truncated: these are engine error strings), context
overflows, tool calls parsed, calls that came back as text (`<tool_call>` / JSON call / harmony syntax in content with
no parsed call: malformed or dropped), argument JSON that doesn't parse, calls the harness rejected (its tool result
says unknown tool / invalid arguments), TTFT to the first token-bearing event, and wall time. Axon's /metrics deltas
add `axon_tool_calls_dropped_total` and `axon_prompt_refused_total`. Output: one `RESULT {json}` line per task run on
stdout (and results.jsonl in --out), ENGINE lines per server start, a SUMMARY table at the end.

Safety: every harness, and every check that runs model-written code, runs under bubblewrap (no --no-sandbox): only
/usr, /etc, /opt, /sys read-only, an empty /home with the run directory bound in, private /tmp and pid namespace, a
clean environment (scratch HOME / XDG dirs, dummy keys). The network namespace is shared (the harness must reach the
proxy on 127.0.0.1), so a model could still reach the internet.
"""
import argparse, ast, csv, hashlib, http.client, http.server, io, json, os, re, shutil, signal, socket, socketserver
import statistics, subprocess, sys, tempfile, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESSES = ("opencode", "codex", "pi")
CTX = 32768

# Model profiles. KV: q8 for the dense models on both engines (Axon --kv q8, llama-server -ctk/-ctv q8_0); gpt-oss
# keeps f16 (Axon's gpt-oss path is f16 KV only). Axon serves a MoE one request at a time (--parallel is an error).
# sampler: the model card's recommended settings (generation_config.json / the README), given to BOTH servers as
# their defaults (no harness sends temperature or top_p, so the server's defaults are what they sample with). Without
# it the engines' own defaults differ (Axon serve: top_p 0.9, min_p 0; llama-server: top_p 0.95, min_p 0.05).
# Repeat penalty 1 (off) on both: Qwen2.5's card has 1.05 over all tokens, which neither engine's penalty matches.
# --sampler defaults runs each engine's own defaults instead (the out-of-the-box comparison).
MODELS = {
    "qwen2.5-7b": dict(file="Qwen2.5-7B-Instruct-Q4_K_M.gguf", moe=False,
                       sampler=dict(temp=0.7, top_k=20, top_p=0.8, min_p=0.0)),
    "qwen3-8b": dict(file="Qwen3-8B-Q4_K_M.gguf", moe=False,
                     sampler=dict(temp=0.6, top_k=20, top_p=0.95, min_p=0.0)),       # thinking mode
    "gpt-oss-20b": dict(file="gpt-oss-20b-MXFP4.gguf", moe=True,
                        sampler=dict(temp=1.0, top_k=0, top_p=1.0, min_p=0.0)),       # top_k 0 = off on both
    "qwen2.5-0.5b": dict(file="Qwen2.5-0.5B-Instruct-Q4_0.gguf", moe=False,          # plumbing smoke only
                         sampler=dict(temp=0.7, top_k=20, top_p=0.8, min_p=0.0)),
}


def sampler_flags(prof):
    """The same sampling flags for either server (axon serve and llama-server spell them alike)."""
    s = prof.get("sampler")
    if not s:
        return []
    return ["--temp", str(s["temp"]), "--top-k", str(s["top_k"]), "--top-p", str(s["top_p"]), "--min-p",
            str(s["min_p"]), "--repeat-penalty", "1.0"]

# ----------------------------------------------------------------------------------------------------------------
# Workspaces
# ----------------------------------------------------------------------------------------------------------------


_MR = []


def make_repo_mod():
    import importlib.util
    if _MR:
        return _MR[0]
    for p in (os.path.join(HERE, "examples", "harness-repo", "make_repo.py"), os.path.join(HERE, "make_repo.py"),
              os.path.join(os.getcwd(), "make_repo.py")):
        if os.path.exists(p):
            spec = importlib.util.spec_from_file_location("make_repo", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            _MR.append(m)
            return m
    sys.exit("harness_accept: make_repo.py not found (scripts/examples/harness-repo/ or next to this file)")


BUG_LINE = "    # BUG (planted): integer division truncates instead of rounding half up.\n" \
           "    return cents - (cents * pct) // 100\n"
FIXED_LINE = "    return cents - (cents * pct + 50) // 100\n"

TEXTKIT = {
    "textkit/__init__.py": '"""textkit: small text helpers."""\n',
    "textkit/core.py": '''"""Small text helpers."""
import re


def slugify(s):
    """Lower-case s, turn every run of characters that are not ASCII letters or digits into a single '-', and
    strip '-' from both ends. slugify('Hello, World!') == 'hello-world'."""
    raise NotImplementedError


def top_words(text, k):
    """The k most common words of text as a list of (word, count) tuples. Words are runs of ASCII letters,
    compared case-insensitively and returned lower-case. Order: higher count first, ties alphabetically.
    top_words('b a B c a b', 2) == [('b', 3), ('a', 2)]."""
    raise NotImplementedError


def wrap(text, width):
    """Greedy word wrap: split text on whitespace and pack the words into lines of at most width characters,
    words separated by one space. A word longer than width gets a line of its own. Returns the list of lines;
    no words -> []. wrap('a bb ccc', 4) == ['a bb', 'ccc']."""
    raise NotImplementedError
''',
    "tests/__init__.py": "",
    "tests/test_core.py": '''import unittest
from textkit.core import slugify, top_words, wrap


class TestCore(unittest.TestCase):
    def test_slugify(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")
        self.assertEqual(slugify("snake_case name"), "snake-case-name")

    def test_top_words(self):
        self.assertEqual(top_words("b a B c a b", 2), [("b", 3), ("a", 2)])

    def test_wrap(self):
        self.assertEqual(wrap("a bb ccc", 4), ["a bb", "ccc"])
        self.assertEqual(wrap("", 3), [])


if __name__ == "__main__":
    unittest.main()
''',
    "README.md": "# textkit\n\nSmall text helpers. Tests: `python3 -m unittest discover -s tests -t .`\n",
}


def write_files(d, files):
    for rel, src in files.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(src)


def rd(d, rel):
    try:
        with open(os.path.join(d, rel)) as f:
            return f.read()
    except OSError:
        return None


def wr(d, rel, s):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write(s)


def sub(d, rel, old, new, count=1):
    s = rd(d, rel)
    assert s is not None and old in s, (rel, old[:60])
    wr(d, rel, s.replace(old, new, count))


def build_base(d, base):
    """base: ledger-bug (the planted discount bug), ledger (bug fixed), ledger-sku (fixed + a planted SKU regex
    bug the tests don't cover), textkit (stubs)."""
    os.makedirs(d, exist_ok=True)
    if base == "textkit":
        write_files(d, TEXTKIT)
        return
    mr = make_repo_mod()
    old = sys.argv
    try:
        sys.argv = ["make_repo.py", d]
        mr.main()
    finally:
        sys.argv = old
    if base in ("ledger", "ledger-sku"):
        sub(d, "ledger/pricing.py", BUG_LINE, FIXED_LINE)
    if base == "ledger-sku":
        sub(d, "ledger/validate.py", 'SKU = re.compile(r"^[A-Z]{3}-\\d{3}$")', 'SKU = re.compile(r"^[A-Z]{3}\\d{3}$")')


def git_init(d, msg="initial"):
    env = dict(os.environ, GIT_AUTHOR_NAME="eval", GIT_AUTHOR_EMAIL="eval@axon.invalid", GIT_COMMITTER_NAME="eval",
               GIT_COMMITTER_EMAIL="eval@axon.invalid", GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    for c in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"], ["git", "commit", "-q", "-m", msg]):
        subprocess.run(c, cwd=d, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def count_tests(d):
    n = 0
    for root, _, files in os.walk(os.path.join(d, "tests")):
        for f in files:
            if f.endswith(".py"):
                n += len(re.findall(r"^\s*def test", rd(root, f) or "", re.M))
    return n


# ----------------------------------------------------------------------------------------------------------------
# Sandbox (bubblewrap) and running model-written code
# ----------------------------------------------------------------------------------------------------------------
SANDBOX = True


def sbx(binds, chdir):
    """bwrap argv: system dirs read-only, an empty /home with `binds` (rw), private /tmp and pid namespace."""
    if not SANDBOX:
        return []
    a = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/etc", "/etc", "--ro-bind-try", "/opt", "/opt",
         "--ro-bind-try", "/sys", "/sys", "--ro-bind-try", "/run/systemd/resolve", "/run/systemd/resolve",
         "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp", "--tmpfs", "/var/tmp", "--tmpfs", "/home",
         "--unshare-pid", "--unshare-ipc", "--die-with-parent", "--new-session"]
    for p in ("/bin", "/sbin", "/lib", "/lib64", "/lib32"):
        if os.path.islink(p):
            a += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            a += ["--ro-bind", p, p]
    for b in binds:
        a += ["--bind", b, b]
    return a + ["--chdir", chdir, "--"]


def clean_env(home, extra=None):
    e = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": home, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
         "TERM": "dumb", "TMPDIR": "/tmp", "XDG_CONFIG_HOME": home + "/.config", "XDG_DATA_HOME": home + "/.local/share",
         "XDG_STATE_HOME": home + "/.local/state", "XDG_CACHE_HOME": home + "/.cache", "XDG_RUNTIME_DIR": home + "/run",
         "NO_COLOR": "1", "CI": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    e.update(extra or {})
    return e


def run_code(ws, argv, timeout=120, inp=None):
    """Run model-written code (tests, the CLI) in ws, sandboxed. -> (rc, combined output)."""
    home = os.path.join(ws, ".chkhome")
    os.makedirs(home, exist_ok=True)
    try:
        r = subprocess.run(sbx([ws], ws) + argv, cwd=ws, env=clean_env(home), input=inp, capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def py(ws, code, timeout=60):
    rc, out = run_code(ws, ["python3", "-c", code], timeout)
    return rc == 0 and out.rstrip().endswith("CHECK-OK"), out.strip().splitlines()[-1:] if out.strip() else ["(no output)"]


def suite(ws, mod=None):
    argv = ["python3", "-m", "unittest"] + ([mod] if mod else ["discover", "-s", "tests", "-t", "."])
    rc, out = run_code(ws, argv, 180)
    return rc == 0, out


class Checks:
    def __init__(self, ws):
        self.ws, self.fails = ws, []

    def need(self, ok, what):
        if not ok:
            self.fails.append(what)
        return ok

    def tests_pass(self):
        ok, out = suite(self.ws)
        last = [ln for ln in out.strip().splitlines() if ln.strip()][-1:] or ["?"]
        return self.need(ok, "test suite fails (%s)" % last[0][:80])

    def hidden(self, code, what):
        ok, last = py(self.ws, code + "\nprint('CHECK-OK')\n")
        return self.need(ok, "%s: %s" % (what, (last[0] if last else "")[:120]))

    def unchanged(self, base, rels):
        ref = tempfile.mkdtemp(prefix="ha-ref-")
        try:
            build_base(ref, base)
            for rel in rels:
                self.need(rd(self.ws, rel) == rd(ref, rel), "%s was modified" % rel)
        finally:
            shutil.rmtree(ref, ignore_errors=True)

    def more_tests(self, base_n):
        return self.need(count_tests(self.ws) > base_n, "no new test (test count %d, base %d)" % (count_tests(self.ws), base_n))

    def cli(self, args, expect_rc=0, has=(), lacks=()):
        rc, out = run_code(self.ws, ["python3", "-m", "ledger.cli"] + args, 60)
        ok = rc == expect_rc and all(h in out for h in has) and not any(x in out for x in lacks)
        return self.need(ok, "ledger %s: rc %s, output %r" % (" ".join(args), rc, out.strip()[-120:]))

    def result(self):
        return (not self.fails, "; ".join(self.fails) or "ok")


SEED_DB = """
import sys
sys.path.insert(0, '.')
from ledger.models import Item, Order
from ledger.storage import Store
st = Store('DBFILE')
a = Order('a1', 'ann'); a.add(Item('AAA-001', 'pen', 100), 3); st.save(a)
b = Order('b1', 'bob', discount_pct=10); b.add(Item('BBB-002', 'mug', 500), 2); st.save(b)
c = Order('c1', 'ann'); c.add(Item('CCC-003', 'cup', 250), 1); st.save(c)
print('CHECK-OK')
""".replace("DBFILE", "chk.jsonl")
BASE_TESTS = 7  # def test_* in the ledger repo's tests


# ----------------------------------------------------------------------------------------------------------------
# The tasks. Each: id, base, prompts (a second prompt is sent with the harness's continue), check(ws) -> (ok, detail),
# solve(ws): the reference solution selfcheck applies.
# ----------------------------------------------------------------------------------------------------------------
TESTCMD = "python3 -m unittest discover -s tests -t ."
TASKS = []


def task(tid, base, prompts, kind):
    def deco(cls):
        TASKS.append(dict(id=tid, base=base, prompts=prompts if isinstance(prompts, list) else [prompts], kind=kind,
                          check=cls.check, solve=cls.solve))
        return cls
    return deco


@task("bugfix", "ledger-bug", f"The test suite fails. Run it with: {TESTCMD}   Find the bug in the ledger package, fix "
      "it with a minimal change, and rerun the tests until they all pass.", "fix a failing test")
class _:
    def check(ws):
        c = Checks(ws)
        c.tests_pass()
        c.unchanged("ledger-bug", ["tests/test_pricing.py"])
        c.hidden("from ledger.pricing import apply_discount as d\n"
                 "assert (d(999, 15), d(1001, 50), d(200, 25), d(5, 50), d(0, 30), d(7, 0)) == (849, 500, 150, 2, 0, 7)",
                 "apply_discount values")
        return c.result()

    def solve(ws):
        sub(ws, "ledger/pricing.py", BUG_LINE, FIXED_LINE)


@task("cli-bug", "ledger-sku", "Running `python3 -m ledger.cli --db orders.jsonl add o1 ann --item ABC-001 pen 150 2` "
      "fails with \"bad sku\", but ABC-001 is a valid SKU (the format is three capital letters, a dash and three "
      "digits). Find and fix the bug, then run that command again to confirm that it prints \"saved o1\".",
      "reproduce + fix a CLI bug")
class _:
    def check(ws):
        c = Checks(ws)
        c.cli(["--db", "chk.jsonl", "add", "o9", "ann", "--item", "ABC-001", "pen", "150", "2"], 0, ["saved o9"])
        c.hidden("from ledger.validate import check_sku as s\nassert s('XYZ-123') == 'XYZ-123'\n"
                 "for bad in ('ABC001', 'abc-001', 'AB-001', 'ABC-01', 'ABC-0012'):\n"
                 "    try:\n        s(bad)\n    except ValueError:\n        continue\n"
                 "    raise AssertionError('accepted ' + bad)", "check_sku still validates")
        c.tests_pass()
        return c.result()

    def solve(ws):
        sub(ws, "ledger/validate.py", 'SKU = re.compile(r"^[A-Z]{3}\\d{3}$")', 'SKU = re.compile(r"^[A-Z]{3}-\\d{3}$")')


@task("feature-csv", "ledger", "Add a CSV output format to the reports: a --format option on the 'report' command "
      "(choices: text, csv; default text) wired through ledger/cli.py, ledger/reports.py (report) and ledger/formats.py "
      "(render), with the CSV writer implemented in formats.py using the csv module. Add a test for the CSV output in "
      f"tests/test_reports.py and run the full test suite ({TESTCMD}).", "feature across 3 files")
class _:
    def check(ws):
        c = Checks(ws)
        c.hidden("import csv, io\nfrom ledger.formats import render\n"
                 "out = render([{'a': 1, 'b': 'x,y'}, {'a': 22, 'b': 'z'}], 'csv')\n"
                 "rows = [r for r in csv.reader(io.StringIO(out)) if r]\n"
                 "assert rows == [['a', 'b'], ['1', 'x,y'], ['22', 'z']], rows\n"
                 "assert 'a  b' in render([{'a': 1, 'b': 2}], 'text')", "render(rows, 'csv')")
        if c.hidden(SEED_DB, "seed db"):
            c.cli(["--db", "chk.jsonl", "report", "--format", "csv"], 0, ["order,customer,lines,subtotal,total", "a1,ann,1,300,360"])
            c.cli(["--db", "chk.jsonl", "report"], 0, ["subtotal"], ["order,customer"])
        c.more_tests(BASE_TESTS)
        c.tests_pass()
        return c.result()

    def solve(ws):
        sub(ws, "ledger/formats.py", '"""Output formats for reports. Each takes a list of row dicts with the same keys."""\n',
            '"""Output formats for reports. Each takes a list of row dicts with the same keys."""\nimport csv, io\n')
        sub(ws, "ledger/formats.py", 'FORMATS = {"text": as_text}',
            'def as_csv(rows):\n    buf = io.StringIO()\n    if rows:\n        w = csv.DictWriter(buf, fieldnames=list(rows[0]))\n'
            '        w.writeheader()\n        w.writerows(rows)\n    return buf.getvalue()\n\n\nFORMATS = {"text": as_text, "csv": as_csv}')
        sub(ws, "ledger/cli.py", '    r.add_argument("--rate", default="standard")\n',
            '    r.add_argument("--rate", default="standard")\n    r.add_argument("--format", choices=["text", "csv"], default="text")\n')
        sub(ws, "ledger/cli.py", "kind=args.kind, rate=args.rate", "kind=args.kind, fmt=args.format, rate=args.rate")
        sub(ws, "tests/test_reports.py", "    def test_customers_sorted_by_total(self):",
            "    def test_csv(self):\n        self.assertTrue(report(self.orders, fmt='csv').startswith('order,customer'))\n\n"
            "    def test_customers_sorted_by_total(self):")


@task("filter-resume", "ledger", [
    "Add a --customer NAME option to the 'report' command in ledger/cli.py that limits the report to that customer's "
    f"orders, for both --kind orders and --kind customers. Then run the test suite ({TESTCMD}).",
    "Now add a test for the --customer option (in tests/), mention the option in README.md, and run the full test "
    "suite again."], "feature, then a continued session")
class _:
    def check(ws):
        c = Checks(ws)
        if c.hidden(SEED_DB, "seed db"):
            c.cli(["--db", "chk.jsonl", "report", "--customer", "ann"], 0, ["a1", "c1"], ["b1"])
            c.cli(["--db", "chk.jsonl", "report", "--kind", "customers", "--customer", "bob"], 0, ["bob"], ["ann"])
            c.cli(["--db", "chk.jsonl", "report"], 0, ["a1", "b1", "c1"])
        c.need("--customer" in (rd(ws, "README.md") or ""), "README.md doesn't mention --customer")
        c.more_tests(BASE_TESTS)
        c.tests_pass()
        return c.result()

    def solve(ws):
        sub(ws, "ledger/cli.py", '    r.add_argument("--rate", default="standard")\n',
            '    r.add_argument("--rate", default="standard")\n    r.add_argument("--customer")\n')
        sub(ws, "ledger/cli.py", "        print(report(st.load_all(), kind=args.kind, rate=args.rate))",
            "        orders = [o for o in st.load_all() if args.customer is None or o.customer == args.customer]\n"
            "        print(report(orders, kind=args.kind, rate=args.rate))")
        wr(ws, "README.md", rd(ws, "README.md") + "\n`report --customer NAME` limits a report to one customer.\n")
        wr(ws, "tests/test_cli.py", "import io, os, tempfile, unittest\nfrom contextlib import redirect_stdout\n"
           "from ledger.cli import main\n\n\nclass TestCli(unittest.TestCase):\n    def test_customer(self):\n"
           "        db = os.path.join(tempfile.mkdtemp(), 'o.jsonl')\n"
           "        main(['--db', db, 'add', 'a1', 'ann', '--item', 'AAA-001', 'x', '100', '1'])\n"
           "        main(['--db', db, 'add', 'b1', 'bob', '--item', 'BBB-001', 'y', '100', '1'])\n"
           "        out = io.StringIO()\n        with redirect_stdout(out):\n"
           "            main(['--db', db, 'report', '--customer', 'ann'])\n"
           "        self.assertIn('a1', out.getvalue())\n        self.assertNotIn('b1', out.getvalue())\n")


@task("rename", "ledger", "Rename the function customer_rows in ledger/reports.py to rows_by_customer, and update every "
      f"use of it in the repository, tests included. Run the test suite ({TESTCMD}) afterwards.", "rename across files")
class _:
    def check(ws):
        c = Checks(ws)
        left = [os.path.relpath(os.path.join(r, f), ws) for r, _, fs in os.walk(ws) if "/." not in r
                for f in fs if f.endswith(".py") and "customer_rows" in (rd(r, f) or "")]
        c.need(not left, "customer_rows still in %s" % left)
        c.hidden(SEED_DB + "from ledger.reports import rows_by_customer, report\nfrom ledger.storage import Store\n"
                 "rows = rows_by_customer(Store('chk.jsonl').load_all())\n"
                 "assert [r['customer'] for r in rows] == ['bob', 'ann'], rows\n"
                 "assert 'bob' in report(Store('chk.jsonl').load_all(), kind='customers')", "rows_by_customer")
        c.need(count_tests(ws) >= BASE_TESTS, "tests were removed")
        c.tests_pass()
        return c.result()

    def solve(ws):
        for rel in ("ledger/reports.py", "tests/test_reports.py"):
            wr(ws, rel, rd(ws, rel).replace("customer_rows", "rows_by_customer"))


@task("spec-func", "ledger", "Add a function check_rate(s) to ledger/validate.py: it returns s unchanged if s is one of "
      "the tax rate names in ledger.pricing.TAX_RATES, and otherwise raises ValueError with a message that lists the "
      "valid names. Use it in ledger/cli.py to validate the --rate option of the report command. Run the tests "
      f"({TESTCMD}).", "implement to a spec")
class _:
    def check(ws):
        c = Checks(ws)
        c.hidden("from ledger.validate import check_rate as r\nassert r('reduced') == 'reduced' and r('zero') == 'zero'\n"
                 "try:\n    r('bogus')\nexcept ValueError as e:\n"
                 "    assert all(n in str(e) for n in ('standard', 'reduced', 'zero')), str(e)\n"
                 "else:\n    raise AssertionError('bogus accepted')", "check_rate")
        c.need("check_rate" in (rd(ws, "ledger/cli.py") or ""), "cli.py doesn't use check_rate")
        if c.hidden(SEED_DB, "seed db"):
            c.cli(["--db", "chk.jsonl", "report", "--rate", "reduced"], 0, ["a1"])
            rc, out = run_code(ws, ["python3", "-m", "ledger.cli", "--db", "chk.jsonl", "report", "--rate", "bogus"], 60)
            c.need(rc != 0, "report --rate bogus exits 0")
        c.tests_pass()
        return c.result()

    def solve(ws):
        wr(ws, "ledger/validate.py", rd(ws, "ledger/validate.py") +
           "\n\ndef check_rate(s):\n    from .pricing import TAX_RATES\n    if s not in TAX_RATES:\n"
           "        raise ValueError(f\"unknown rate {s!r}; valid: {', '.join(sorted(TAX_RATES))}\")\n    return s\n")
        sub(ws, "ledger/cli.py", "kind=args.kind, rate=args.rate", "kind=args.kind, rate=validate.check_rate(args.rate)")


@task("write-tests", "ledger", "Write unit tests for ledger/validate.py in a new file tests/test_validate.py: cover "
      "check_sku, check_qty and check_discount, each with valid inputs and with inputs it must reject. Run the full "
      f"test suite ({TESTCMD}) and make sure it passes.", "write tests (mutants must die)")
class _:
    MUTANTS = {"check_sku": "def check_sku(s):\n    return s\n", "check_qty": "def check_qty(s):\n    return int(s)\n",
               "check_discount": "def check_discount(s):\n    return int(s)\n"}

    def check(ws):
        c = Checks(ws)
        t = rd(ws, "tests/test_validate.py")
        if not c.need(t is not None, "tests/test_validate.py missing"):
            return c.result()
        c.need(len(re.findall(r"^\s*def test", t, re.M)) >= 3, "fewer than 3 tests in test_validate.py")
        c.tests_pass()
        src = rd(ws, "ledger/validate.py")
        for name, mut in _MUT.items():
            wr(ws, "ledger/validate.py", src + "\n\n" + mut)
            ok, _o = suite(ws, "tests.test_validate")
            c.need(not ok, "a %s that accepts everything passes the tests" % name)
        wr(ws, "ledger/validate.py", src)
        return c.result()

    def solve(ws):
        wr(ws, "tests/test_validate.py", "import unittest\nfrom ledger import validate as v\n\n\n"
           "class TestValidate(unittest.TestCase):\n"
           "    def test_sku(self):\n        self.assertEqual(v.check_sku('ABC-123'), 'ABC-123')\n"
           "        with self.assertRaises(ValueError):\n            v.check_sku('abc')\n\n"
           "    def test_qty(self):\n        self.assertEqual(v.check_qty('5'), 5)\n"
           "        with self.assertRaises(ValueError):\n            v.check_qty('0')\n\n"
           "    def test_discount(self):\n        self.assertEqual(v.check_discount('10'), 10)\n"
           "        with self.assertRaises(ValueError):\n            v.check_discount('101')\n")


_MUT = _.MUTANTS


@task("pyproject", "ledger", "Add a pyproject.toml for this package: project name \"ledger\", version \"0.3.1\" (the "
      "version in ledger/__init__.py), requires-python \">=3.9\", and a console script named ledger that runs "
      "ledger.cli:main. Use setuptools as the build backend.", "write a config file")
class _:
    def check(ws):
        c = Checks(ws)
        c.hidden("import tomllib\nd = tomllib.load(open('pyproject.toml', 'rb'))\np = d['project']\n"
                 "assert p['name'] == 'ledger' and p['version'] == '0.3.1', p\n"
                 "assert p['scripts']['ledger'] == 'ledger.cli:main', p.get('scripts')\n"
                 "assert '3.9' in p['requires-python'], p.get('requires-python')\n"
                 "assert 'setuptools' in d['build-system']['build-backend'], d.get('build-system')", "pyproject.toml")
        return c.result()

    def solve(ws):
        wr(ws, "pyproject.toml", '[build-system]\nrequires = ["setuptools>=61"]\nbuild-backend = "setuptools.build_meta"\n\n'
           '[project]\nname = "ledger"\nversion = "0.3.1"\nrequires-python = ">=3.9"\n\n'
           '[project.scripts]\nledger = "ledger.cli:main"\n')


@task("answer", "ledger", "Using this package's code (ledger.pricing and ledger.models), compute the total in cents, "
      "tax included, of one order with 3 units of an item priced 199 cents, a 10 percent order discount, at the "
      "'reduced' tax rate. Run Python to compute it rather than doing it by hand, and write only the resulting integer "
      "to a file named ANSWER.txt.", "run code, report a number")
class _:
    def check(ws):
        c = Checks(ws)
        a = (rd(ws, "ANSWER.txt") or "").strip()
        c.need(a == "564", "ANSWER.txt is %r, want 564" % a[:40])
        return c.result()

    def solve(ws):
        wr(ws, "ANSWER.txt", "564\n")


@task("align", "ledger", "In ledger/formats.py, change the text format (as_text) so that a column whose values are all "
      "integers is right-aligned, its header included, while other columns stay left-aligned. Keep everything else "
      f"the same, and run the test suite ({TESTCMD}).", "precise behaviour change")
class _:
    def check(ws):
        c = Checks(ws)
        c.hidden("from ledger.formats import as_text\n"
                 "L = [l.rstrip() for l in as_text([{'name': 'a', 'n': 5}, {'name': 'bbb', 'n': 123}]).splitlines()]\n"
                 "assert L[0] == 'name    n' and L[2] == 'a       5' and L[3] == 'bbb   123' and set(L[1]) == {'-'}, L\n"
                 "M = [l.rstrip() for l in as_text([{'n': 5, 's': 'x'}, {'n': 10, 's': 'yy'}]).splitlines()]\n"
                 "assert M[0] == ' n  s' and M[2] == ' 5  x' and M[3] == '10  yy', M\n"
                 "assert as_text([]) == '(no rows)'", "as_text alignment")
        c.tests_pass()
        return c.result()

    def solve(ws):
        s = rd(ws, "ledger/formats.py")
        a, b = s.index("def as_text(rows):"), s.index("FORMATS = ")
        wr(ws, "ledger/formats.py", s[:a] + '''def as_text(rows):
    if not rows:
        return "(no rows)"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    num = {k: all(isinstance(r[k], int) for r in rows) for k in keys}

    def cell(k, v):
        return str(v).rjust(widths[k]) if num[k] else str(v).ljust(widths[k])
    head = "  ".join(cell(k, k) for k in keys)
    body = ["  ".join(cell(k, r[k]) for k in keys) for r in rows]
    return "\\n".join([head, "-" * len(head)] + body)


''' + s[b:])


@task("delete-cmd", "ledger", "Add a 'delete ORDER_ID' command to the CLI. It removes that order from the store file "
      "(implement Store.delete(order_id) in ledger/storage.py, raising KeyError if there is no such order) and prints "
      "'deleted ORDER_ID'. For an order that doesn't exist it prints 'no such order: ORDER_ID' and exits with status 1. "
      f"Add a test, and run the test suite ({TESTCMD}).", "new command + storage method")
class _:
    def check(ws):
        c = Checks(ws)
        if c.hidden(SEED_DB, "seed db"):
            c.cli(["--db", "chk.jsonl", "delete", "b1"], 0, ["deleted b1"])
            c.cli(["--db", "chk.jsonl", "delete", "zz9"], 1, ["no such order: zz9"])
            c.cli(["--db", "chk.jsonl", "report"], 0, ["a1", "c1"], ["b1"])
        c.hidden(SEED_DB.replace("chk.jsonl", "chk2.jsonl") + "st = Store('chk2.jsonl')\nst.delete('a1')\n"
                 "assert sorted(o.order_id for o in st.load_all()) == ['b1', 'c1']\n"
                 "try:\n    st.delete('a1')\nexcept KeyError:\n    pass\nelse:\n    raise AssertionError('no KeyError')",
                 "Store.delete")
        c.more_tests(BASE_TESTS)
        c.tests_pass()
        return c.result()

    def solve(ws):
        sub(ws, "ledger/storage.py", "    def find(self, order_id):",
            "    def delete(self, order_id):\n        orders = self.load_all()\n"
            "        keep = [o for o in orders if o.order_id != order_id]\n        if len(keep) == len(orders):\n"
            "            raise KeyError(order_id)\n        open(self.path, 'w').close()\n        for o in keep:\n"
            "            self.save(o)\n\n    def find(self, order_id):")
        sub(ws, "ledger/cli.py", '    r = sub.add_parser("report")\n',
            '    dl = sub.add_parser("delete")\n    dl.add_argument("order_id")\n    r = sub.add_parser("report")\n')
        sub(ws, "ledger/cli.py", "    else:\n        print(report(",
            "    elif args.cmd == \"delete\":\n        try:\n            st.delete(args.order_id)\n        except KeyError:\n"
            "            print(f\"no such order: {args.order_id}\")\n            return 1\n"
            "        print(f\"deleted {args.order_id}\")\n    else:\n        print(report(")
        sub(ws, "tests/test_storage.py", "        self.assertEqual(back.discount_pct, 5)\n",
            "        self.assertEqual(back.discount_pct, 5)\n\n    def test_delete(self):\n"
            "        st = Store(os.path.join(tempfile.mkdtemp(), 'o.jsonl'))\n        o = Order('x1', 'a')\n"
            "        o.add(Item('QQQ-123', 'cup', 300), 1)\n        st.save(o)\n        st.delete('x1')\n"
            "        self.assertEqual(st.load_all(), [])\n")


@task("stubs", "textkit", "The functions in textkit/core.py are stubs. Implement them as their docstrings describe, so "
      f"that the tests pass ({TESTCMD}). Do not change the tests.", "implement from docstrings")
class _:
    def check(ws):
        c = Checks(ws)
        c.unchanged("textkit", ["tests/test_core.py"])
        c.tests_pass()
        c.hidden("from textkit.core import slugify as s, top_words as t, wrap as w\n"
                 "assert s('  A--b  ') == 'a-b' and s('x_y z!') == 'x-y-z' and s('Rock & Roll 2') == 'rock-roll-2'\n"
                 "assert t('The the THE cat. Cat dog', 3) == [('the', 3), ('cat', 2), ('dog', 1)], t('The the THE cat. Cat dog', 3)\n"
                 "assert t('', 3) == [] and t('b a', 5) == [('a', 1), ('b', 1)]\n"
                 "assert w('one two three four', 9) == ['one two', 'three', 'four']\n"
                 "assert w('abcdefgh ij', 4) == ['abcdefgh', 'ij'] and w('  ', 4) == []", "hidden cases")
        return c.result()

    def solve(ws):
        s = rd(ws, "textkit/core.py")
        impl = ["    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')\n",
                "    from collections import Counter\n    c = Counter(re.findall(r'[a-z]+', text.lower()))\n"
                "    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]\n",
                "    lines = []\n    for word in text.split():\n"
                "        if lines and len(lines[-1]) + 1 + len(word) <= width:\n            lines[-1] += ' ' + word\n"
                "        else:\n            lines.append(word)\n    return lines\n"]
        for i in impl:
            s = s.replace("    raise NotImplementedError\n", i, 1)
        wr(ws, "textkit/core.py", s)


@task("typehints", "ledger", "Add type hints to every function in ledger/pricing.py: every parameter and every return "
      f"type. Don't change the behaviour, and run the tests ({TESTCMD}).", "annotate a module")
class _:
    def check(ws):
        c = Checks(ws)
        try:
            tree = ast.parse(rd(ws, "ledger/pricing.py") or "")
            fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            bad = [f.name for f in fns if f.returns is None or any(
                x.annotation is None for x in f.args.posonlyargs + f.args.args + f.args.kwonlyargs)]
            c.need(len(fns) >= 3 and not bad, "unannotated: %s" % bad)
        except SyntaxError as e:
            c.need(False, "pricing.py: %s" % e)
        c.hidden("from ledger import pricing as p\nfrom ledger.models import Item, Order\n"
                 "o = Order('x', 'y', discount_pct=10); o.add(Item('AAA-001', 'p', 199), 3)\n"
                 "assert (p.apply_discount(999, 15), p.tax(537, 'reduced'), p.total(o, 'reduced')) == (849, 27, 564)",
                 "behaviour unchanged")
        c.tests_pass()
        return c.result()

    def solve(ws):
        s = rd(ws, "ledger/pricing.py")
        s = s.replace("def apply_discount(cents, pct):", "def apply_discount(cents: int, pct: int) -> int:")
        s = s.replace('def tax(cents, rate="standard"):', 'def tax(cents: int, rate: str = "standard") -> int:')
        s = s.replace('def total(order, rate="standard"):', 'def total(order: "Order", rate: str = "standard") -> int:')
        wr(ws, "ledger/pricing.py", s)


@task("exception", "ledger", "Make Store.find in ledger/storage.py raise a new exception class OrderNotFound when the "
      "order doesn't exist. Define OrderNotFound in ledger/storage.py as a subclass of KeyError, with the missing order "
      f"id in its message. Add a test for it and run the test suite ({TESTCMD}).", "new exception + test")
class _:
    def check(ws):
        c = Checks(ws)
        c.hidden(SEED_DB + "from ledger.storage import OrderNotFound\nassert issubclass(OrderNotFound, KeyError)\n"
                 "try:\n    Store('chk.jsonl').find('qq7')\nexcept OrderNotFound as e:\n    assert 'qq7' in str(e), str(e)\n"
                 "else:\n    raise AssertionError('no OrderNotFound')\nassert Store('chk.jsonl').find('a1').customer == 'ann'",
                 "OrderNotFound")
        c.more_tests(BASE_TESTS)
        c.tests_pass()
        return c.result()

    def solve(ws):
        sub(ws, "ledger/storage.py", "class Store:", "class OrderNotFound(KeyError):\n    pass\n\n\nclass Store:")
        sub(ws, "ledger/storage.py", "        raise KeyError(order_id)", "        raise OrderNotFound(f'no order {order_id}')")
        sub(ws, "tests/test_storage.py", "        self.assertEqual(back.discount_pct, 5)\n",
            "        self.assertEqual(back.discount_pct, 5)\n\n    def test_missing(self):\n"
            "        from ledger.storage import OrderNotFound\n"
            "        with self.assertRaises(OrderNotFound):\n            Store(os.path.join(tempfile.mkdtemp(), 'n.jsonl')).find('zz')\n")


TASK_BY_ID = {t["id"]: t for t in TASKS}


def run_check(t, ws):
    """The task's check on a copy of ws (the agent's tree stays as it left it)."""
    chk = ws.rstrip("/") + ".check"
    shutil.rmtree(chk, ignore_errors=True)
    shutil.copytree(ws, chk, symlinks=True, ignore=shutil.ignore_patterns(".git", "node_modules"))
    try:
        return t["check"](chk)
    except Exception as e:  # a check must never take the run down
        return False, "check crashed: %s: %s" % (type(e).__name__, e)
    finally:
        shutil.rmtree(chk, ignore_errors=True)


# ----------------------------------------------------------------------------------------------------------------
# The recording proxy
# ----------------------------------------------------------------------------------------------------------------
ROUTES = {"/v1/chat/completions": "chat", "/chat/completions": "chat", "/v1/responses": "resp", "/responses": "resp",
          "/v1/completions": "compl", "/api/chat": "ochat"}
HOP = {"connection", "keep-alive", "proxy-connection", "transfer-encoding", "te", "trailer", "upgrade", "content-length",
       "host"}
TEXT_TC = re.compile(r'(<tool_call>|<function=|"name"\s*:\s*"[A-Za-z0-9_.\-]+"\s*,\s*"(arguments|parameters)"\s*:|'
                     r'to=functions\.|<\|call\|>|<\|constrain\|>)')
ERR_HARNESS = re.compile(r"(?i)(unknown tool|no such tool|tool .{0,40}not found|invalid (tool|argument|input|param)|"
                         r"failed to parse|could not parse|missing required|validation error|unavailable tool|"
                         r"is not a valid tool|invalid_request)")
CTX_ERR = re.compile(r"(?i)(context|exceed|too long|n_ctx)")


class Rec:
    def __init__(self, route, body, label):
        self.d = dict(label, route=route, status=None, ttft_ms=None, total_ms=None, prompt_tokens=None,
                      completion_tokens=None, cached_tokens=None, tc=0, tc_badjson=0, tc_text=0, tc_text_r=0, empty=0,
                      finish=None, err=None)
        self.calls, self.text, self.rtext, self.first = {}, [], [], None
        b = body if isinstance(body, dict) else {}
        self.d.update(stream=bool(b.get("stream")), n_tools=len(b.get("tools") or []),
                      keys=sorted(b.keys())[:40], temperature=b.get("temperature"), top_p=b.get("top_p"),
                      max_tokens=b.get("max_tokens", b.get("max_completion_tokens", b.get("max_output_tokens"))),
                      tool_choice=b.get("tool_choice") if isinstance(b.get("tool_choice"), str) else
                      ("object" if b.get("tool_choice") else None))
        items = b.get("messages") if route != "resp" else ([b["input"]] if isinstance(b.get("input"), str) else b.get("input"))
        items = [m for m in (items or []) if isinstance(m, dict)]
        self.d["n_msgs"] = len(items)
        tail = []
        for m in items:
            if m.get("role") == "assistant" or m.get("type") in ("function_call", "message"):
                tail = []
            elif m.get("role") == "tool" or m.get("type") == "function_call_output":
                tail.append(m)
        errs = 0
        for m in tail:
            x = m.get("content", m.get("output", ""))
            x = x if isinstance(x, str) else json.dumps(x)
            errs += bool(ERR_HARNESS.search(x[:4000]))
        self.d.update(tool_res=len(tail), tool_res_err=errs)

    def tok(self, t):
        if self.first is None:
            self.first = t

    def event(self, o, t):
        if not isinstance(o, dict):
            return
        if isinstance(o.get("error"), (dict, str)) and self.d["err"] is None:
            e = o["error"]
            self.d["err"] = (e.get("code") or e.get("type") or "") + ": " + str(e.get("message"))[:300] if isinstance(e, dict) else str(e)[:300]
        if self.d["route"] == "resp":
            ty = o.get("type", "")
            if ty in ("response.output_text.delta",):
                self.text.append(o.get("delta") or ""), self.tok(t)
            elif ty.startswith("response.reasoning") and ty.endswith(".delta"):
                self.rtext.append(o.get("delta") or ""), self.tok(t)
            elif ty == "response.function_call_arguments.delta":
                self.tok(t)
            elif ty == "response.output_item.added" and (o.get("item") or {}).get("type") == "function_call":
                self.tok(t)
            resp = o.get("response") if isinstance(o.get("response"), dict) else (o if o.get("object") == "response" else None)
            if resp is not None and (ty in ("response.completed", "response.incomplete", "response.failed") or not ty):
                u = resp.get("usage") or {}
                self.d.update(prompt_tokens=u.get("input_tokens"), completion_tokens=u.get("output_tokens"),
                              cached_tokens=(u.get("input_tokens_details") or {}).get("cached_tokens"),
                              finish=resp.get("status"))
                txt = []
                for it in resp.get("output") or []:
                    if not isinstance(it, dict):
                        continue
                    if it.get("type") == "function_call":
                        self.calls[it.get("call_id") or len(self.calls)] = it.get("arguments") or ""
                    elif it.get("type") == "message":
                        txt += [c.get("text", "") for c in it.get("content") or [] if isinstance(c, dict)]
                if not self.text:
                    self.text += txt
                    if txt or self.calls:
                        self.tok(t)
            return
        u = o.get("usage")
        if isinstance(u, dict):
            self.d.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                          cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"))
        for ch in o.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            m = ch.get("delta") or ch.get("message") or {}
            if isinstance(m.get("content"), str) and m["content"]:
                self.text.append(m["content"]), self.tok(t)
            for k in ("reasoning_content", "reasoning"):
                if isinstance(m.get(k), str) and m[k]:
                    self.rtext.append(m[k]), self.tok(t)
            for i, tc in enumerate(m.get("tool_calls") or []):
                k = tc.get("index", tc.get("id", i))
                a = (tc.get("function") or {}).get("arguments")
                self.calls[k] = self.calls.get(k, "") + (a if isinstance(a, str) else "")
                self.tok(t)
            if ch.get("finish_reason"):
                self.d["finish"] = ch["finish_reason"]
            if isinstance(ch.get("text"), str) and ch["text"]:
                self.text.append(ch["text"]), self.tok(t)

    def close(self, status, t0, t_end, raw_err=None):
        bad = 0
        for a in self.calls.values():
            try:
                json.loads(a or "{}")
            except ValueError:
                bad += 1
        text, rtext = "".join(self.text), "".join(self.rtext)
        n = len(self.calls)
        self.d.update(status=status, tc=n, tc_badjson=bad, tc_text=int(n == 0 and bool(TEXT_TC.search(text))),
                      tc_text_r=int(n == 0 and bool(TEXT_TC.search(rtext))), empty=int(n == 0 and not text.strip()),
                      ttft_ms=None if self.first is None else int((self.first - t0) * 1000), total_ms=int((t_end - t0) * 1000))
        if status >= 400 and self.d["err"] is None:
            self.d["err"] = (raw_err or b"")[:300].decode("utf-8", "replace")
        # an error is a 4xx/5xx, or an error event inside a 200 stream
        self.d["is_err"] = int(status >= 400 or self.d["err"] is not None)
        self.d["ctx_overflow"] = int(bool(self.d["is_err"]) and bool(CTX_ERR.search(self.d["err"] or "")))
        return self.d


class Proxy:
    def __init__(self, up_port):
        self.up_port, self.recs, self.lock, self.label = up_port, [], threading.Lock(), {}
        px = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    out = b""
                    while True:
                        n = int(self.rfile.readline().strip().split(b";")[0], 16)
                        if n == 0:
                            self.rfile.readline()
                            return out
                        out += self.rfile.read(n)
                        self.rfile.readline()
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def _proxy(self):
                t0 = time.time()
                raw = self._body()
                route = ROUTES.get(self.path.split("?")[0])
                body = None
                if route and raw:
                    try:
                        body = json.loads(raw)
                    except ValueError:
                        pass
                rec = Rec(route, body, dict(px.label, t=round(t0, 3))) if route else None
                c = http.client.HTTPConnection("127.0.0.1", px.up_port, timeout=7200)
                hdrs = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
                hdrs["Content-Length"] = str(len(raw))
                try:
                    c.request(self.command, self.path, body=raw if raw or self.command in ("POST", "PUT") else None,
                              headers=hdrs)
                    r = c.getresponse()
                except OSError as e:
                    if rec:
                        rec.d["err"] = "upstream: %s" % type(e).__name__
                        px.add(rec.close(599, t0, time.time()))
                    try:
                        self.send_error(502, "upstream: %s" % type(e).__name__)
                    except OSError:
                        pass
                    return
                self.send_response(r.status)
                for k, v in r.getheaders():
                    if k.lower() not in HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                buf, whole, sse = b"", b"", False
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        now = time.time()
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                        if rec is None:
                            continue
                        if len(whole) < 1 << 22:
                            whole += chunk
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for ln in lines:
                            ln = ln.strip()
                            if ln.startswith(b"data:"):
                                sse = True
                                ln = ln[5:].strip()
                            elif ln.startswith(b"error:"):
                                sse = True
                                ln = ln[6:].strip()
                            if ln.startswith(b"{") and sse:
                                try:
                                    rec.event(json.loads(ln), now)
                                except ValueError:
                                    pass
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except OSError:
                    pass  # the harness hung up (timeout kill)
                c.close()
                if rec is None:
                    return
                t_end = time.time()
                tail = buf.strip()
                if tail.startswith(b"data:"):
                    sse, tail = True, tail[5:].strip()
                if sse and tail.startswith(b"{"):
                    try:
                        rec.event(json.loads(tail), t_end)
                    except ValueError:
                        pass
                if not sse:
                    try:
                        rec.event(json.loads(whole), t_end)
                    except ValueError:
                        pass
                px.add(rec.close(r.status, t0, t_end, whole if r.status >= 400 else None))

            do_POST = do_GET = do_PUT = do_DELETE = _proxy

        class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.srv = S(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def add(self, d):
        with self.lock:
            self.recs.append(d)

    def take(self):
        with self.lock:
            r, self.recs = self.recs, []
        return r

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


# ----------------------------------------------------------------------------------------------------------------
# Engines
# ----------------------------------------------------------------------------------------------------------------
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_get(port, path, timeout=5):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d%s" % (port, path), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return None, b""


def http_post(port, path, obj, timeout=600):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path), data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError as e:
        return None, str(e).encode()


def vram_used():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "?"


class Engine:
    def __init__(self, kind, binp, model_path, alias, prof, outdir):
        self.kind, self.bin, self.model, self.alias, self.prof, self.outdir = kind, binp, model_path, alias, prof, outdir
        self.proc, self.port, self.starts, self.logf, self.argv = None, None, 0, None, None

    def variants(self):
        """The argv ladder: the first that comes up and answers a request is used."""
        p, moe = self.port, self.prof["moe"]
        if self.kind == "axon":
            a = [self.bin, "serve", self.model, "--device", "cuda", "--host", "127.0.0.1", "--port", str(p),
                 "--ctx", str(CTX), "--alias", self.alias]
            if not moe:
                a += ["--kv", "q8", "--parallel", "4"]
            return [a + sampler_flags(self.prof)]
        base = [self.bin, "-m", self.model, "--host", "127.0.0.1", "--port", str(p), "-a", self.alias, "-ngl", "999",
                "-fa", "on", "--jinja", "-b", "2048", "-ub", "2048", "-kvu", "--kv-unified-per-slot", str(CTX),
                "--cache-reuse", "256", "--metrics", "--no-webui"] + sampler_flags(self.prof)
        if not moe:
            return [base + ["-ctk", "q8_0", "-ctv", "q8_0", "-np", "4", "-c", str(c)] for c in (131072, 98304, 65536, 49152, 32768)]
        return [base + ["-np", "2", "-c", str(2 * CTX), "--n-cpu-moe", str(n)] for n in (4, 6, 8, 10, 12, 14, 16, 18, 20, 24)]

    def start(self):
        t0 = time.time()
        self.port = free_port()
        tried = []
        for argv in self.variants():
            self.starts += 1
            self.logf = os.path.join(self.outdir, "engine-%s-%d.log" % (self.kind, self.starts))
            lf = open(self.logf, "w")
            self.proc = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True)
            lf.close()
            ok = False
            for _ in range(600):
                if self.proc.poll() is not None:
                    break
                st, _b = http_get(self.port, "/health")
                if st == 200:
                    ok = True
                    break
                time.sleep(0.5)
            if ok:
                st, b = http_post(self.port, "/v1/chat/completions", {"model": self.alias, "max_tokens": 8, "messages": [
                    {"role": "user", "content": "Say hi."}]}, timeout=900)
                ok = st == 200
                if not ok:
                    tried.append("%s: warm-up %s %s" % (" ".join(argv[-4:]), st, b[:120]))
            else:
                tried.append("%s: exit %s (%s)" % (" ".join(argv[-4:]), self.proc.poll(), tail(self.logf, 2)))
            if ok:
                self.argv = argv
                return dict(ready_s=round(time.time() - t0, 1), argv=" ".join(os.path.basename(x) if i < 1 else x
                            for i, x in enumerate(argv)), vram=vram_used(), tried=tried)
            self.stop()
        raise RuntimeError("engine %s never came up: %s" % (self.kind, " | ".join(tried)[-1500:]))

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                self.proc.wait()
        self.proc = None
        time.sleep(2)

    def metrics(self):
        st, b = http_get(self.port, "/metrics")
        out = {}
        if st == 200:
            for ln in b.decode("utf-8", "replace").splitlines():
                m = re.match(r"^(axon_tool_calls_dropped_total|axon_prompt_refused_total|axon_prompt_tokens_total|"
                             r"axon_prompt_cached_tokens_total|llamacpp:prompt_tokens_total)\s+([0-9.e+]+)", ln)
                if m:
                    out[m.group(1)] = float(m.group(2))
        return out


def tail(path, n):
    try:
        with open(path, errors="replace") as f:
            return " / ".join(ln.strip() for ln in f.readlines()[-n:])[-600:]
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------------------------
# Harnesses
# ----------------------------------------------------------------------------------------------------------------
def setup_home(h, home, alias, px_port):
    """Scratch HOME with the harness's config, pointing at the proxy. Real configs are never read or written."""
    url = "http://127.0.0.1:%d/v1" % px_port
    for d in (".config", ".local/share", ".local/state", ".cache", "run"):
        os.makedirs(os.path.join(home, d), exist_ok=True)
    os.chmod(os.path.join(home, "run"), 0o700)
    if h == "pi":
        wr(home, "pi-agent/models.json", json.dumps({"providers": {"local": {
            "baseUrl": url, "api": "openai-completions", "apiKey": "dummy",
            "compat": {"supportsDeveloperRole": False, "supportsReasoningEffort": False},
            "models": [{"id": alias, "name": alias, "contextWindow": CTX, "maxTokens": 4096}]}}}, indent=1))
    elif h == "opencode":
        wr(home, ".config/opencode/opencode.json", json.dumps({
            "$schema": "https://opencode.ai/config.json", "model": "local/" + alias, "small_model": "local/" + alias,
            "share": "disabled", "autoupdate": False,
            "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "local",
                                   "options": {"baseURL": url, "apiKey": "dummy"},
                                   "models": {alias: {"name": alias, "limit": {"context": CTX, "output": 4096}}}}}}, indent=1))
    elif h == "codex":
        wr(home, ".codex/config.toml", 'model = "%s"\nmodel_provider = "local"\napproval_policy = "never"\n'
           'sandbox_mode = "workspace-write"\nmodel_context_window = %d\n\n[model_providers.local]\nname = "local"\n'
           'base_url = "%s"\nwire_api = "responses"\nenv_key = "LOCAL_DUMMY_KEY"\n' % (alias, CTX, url))


def harness_argv(h, alias, home, ws, cont, prompt):
    if h == "pi":
        env = {"PI_CODING_AGENT_DIR": home + "/pi-agent", "PI_OFFLINE": "1", "PI_TELEMETRY": "0",
               "PI_SKIP_VERSION_CHECK": "1"}
        return env, ["pi", "--offline", "--provider", "local", "--model", alias, "--api-key", "dummy",
                     "--session-dir", ws + "/.pi-sessions"] + (["-c"] if cont else []) + ["-p", prompt]
    if h == "opencode":
        return {"OPENCODE_DISABLE_AUTOUPDATE": "1"}, ["opencode", "run", "--standalone", "--auto", "-m",
                                                       "local/" + alias] + (["-c"] if cont else []) + [prompt]
    env = {"CODEX_HOME": home + "/.codex", "LOCAL_DUMMY_KEY": "dummy"}
    if cont:
        return env, ["codex", "exec", "resume", "--last", "--skip-git-repo-check", prompt]
    return env, ["codex", "exec", "--skip-git-repo-check", "-s", "workspace-write", prompt]


def run_turn(h, alias, root, home, ws, cont, prompt, timeout, outp):
    env, argv = harness_argv(h, alias, home, ws, cont, prompt)
    t0 = time.time()
    with open(outp, "w") as f:
        try:
            p = subprocess.Popen(sbx([root], ws) + argv, cwd=ws, env=clean_env(home, env), stdin=subprocess.DEVNULL,
                                 stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = p.wait(timeout)
                to = False
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rc, to = 124, True
        except OSError as e:
            f.write("harness_accept: %s\n" % e)
            rc, to = 127, False
    return rc, to, round(time.time() - t0, 1)


# ----------------------------------------------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------------------------------------------
def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def summarize(recs, m0, m1):
    st = [r["status"] for r in recs]
    errs = [r for r in recs if r.get("is_err")]
    okr = [r for r in recs if not r.get("is_err")]
    md = lambda k: (m1.get(k, 0) - m0.get(k, 0)) if k in m1 else None
    return dict(
        requests=len(recs), http_err=sum(1 for r in errs if not r.get("ctx_overflow")),
        ctx_overflow=sum(r.get("ctx_overflow", 0) for r in errs), upstream_fail=st.count(599),
        tc=sum(r["tc"] for r in recs), tc_text=sum(r["tc_text"] for r in recs), tc_text_r=sum(r["tc_text_r"] for r in recs),
        tc_badjson=sum(r["tc_badjson"] for r in recs), tool_res_err=sum(r["tool_res_err"] for r in recs),
        empty=sum(r["empty"] for r in okr), length=sum(1 for r in recs if r["finish"] in ("length", "incomplete")),
        dropped=md("axon_tool_calls_dropped_total"), refused=md("axon_prompt_refused_total"),
        ttft_p50=pct([r["ttft_ms"] for r in okr], 0.5),
        ttft_first=next((r["ttft_ms"] for r in okr if r["ttft_ms"] is not None), None),
        ttft_max=pct([r["ttft_ms"] for r in okr], 1.0),
        prompt_max=max([r["prompt_tokens"] or 0 for r in recs] or [0]),
        errors=sorted({(r.get("err") or "")[:200] for r in errs})[:5],
        keys=sorted({k for r in recs for k in r.get("keys", [])}),
        temps=sorted({str(r.get("temperature")) for r in recs}))


def cmd_run(a):
    global SANDBOX
    SANDBOX = not a.no_sandbox
    if SANDBOX and not shutil.which("bwrap"):
        sys.exit("harness_accept: bwrap not found (needed to sandbox the harnesses); --no-sandbox to run without")
    prof = dict(MODELS[a.model])
    if a.sampler == "defaults":
        prof.pop("sampler", None)
    mpath = a.model_path or os.path.join(os.environ.get("AXON_MODELS", "models"), prof["file"])
    mpath = os.path.abspath(mpath)
    if not os.path.exists(mpath):
        sys.exit("harness_accept: model not found: %s" % mpath)
    tasks = TASKS if a.tasks == "all" else [TASK_BY_ID[x] for x in a.tasks.split(",")]
    harn = a.harness.split(",")
    engines = a.engines.split(",")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    resf = open(os.path.join(out, "results.jsonl"), "a")
    reqf = open(os.path.join(out, "requests.jsonl"), "a")
    bins = {"axon": a.axon_bin, "llama": a.llama_bin}
    sha = {}
    for k in engines:
        if not bins.get(k) or not os.path.exists(bins[k]):
            sys.exit("harness_accept: --%s-bin missing" % k)
        with open(bins[k], "rb") as f:
            sha[k] = hashlib.sha256(f.read()).hexdigest()[:12]
    vers = {}
    vh = os.path.join(out, "verhome")
    os.makedirs(vh, exist_ok=True)
    for h in harn:
        try:
            r = subprocess.run([h, "--version"], capture_output=True, text=True, timeout=60, env=clean_env(vh),
                               stdin=subprocess.DEVNULL)
            vers[h] = ((r.stdout.strip() or r.stderr.strip()).splitlines() or ["?"])[-1] if r.returncode == 0 else "?"
        except (OSError, subprocess.TimeoutExpired):
            vers[h] = "missing"
            sys.exit("harness_accept: harness %s not runnable" % h)
    me = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:12]
    print("ACCEPT start model=%s file=%s engines=%s harness=%s tasks=%d rounds=%d turn_timeout=%ds sandbox=%s script=%s "
          "bins=%s harness_versions=%s sampler=%s" % (a.model, os.path.basename(mpath), engines, harn, len(tasks),
                                                      a.rounds, a.turn_timeout, SANDBOX, me, sha, vers,
                                                      prof.get("sampler", "engine defaults")), flush=True)
    results = []
    for ek in engines:
        eng = Engine(ek, os.path.abspath(bins[ek]), mpath, a.model, prof, out)
        try:
            info = eng.start()
        except RuntimeError as e:
            print("ENGINE %s FAILED %s" % (ek, e), flush=True)
            for h in harn:
                for t in tasks:
                    r = dict(engine=ek, harness=h, task=t["id"], model=a.model, round=1, ok=False, detail="engine failed to start",
                             crash=False, engine_fail=True, wall_s=0, rc=None, timed_out=False, engine_sha=sha[ek])
                    results.append(r)
                    print("RESULT " + json.dumps(r), flush=True)
            continue
        print("ENGINE %s up in %ss: %s (vram %s)%s" % (ek, info["ready_s"], info["argv"], info["vram"],
              (" tried: " + " | ".join(info["tried"])) if info["tried"] else ""), flush=True)
        px = Proxy(eng.port)
        for h in harn:
            root = os.path.join(out, "%s-%s" % (ek, h))
            home = os.path.join(root, "home")
            os.makedirs(home, exist_ok=True)
            setup_home(h, home, a.model, px.port)
            for rnd in range(1, a.rounds + 1):
                for t in tasks:
                    crash = False
                    if not eng.alive():
                        print("ENGINE %s DIED before %s/%s (rc %s): %s" % (ek, h, t["id"], eng.proc and eng.proc.poll(),
                              tail(eng.logf, 3)), flush=True)
                        eng.stop()
                        try:
                            info = eng.start()
                        except RuntimeError as e:
                            print("ENGINE %s RESTART FAILED %s" % (ek, e), flush=True)
                            r = dict(engine=ek, harness=h, task=t["id"], model=a.model, round=rnd, ok=False,
                                     detail="engine restart failed", crash=True, engine_fail=True, wall_s=0, rc=None,
                                     timed_out=False, engine_sha=sha[ek])
                            results.append(r)
                            print("RESULT " + json.dumps(r), flush=True)
                            continue
                        px.up_port = eng.port
                        print("ENGINE %s restarted in %ss" % (ek, info["ready_s"]), flush=True)
                    ws = os.path.join(root, "r%d-%s" % (rnd, t["id"]))
                    shutil.rmtree(ws, ignore_errors=True)
                    build_base(ws, t["base"])
                    git_init(ws, "ledger: %s r%d (%s %s)" % (t["id"], rnd, ek, h))
                    m0 = eng.metrics()
                    px.take()
                    t0 = time.time()
                    turns = []
                    for ti, prompt in enumerate(t["prompts"]):
                        px.label = dict(engine=ek, harness=h, task=t["id"], round=rnd, turn=ti)
                        rc, to, secs = run_turn(h, a.model, root, home, ws, ti > 0, prompt, a.turn_timeout,
                                                os.path.join(root, "r%d-%s-t%d.out" % (rnd, t["id"], ti)))
                        turns.append(dict(rc=rc, timed_out=to, s=secs))
                        if not eng.alive():
                            crash = True
                            break
                    wall = round(time.time() - t0, 1)
                    time.sleep(0.5)
                    recs = px.take()
                    for r in recs:
                        reqf.write(json.dumps(r) + "\n")
                    reqf.flush()
                    m1 = eng.metrics() if eng.alive() else m0
                    ok, detail = run_check(t, ws)
                    res = dict(engine=ek, harness=h, task=t["id"], model=a.model, round=rnd, ok=ok, detail=detail[:400],
                               wall_s=wall, rc=turns[-1]["rc"], timed_out=any(x["timed_out"] for x in turns),
                               turns=turns, crash=crash, engine_sha=sha[ek])
                    res.update(summarize(recs, m0, m1))
                    if crash:
                        res["crash_log"] = tail(eng.logf, 8)
                    results.append(res)
                    resf.write(json.dumps(res) + "\n")
                    resf.flush()
                    print("RESULT " + json.dumps(res), flush=True)
        px.stop()
        eng.stop()
    print_summary(results)
    return 0


def print_summary(results):
    print("SUMMARY (pass / runs; malformed = tool calls as text + bad JSON args + axon dropped)")
    cells = {}
    for r in results:
        cells.setdefault((r["model"], r["engine"], r["harness"]), []).append(r)
    print("%-12s %-6s %-9s %6s %5s %5s %5s %5s %5s %8s %8s" % ("model", "engine", "harness", "pass", "malf", "ctx",
                                                                "herr", "crash", "tmo", "ttft50", "wall50"))
    for k in sorted(cells):
        rs = cells[k]
        malf = sum((r.get("tc_text") or 0) + (r.get("tc_badjson") or 0) + int(r.get("dropped") or 0) for r in rs)
        print("%-12s %-6s %-9s %3d/%-2d %5d %5d %5d %5d %5d %8s %8s" % (
            k[0], k[1], k[2], sum(r["ok"] for r in rs), len(rs), malf, sum(r.get("ctx_overflow") or 0 for r in rs),
            sum(r.get("http_err") or 0 for r in rs), sum(bool(r.get("crash")) for r in rs),
            sum(bool(r.get("timed_out")) for r in rs), pct([r.get("ttft_p50") for r in rs], 0.5),
            pct([r.get("wall_s") for r in rs], 0.5)), flush=True)


# ----------------------------------------------------------------------------------------------------------------
# table: aggregate RESULT lines from job logs / results.jsonl
# ----------------------------------------------------------------------------------------------------------------
def load_results(paths):
    out, seen = [], set()
    for p in paths:
        with open(p, errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if ln.startswith("RESULT "):
                    ln = ln[7:]
                if not ln.startswith("{"):
                    continue
                try:
                    r = json.loads(ln)
                except ValueError:
                    continue
                if "task" not in r or "engine" not in r:
                    continue
                k = (r["model"], r["engine"], r["harness"], r["task"], r.get("round"), r.get("engine_sha"))
                if k in seen:
                    continue
                seen.add(k)
                out.append(r)
    return out


def norm_err(e):
    e = re.sub(r"\d+", "N", e or "")
    return e[:110]


def cmd_table(a):
    rs = load_results(a.paths)
    if not rs:
        sys.exit("no RESULT lines")
    cells = {}
    for r in rs:
        cells.setdefault((r["model"], r["harness"], r["engine"]), []).append(r)
    print("| model | harness | engine | tasks passed | malformed / dropped calls | ctx overflows | other HTTP errors | "
          "crashes | timeouts | TTFT p50 (ms) | wall p50 (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in sorted(cells):
        x = cells[k]
        malf = sum((r.get("tc_text") or 0) + (r.get("tc_badjson") or 0) + int(r.get("dropped") or 0) for r in x)
        print("| %s | %s | %s | %d / %d | %d (%d runs) | %d | %d | %d | %d | %s | %s |" % (
            k[0], k[1], k[2], sum(r["ok"] for r in x), len(x), malf,
            sum(1 for r in x if (r.get("tc_text") or 0) + (r.get("tc_badjson") or 0) + int(r.get("dropped") or 0)),
            sum(r.get("ctx_overflow") or 0 for r in x), sum(r.get("http_err") or 0 for r in x),
            sum(bool(r.get("crash")) for r in x), sum(bool(r.get("timed_out")) for r in x),
            pct([r.get("ttft_p50") for r in x], 0.5), pct([r.get("wall_s") for r in x], 0.5)))
    # per model x engine totals and the majority verdict
    print("\n| model | engine | passed (all harnesses) | best harness |")
    print("|---|---|---|---|")
    tot = {}
    for r in rs:
        tot.setdefault((r["model"], r["engine"]), []).append(r)
    for k in sorted(tot):
        x = tot[k]
        byh = {}
        for r in x:
            byh.setdefault(r["harness"], []).append(r["ok"])
        best = max(byh.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        print("| %s | %s | %d / %d | %s %d / %d |" % (k[0], k[1], sum(r["ok"] for r in x), len(x), best[0], sum(best[1]), len(best[1])))
    # per task
    print("\n| task | " + " | ".join("%s %s" % (m, e) for m, e in sorted(tot)) + " |")
    print("|---" * (len(tot) + 1) + "|")
    for t in [t["id"] for t in TASKS]:
        row = []
        for k in sorted(tot):
            x = [r for r in tot[k] if r["task"] == t]
            row.append("%d / %d" % (sum(r["ok"] for r in x), len(x)) if x else "-")
        print("| %s | %s |" % (t, " | ".join(row)))
    # engine-side failure counts: runs affected, Axon vs llama-server
    print("\n| failure (engine-side signal) | runs on axon | runs on llama |")
    print("|---|---|---|")
    kinds = {
        "tool call returned as text (malformed/unparsed)": lambda r: (r.get("tc_text") or 0) > 0,
        "tool call with unparseable JSON arguments": lambda r: (r.get("tc_badjson") or 0) > 0,
        "axon_tool_calls_dropped_total > 0": lambda r: (r.get("dropped") or 0) > 0,
        "context overflow (400)": lambda r: (r.get("ctx_overflow") or 0) > 0,
        "other HTTP error": lambda r: (r.get("http_err") or 0) > 0,
        "engine crash": lambda r: bool(r.get("crash")),
        "empty reply (no text, no call)": lambda r: (r.get("empty") or 0) > 0,
        "harness rejected a call (tool result error)": lambda r: (r.get("tool_res_err") or 0) > 0,
        "hit max_tokens / incomplete": lambda r: (r.get("length") or 0) > 0,
        "task timed out": lambda r: bool(r.get("timed_out")),
    }
    for name, f in kinds.items():
        n = {e: sum(1 for r in rs if r["engine"] == e and f(r)) for e in ("axon", "llama")}
        print("| %s | %d | %d |" % (name, n["axon"], n["llama"]))
    errs = {}
    for r in rs:
        for e in r.get("errors") or []:
            errs.setdefault((r["engine"], norm_err(e)), set()).add((r["model"], r["harness"], r["task"], r.get("round")))
    if errs:
        print("\nHTTP error messages (runs):")
        for (e, m), runs in sorted(errs.items(), key=lambda kv: -len(kv[1])):
            print("- %s: %d: `%s`" % (e, len(runs), m.replace("`", "'")))
    # pairs: same model/harness/task/round, one engine passed and the other failed
    by = {}
    for r in rs:
        by.setdefault((r["model"], r["harness"], r["task"], r.get("round")), {})[r["engine"]] = r
    ax_only = [k for k, v in by.items() if "axon" in v and "llama" in v and v["axon"]["ok"] and not v["llama"]["ok"]]
    ll_only = [k for k, v in by.items() if "axon" in v and "llama" in v and v["llama"]["ok"] and not v["axon"]["ok"]]
    print("\npaired runs: passed only on axon %d, only on llama %d (of %d pairs)" % (len(ax_only), len(ll_only),
          sum(1 for v in by.values() if len(v) == 2)))
    for k in ll_only:
        print("- llama only: %s / %s / %s r%s: axon: %s" % (k[0], k[1], k[2], k[3], by[k]["axon"]["detail"][:140]))
    return 0


# ----------------------------------------------------------------------------------------------------------------
# selfcheck
# ----------------------------------------------------------------------------------------------------------------
def fake_upstream():
    """A canned upstream for the proxy's self-test."""
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            mode = body.get("mode")
            if mode == "ctx":
                b = json.dumps({"error": {"code": "context_length_exceeded", "message": "prompt is 40000 tokens"}}).encode()
                self.send_response(400)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def ev(o):
                d = b"data: " + json.dumps(o).encode() + b"\n\n"
                self.wfile.write(b"%x\r\n%s\r\n" % (len(d), d))
                self.wfile.flush()
            if self.path == "/v1/responses":
                ev({"type": "response.created", "response": {"id": "r1", "object": "response"}})
                time.sleep(0.2)
                ev({"type": "response.output_item.added", "item": {"type": "function_call"}})
                ev({"type": "response.completed", "response": {"object": "response", "status": "completed",
                    "usage": {"input_tokens": 50, "output_tokens": 5}, "output": [
                        {"type": "function_call", "call_id": "c1", "arguments": "{\"cmd\": \"ls\"}"}]}})
            else:
                ev({"choices": [{"delta": {"role": "assistant"}}]})
                time.sleep(0.2)
                if mode == "text_tc":
                    ev({"choices": [{"delta": {"content": "<tool_call>{\"name\": \"bash\", \"arguments\": {\"c\""}}]})
                    ev({"choices": [{"delta": {}, "finish_reason": "stop"}]})
                else:
                    ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "x", "function": {"name": "bash", "arguments": "{\"c\":"}}]}}]})
                    ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": " 1}" if mode != "badjson" else " 1"}}]}}]})
                    ev({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}})
            d = b"data: [DONE]\n\n"
            self.wfile.write(b"%x\r\n%s\r\n0\r\n\r\n" % (len(d), d))
            self.wfile.flush()

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
    s = S(("127.0.0.1", 0), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def cmd_selfcheck(a):
    global SANDBOX
    SANDBOX = not a.no_sandbox and bool(shutil.which("bwrap"))
    print("selfcheck: sandbox %s" % SANDBOX)
    bad = 0
    for t in TASKS:
        d = tempfile.mkdtemp(prefix="ha-self-")
        ws = os.path.join(d, "ws")
        try:
            build_base(ws, t["base"])
            ok0, det0 = run_check(t, ws)
            t["solve"](ws)
            ok1, det1 = run_check(t, ws)
            good = (not ok0) and ok1
            bad += not good
            print("%-14s %s  before: %s | after: %s" % (t["id"], "OK  " if good else "FAIL", det0[:90], det1[:90]))
        finally:
            shutil.rmtree(d, ignore_errors=True)
    # the write-tests check must fail a test file that tests nothing useful
    d = tempfile.mkdtemp(prefix="ha-self-")
    try:
        build_base(d + "/ws", "ledger")
        wr(d + "/ws", "tests/test_validate.py", "import unittest\nfrom ledger import validate as v\n\n\nclass T(unittest.TestCase):\n"
           "    def test_a(self):\n        v.check_sku('ABC-123')\n\n    def test_b(self):\n        v.check_qty('3')\n\n"
           "    def test_c(self):\n        v.check_discount('3')\n")
        ok, det = run_check(TASK_BY_ID["write-tests"], d + "/ws")
        print("%-14s %s  weak tests rejected: %s" % ("write-tests-2", "OK  " if not ok else "FAIL", det[:120]))
        bad += ok
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # the proxy
    up = fake_upstream()
    px = Proxy(up.server_address[1])
    px.label = dict(task="self")
    cases = [("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "x"}], "tools": [{}]},
              dict(status=200, tc=1, tc_badjson=0, tc_text=0)),
             ("/v1/chat/completions", {"stream": True, "mode": "badjson", "messages": [{"role": "tool", "content": "Error: unknown tool foo"}]},
              dict(status=200, tc=1, tc_badjson=1, tool_res_err=1)),
             ("/v1/chat/completions", {"stream": True, "mode": "text_tc", "messages": []}, dict(status=200, tc=0, tc_text=1, empty=0)),
             ("/v1/chat/completions", {"mode": "ctx", "messages": []}, dict(status=400, ctx_overflow=1)),
             ("/v1/responses", {"stream": True, "input": "hi"}, dict(status=200, tc=1, prompt_tokens=50))]
    for path, body, want in cases:
        st, b = http_post(px.port, path, body, timeout=10)
        r = []
        for _ in range(40):  # the record lands just after the client has the whole response
            r = px.take()
            if r:
                break
            time.sleep(0.05)
        got = r[0] if r else {}
        ok = st == want["status"] and all(got.get(k) == v for k, v in want.items())
        if path == "/v1/responses" or body.get("stream"):
            ok = ok and got.get("ttft_ms") is not None and got["ttft_ms"] >= 150  # the first token, not the first byte
        bad += not ok
        print("proxy %-22s %-8s %s  %s" % (path, body.get("mode") or "", "OK  " if ok else "FAIL",
                                           {k: got.get(k) for k in list(want) + ["ttft_ms", "err"]}))
    px.stop()
    up.shutdown()
    print("SELFCHECK %s (%d failures)" % ("PASS" if not bad else "FAIL", bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--model", required=True, choices=sorted(MODELS))
    r.add_argument("--model-path")
    r.add_argument("--engines", default="axon,llama")
    r.add_argument("--harness", default="opencode,codex,pi")
    r.add_argument("--tasks", default="all")
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--turn-timeout", type=int, default=420)
    r.add_argument("--sampler", choices=("card", "defaults"), default="card",
                   help="card: the model card's sampler on both servers (default); defaults: each engine's own")
    r.add_argument("--axon-bin")
    r.add_argument("--llama-bin")
    r.add_argument("--out", required=True)
    r.add_argument("--no-sandbox", action="store_true")
    t = sp.add_parser("table")
    t.add_argument("paths", nargs="+")
    s = sp.add_parser("selfcheck")
    s.add_argument("--no-sandbox", action="store_true")
    sp.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "table":
        return cmd_table(a)
    if a.cmd == "selfcheck":
        return cmd_selfcheck(a)
    for t in TASKS:
        print("%-14s %-11s %-34s %s" % (t["id"], t["base"], t["kind"], t["prompts"][0][:90]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
