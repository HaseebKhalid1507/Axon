#!/usr/bin/env python3
"""make_repo.py DIR: a small synthetic Python project ("ledger") for the agent-harness captures
(specs/agent-harness-traces.md section 4). ~9 modules + a unittest suite, deterministic, no dependencies.
One planted bug (pricing.apply_discount) makes test_pricing fail; everything else passes. The CLI, reports and
formats are wired across files so a feature (a new output format) touches 3-4 files. Synthetic: no real data."""
import os, sys, textwrap

FILES = {
"ledger/__init__.py": '''
"""ledger: a tiny order ledger (synthetic project for agent-harness captures)."""
__version__ = "0.3.1"
''',
"ledger/models.py": '''
"""Domain objects."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Item:
    sku: str
    name: str
    unit_cents: int

    def __post_init__(self):
        if self.unit_cents < 0:
            raise ValueError(f"negative price for {self.sku}")


@dataclass
class Line:
    item: Item
    qty: int

    @property
    def cents(self):
        return self.item.unit_cents * self.qty


@dataclass
class Order:
    order_id: str
    customer: str
    lines: list = field(default_factory=list)
    discount_pct: int = 0

    def add(self, item, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        for ln in self.lines:
            if ln.item.sku == item.sku:
                ln.qty += qty
                return ln
        ln = Line(item, qty)
        self.lines.append(ln)
        return ln

    @property
    def subtotal_cents(self):
        return sum(ln.cents for ln in self.lines)
''',
"ledger/pricing.py": '''
"""Prices, discounts and tax. All amounts are integer cents."""
TAX_RATES = {"standard": 20, "reduced": 5, "zero": 0}


def apply_discount(cents, pct):
    """Return cents after a pct% discount, rounded half up to the cent."""
    if not 0 <= pct <= 100:
        raise ValueError("discount out of range")
    # BUG (planted): integer division truncates instead of rounding half up.
    return cents - (cents * pct) // 100


def tax(cents, rate="standard"):
    """Tax on cents at a named rate, rounded half up."""
    pct = TAX_RATES[rate]
    return (cents * pct + 50) // 100


def total(order, rate="standard"):
    """Order total: subtotal, minus the order's discount, plus tax."""
    net = apply_discount(order.subtotal_cents, order.discount_pct)
    return net + tax(net, rate)
''',
"ledger/storage.py": '''
"""A JSON-lines store of orders."""
import json, os
from .models import Item, Order


class Store:
    def __init__(self, path):
        self.path = path

    def save(self, order):
        rec = {"order_id": order.order_id, "customer": order.customer, "discount_pct": order.discount_pct,
               "lines": [{"sku": l.item.sku, "name": l.item.name, "unit_cents": l.item.unit_cents, "qty": l.qty}
                         for l in order.lines]}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\\n")

    def load_all(self):
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                o = Order(r["order_id"], r["customer"], discount_pct=r.get("discount_pct", 0))
                for l in r["lines"]:
                    o.add(Item(l["sku"], l["name"], l["unit_cents"]), l["qty"])
                out.append(o)
        return out

    def find(self, order_id):
        for o in self.load_all():
            if o.order_id == order_id:
                return o
        raise KeyError(order_id)
''',
"ledger/validate.py": '''
"""Input validation for the CLI."""
import re

SKU = re.compile(r"^[A-Z]{3}-\\d{3}$")


def check_sku(s):
    if not SKU.match(s):
        raise ValueError(f"bad sku {s!r}: want AAA-000")
    return s


def check_qty(s):
    q = int(s)
    if q <= 0 or q > 1000:
        raise ValueError("qty must be 1..1000")
    return q


def check_discount(s):
    d = int(s)
    if not 0 <= d <= 100:
        raise ValueError("discount must be 0..100")
    return d
''',
"ledger/formats.py": '''
"""Output formats for reports. Each takes a list of row dicts with the same keys."""


def as_text(rows):
    if not rows:
        return "(no rows)"
    keys = list(rows[0])
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in keys}
    head = "  ".join(k.ljust(widths[k]) for k in keys)
    body = ["  ".join(str(r[k]).ljust(widths[k]) for k in keys) for r in rows]
    return "\\n".join([head, "-" * len(head)] + body)


FORMATS = {"text": as_text}


def render(rows, fmt="text"):
    try:
        return FORMATS[fmt](rows)
    except KeyError:
        raise ValueError(f"unknown format {fmt!r}; have {sorted(FORMATS)}") from None
''',
"ledger/reports.py": '''
"""Reports over stored orders."""
from . import pricing
from .formats import render


def order_rows(orders, rate="standard"):
    rows = []
    for o in orders:
        rows.append({"order": o.order_id, "customer": o.customer, "lines": len(o.lines),
                     "subtotal": o.subtotal_cents, "total": pricing.total(o, rate)})
    return rows


def customer_rows(orders, rate="standard"):
    agg = {}
    for o in orders:
        a = agg.setdefault(o.customer, {"customer": o.customer, "orders": 0, "total": 0})
        a["orders"] += 1
        a["total"] += pricing.total(o, rate)
    return sorted(agg.values(), key=lambda r: -r["total"])


def report(orders, kind="orders", fmt="text", rate="standard"):
    rows = order_rows(orders, rate) if kind == "orders" else customer_rows(orders, rate)
    return render(rows, fmt)
''',
"ledger/cli.py": '''
"""Command line: ledger add / report."""
import argparse, sys
from .models import Item, Order
from .storage import Store
from .reports import report
from . import validate


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ledger")
    ap.add_argument("--db", default="orders.jsonl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("order_id")
    a.add_argument("customer")
    a.add_argument("--item", nargs=4, action="append", metavar=("SKU", "NAME", "CENTS", "QTY"), required=True)
    a.add_argument("--discount", default="0")
    r = sub.add_parser("report")
    r.add_argument("--kind", choices=["orders", "customers"], default="orders")
    r.add_argument("--rate", default="standard")
    args = ap.parse_args(argv)
    st = Store(args.db)
    if args.cmd == "add":
        o = Order(args.order_id, args.customer, discount_pct=validate.check_discount(args.discount))
        for sku, name, cents, qty in args.item:
            o.add(Item(validate.check_sku(sku), name, int(cents)), validate.check_qty(qty))
        st.save(o)
        print(f"saved {o.order_id}")
    else:
        print(report(st.load_all(), kind=args.kind, rate=args.rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
"tests/__init__.py": "",
"tests/test_models.py": '''
import unittest
from ledger.models import Item, Order


class TestModels(unittest.TestCase):
    def test_add_merges_same_sku(self):
        o = Order("o1", "ann")
        it = Item("ABC-001", "pen", 150)
        o.add(it, 2)
        o.add(it, 3)
        self.assertEqual(len(o.lines), 1)
        self.assertEqual(o.subtotal_cents, 750)

    def test_negative_price(self):
        with self.assertRaises(ValueError):
            Item("ABC-002", "bad", -1)
''',
"tests/test_pricing.py": '''
import unittest
from ledger import pricing
from ledger.models import Item, Order


class TestPricing(unittest.TestCase):
    def test_discount_rounds_half_up(self):
        self.assertEqual(pricing.apply_discount(999, 15), 849)   # 999 * 0.85 = 849.15
        self.assertEqual(pricing.apply_discount(1001, 50), 500)  # 500.5 -> 501 off, rounded half up -> 500

    def test_total(self):
        o = Order("o2", "bob", discount_pct=10)
        o.add(Item("XYZ-100", "mug", 1234), 2)
        self.assertEqual(pricing.total(o), 2665)
''',
"tests/test_storage.py": '''
import os, tempfile, unittest
from ledger.models import Item, Order
from ledger.storage import Store


class TestStorage(unittest.TestCase):
    def test_roundtrip(self):
        d = tempfile.mkdtemp()
        st = Store(os.path.join(d, "o.jsonl"))
        o = Order("o3", "cy", discount_pct=5)
        o.add(Item("QQQ-123", "cup", 300), 4)
        st.save(o)
        back = st.find("o3")
        self.assertEqual(back.subtotal_cents, 1200)
        self.assertEqual(back.discount_pct, 5)
''',
"tests/test_reports.py": '''
import unittest
from ledger.models import Item, Order
from ledger.reports import report, customer_rows


class TestReports(unittest.TestCase):
    def setUp(self):
        a = Order("a1", "ann")
        a.add(Item("AAA-001", "x", 100), 1)
        b = Order("b1", "bob")
        b.add(Item("BBB-001", "y", 500), 2)
        self.orders = [a, b]

    def test_text_report_has_header(self):
        out = report(self.orders)
        self.assertIn("order", out.splitlines()[0])

    def test_customers_sorted_by_total(self):
        rows = customer_rows(self.orders)
        self.assertEqual(rows[0]["customer"], "bob")
''',
"README.md": '''
# ledger

A tiny order ledger. `python3 -m ledger.cli add ...` stores orders in a JSON-lines file;
`python3 -m ledger.cli report --kind orders|customers` prints a text report.

Run the tests with `python3 -m unittest discover -s tests -t .`.
''',
}


def main():
    d = sys.argv[1]
    for rel, src in FILES.items():
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(textwrap.dedent(src).lstrip("\n"))


if __name__ == "__main__":
    main()
