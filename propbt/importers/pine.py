"""Pine Script v5/v6 → strategy schema importer (deterministic, strict subset).

Philosophy: idiom-level recognition of the house dialect that pine_author.md
produces, not general Pine interpretation. Every corpus strategy follows the
same skeleton — global inputs and indicators, a `time()` session gate, boolean
condition assigns, `var` frozen risk levels set with `:=` inside entry
if-blocks, position_size-guarded `strategy.exit(stop=, limit=)` brackets, and
`strategy.close_all` at session end. The importer parses exactly those shapes
into the schema; anything outside the subset is rejected with a specific
error. "Unsupported construct" is an agent-rewrite turn, not a dead end.

Supported subset (see backend/tests/test_importer_pine.py for one test per
idiom):
  - `//@version=5|6`; single-line or multi-line `strategy()` declaration with
    header kwargs (initial_capital, default_qty_value → fixed_contracts,
    cash-per-contract commission, slippage).
  - Inputs: input.int/float/bool (bool defaults are constant-folded into
    conditions; all inputs are carried on ImportResult.pine_inputs — the
    future sweep grid).
  - Indicators: ta.ema/sma/atr/rsi/vwap/highest/lowest, `[st, dir] =
    ta.supertrend(f, len)` tuple destructure, trailing `[n]` history refs
    (params.offset), and the narrow same-symbol HTF form
    `request.security(syminfo.tickerid, "TF", ta.ema(close, len))`.
  - Session gate: `x = not na(time(timeframe.period, "HHMM-HHMM[:days]", tz))`
    plus derived sessionStart/sessionEnd flags, `var int` session bar counters
    (warm-up / opening-range windows).
  - Conditions: and/or/not trees over comparisons, crossovers, session flags,
    bool vars; normalizations for barstate.isconfirmed, position_size gates,
    na() warm-up guards, once-per-session traded flags, and the
    `bull_now and bear_now[1]` flip idiom (→ cross of the compared series).
  - Opening-range construction blocks → ORB indicators (frozen high/low).
  - Exits: frozen `var` levels or block locals — ATR stops (close- or
    avg_price-anchored), indicator-level stops (ORB boundary), R-multiple
    targets, range R-multiple targets; strategy.exit loss=/profit= points;
    strategy.close_all on session end → session_filter.flat_at_end.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ────────────────────────── public result types ──────────────────────────


@dataclass
class ImportError_:
    line: int
    message: str


@dataclass
class ImportResult:
    ok: bool
    strategy_json: dict[str, Any] | None
    errors: list[ImportError_] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported_features: list[str] = field(default_factory=list)
    pine_version: int | None = None
    detected_name: str | None = None
    # input.int/float/bool declarations (name/kind/default), in source order.
    # Carried on the result (not in strategy_json — schema metadata is closed):
    # this is the sweep grid the research phase offers the user.
    pine_inputs: list[dict] = field(default_factory=list)


# ────────────────────────── preprocessing ──────────────────────────


def _strip_comment(line: str) -> str:
    """Remove a trailing // comment, respecting string literals."""
    out: list[str] = []
    in_str: str | None = None
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == in_str:
                in_str = None
            out.append(ch)
        elif ch in "\"'":
            in_str = ch
            out.append(ch)
        elif ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _paren_delta(code: str) -> int:
    """Net open-paren/bracket depth of a comment-stripped line (strings ignored)."""
    depth = 0
    in_str: str | None = None
    for ch in code:
        if in_str:
            if ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
    return depth


def _preprocess(source: str) -> list[tuple[int, str]]:
    """Join statements that span lines via unbalanced parens (e.g. the
    multi-line strategy() declaration) into single logical lines. Returns
    (first_lineno, line) pairs; unjoined lines pass through untouched."""
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_lineno = 0
    depth = 0
    for lineno, raw in enumerate(source.splitlines(), start=1):
        code = _strip_comment(raw)
        if not buf:
            d = _paren_delta(code)
            if d > 0:
                buf = [code.rstrip()]
                buf_lineno = lineno
                depth = d
            else:
                out.append((lineno, raw))
        else:
            depth += _paren_delta(code)
            buf.append(code.strip())
            if depth <= 0:
                out.append((buf_lineno, " ".join(p for p in buf if p)))
                buf = []
    if buf:  # unterminated continuation — emit as-is; downstream reports errors
        out.append((buf_lineno, " ".join(p for p in buf if p)))
    return out


# ────────────────────────── statement patterns ──────────────────────────

VERSION_RE = re.compile(r"^\s*//\s*@version\s*=\s*(\d+)\s*$")
STRATEGY_DECL_RE = re.compile(r'^\s*strategy\s*\(\s*(?:title\s*=\s*)?"([^"]+)"')
INDICATOR_DECL_RE = re.compile(r'^\s*indicator\s*\(\s*(?:title\s*=\s*)?"([^"]+)"')

INPUT_RE = re.compile(
    r"^(\w+)\s*=\s*input\.(int|float|bool)\s*\(\s*(-?[0-9]+(?:\.[0-9]+)?|true|false)\s*[,)]"
)

# Indicator assigns. Length may be a literal or an input var.
EMA_RE = re.compile(r"^(\w+)\s*=\s*ta\.ema\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(\[\d+\])?$")
SMA_RE = re.compile(r"^(\w+)\s*=\s*ta\.sma\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(\[\d+\])?$")
ATR_RE = re.compile(r"^(\w+)\s*=\s*ta\.atr\s*\(\s*(\w+)\s*\)\s*(\[\d+\])?$")
RSI_RE = re.compile(r"^(\w+)\s*=\s*ta\.rsi\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(\[\d+\])?$")
VWAP_RE = re.compile(r"^(\w+)\s*=\s*ta\.vwap\b\s*$")
HIGHEST_RE = re.compile(r"^(\w+)\s*=\s*ta\.highest\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(\[\d+\])?$")
LOWEST_RE = re.compile(r"^(\w+)\s*=\s*ta\.lowest\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(\[\d+\])?$")
SUPERTREND_RE = re.compile(
    r"^\[\s*(\w+)\s*,\s*(\w+)\s*\]\s*=\s*ta\.supertrend\s*\(\s*([\w.]+)\s*,\s*(\w+)\s*\)$"
)
# Narrow same-symbol HTF form only; general request.security stays rejected.
SECURITY_EMA_RE = re.compile(
    r'^(\w+)\s*=\s*request\.security\s*\(\s*syminfo\.tickerid\s*,\s*"(\w+)"\s*,\s*'
    r"ta\.ema\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*(?:,\s*lookahead\s*=\s*barmerge\.lookahead_off\s*)?\)$"
)

# A time() window call, substituted with a placeholder before tokenizing so the
# expression parser sees a single atom. Both corpus forms use it:
#   not na(time(timeframe.period, "0930-1600[:days]", tz))
#   time(timeframe.period, "0930-1600", tz) != 0
TIME_CALL_RE = re.compile(
    r'time\s*\(\s*timeframe\.period\s*,\s*(?:"(\d{4})-(\d{4})(?::([1-7]+))?"|(\w+))\s*,\s*"([^"]+)"\s*\)'
)

INPUT_SESSION_RE = re.compile(
    r'^(\w+)\s*=\s*input\.session\s*\(\s*"(\d{4})-(\d{4})(?::([1-7]+))?"'
)


def _window_dict(start: str, end: str, days: str | None, tz: str) -> dict:
    window: dict[str, Any] = {
        "start": f"{start[:2]}:{start[2:]}",
        "end": f"{end[:2]}:{end[2:]}",
        "tz": tz,
    }
    if days:
        # TV session-day digits: 1=Sunday … 7=Saturday; schema is ISO 1=Mon…7=Sun.
        window["days"] = sorted({7 if d == "1" else int(d) - 1 for d in days})
    return window

VAR_DECL_RE = re.compile(
    r"^var\s+(int|float|bool)\s+(\w+)\s*=\s*(na|true|false|-?[0-9.]+)$"
)

ASSIGN_RE = re.compile(r"^(\w+)\s*(:?=|\+=)\s*(.+)$")
IF_RE = re.compile(r"^(if|else if|else)\b\s*(.*)$")

# Extra kwargs (qty=, comment=, …) are tolerated; when= extracted separately.
ENTRY_RE = re.compile(
    r'^strategy\.entry\s*\(\s*"([^"]+)"\s*,\s*strategy\.(long|short)\b[^)]*\)$'
)
ENTRY_WHEN_RE = re.compile(r"\bwhen\s*=\s*(\w+)")
EXIT_RE = re.compile(r"^strategy\.exit\s*\((.*)\)$")
CLOSE_ALL_RE = re.compile(r"^strategy\.close_all\s*\(")

IGNORED_RE = re.compile(
    r"^(plot|plotshape|plotchar|plotarrow|bgcolor|barcolor|hline|fill|alertcondition)\s*\("
)

VALID_SOURCES = {"open", "high", "low", "close", "volume", "hl2", "hlc3", "ohlc4"}

# Constructs we explicitly do not handle. var/input.bool/rsi/vwap moved OUT of
# this list as the dialect grew; request.security's general form stays.
UNSUPPORTED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bta\.macd\s*\("), "ta.macd() — MACD assigns not yet in the importer"),
    (re.compile(r"\bta\.bb\s*\("), "ta.bb() — Bollinger assigns not yet in the importer"),
    (re.compile(r"\brequest\.security_lower_tf\b"), "request.security_lower_tf() — out of scope"),
    (re.compile(r"\barray\.\w+\b"), "Pine arrays — out of scope"),
    (re.compile(r"\bmatrix\.\w+\b"), "Pine matrices — out of scope"),
    (re.compile(r"\bline\.new\b|\blabel\.new\b|\bbox\.new\b"), "dynamic line/label/box drawing — out of scope"),
    (re.compile(r"\binput\.(string|timeframe|source)\("), "input.string/timeframe/source — not yet supported"),
    (re.compile(r"=>"), "user-defined functions — out of scope"),
    (re.compile(r"^\s*(for|while|switch)\b"), "loops/switch — out of scope"),
]


# ────────────────────────── expression AST ──────────────────────────
#
# Nested tuples keep this dependency-free and easy to pattern-match:
#   ("num", float)                     literal or resolved input var
#   ("operand", name)                  indicator var / price source (pre-snake)
#   ("arith", op, a, b)                op in + - * /
#   ("cmp", op, a, b)                  op in gt lt ge le eq ne
#   ("and"|"or", [exprs]) ("not", x)  boolean structure
#   ("true",) ("false",)
#   ("cross", ctype, a, b)            ctype in cross_above/cross_below
#   ("shift", expr, n)                 trailing [n] on a bool var
#   ("na", expr)
#   ("session",) ("session_start",) ("session_end",)
#   ("counter_cmp", op, name, n)      session-bar-counter comparison
#   ("flat",) ("pos", "long"|"short"|"nonzero") ("pos_prev", op_sign)
#   ("traded", var)                    once-per-session flag
#   ("anchor",)                        close / strategy.position_avg_price in risk exprs


_TOKEN_RE = re.compile(
    r"\s*(>=|<=|==|!=|\(|\)|\[|\]|,|\+|-|\*|/|>|<|\?|:|"
    r'"[^"]*"|\d+\.?\d*|[A-Za-z_][A-Za-z0-9_.]*)'
)


def _tokenize(text: str) -> list[str] | None:
    tokens: list[str] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            if text[pos:].strip():
                return None  # untokenizable garbage
            break
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


class _ExprError(Exception):
    pass


def _fold_arith(op: str, a: tuple, b: tuple) -> tuple:
    """Constant-fold literal arithmetic (e.g. `100 - rsiOS` with rsiOS input)."""
    if a[0] == "num" and b[0] == "num":
        x, y = a[1], b[1]
        if op == "+":
            return ("num", x + y)
        if op == "-":
            return ("num", x - y)
        if op == "*":
            return ("num", x * y)
        if op == "/" and y != 0:
            return ("num", x / y)
    return ("arith", op, a, b)


class _ExprParser:
    """Recursive-descent parser for house-dialect boolean/arith expressions."""

    def __init__(self, tokens: list[str], symbols: "_Symbols",
                 windows: dict[str, dict] | None = None):
        self.toks = tokens
        self.i = 0
        self.sym = symbols
        self.windows = windows or {}

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> str:
        tok = self.peek()
        if tok is None:
            raise _ExprError("unexpected end of expression")
        self.i += 1
        return tok

    def expect(self, tok: str) -> None:
        got = self.next()
        if got != tok:
            raise _ExprError(f"expected {tok!r}, got {got!r}")

    def parse_bool(self) -> tuple:
        node = self.parse_ternary()
        if self.peek() is not None:
            raise _ExprError(f"trailing tokens at {self.peek()!r}")
        return node

    def parse_ternary(self) -> tuple:
        cond = self.parse_or()
        if self.peek() != "?":
            return cond
        self.next()
        then = self.parse_ternary()
        self.expect(":")
        other = self.parse_ternary()
        # Corpus ternaries have bool-input conditions that resolve at import
        # time (useSession ? <window test> : true) — fold immediately.
        if cond == ("true",):
            return then
        if cond == ("false",):
            return other
        raise _ExprError("ternary with a non-constant condition")

    def parse_or(self) -> tuple:
        parts = [self.parse_and()]
        while self.peek() == "or":
            self.next()
            parts.append(self.parse_and())
        return parts[0] if len(parts) == 1 else ("or", parts)

    def parse_and(self) -> tuple:
        parts = [self.parse_not()]
        while self.peek() == "and":
            self.next()
            parts.append(self.parse_not())
        return parts[0] if len(parts) == 1 else ("and", parts)

    def parse_not(self) -> tuple:
        if self.peek() == "not":
            self.next()
            return ("not", self.parse_not())
        return self.parse_cmp()

    def parse_cmp(self) -> tuple:
        left = self.parse_arith()
        op_map = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}
        if self.peek() in op_map:
            op = op_map[self.next()]
            right = self.parse_arith()
            return self._make_cmp(op, left, right)
        return left

    def parse_arith(self) -> tuple:
        node = self.parse_term()
        while self.peek() in ("+", "-"):
            op = self.next()
            node = _fold_arith(op, node, self.parse_term())
        return node

    def parse_term(self) -> tuple:
        node = self.parse_atom()
        while self.peek() in ("*", "/"):
            op = self.next()
            node = _fold_arith(op, node, self.parse_atom())
        return node

    def parse_atom(self) -> tuple:
        tok = self.next()
        if tok == "(":
            node = self.parse_ternary()
            self.expect(")")
            return self._maybe_shift(node)
        if re.fullmatch(r"\d+\.?\d*", tok):
            return ("num", float(tok))
        if tok == "-":
            inner = self.parse_atom()
            if inner[0] == "num":
                return ("num", -inner[1])
            return ("arith", "-", ("num", 0.0), inner)
        if tok in ("na", "nz"):
            self.expect("(")
            inner = self.parse_or()
            self.expect(")")
            return inner if tok == "nz" else ("na", inner)
        if tok in ("math.max", "math.min"):
            self.expect("(")
            a = self.parse_arith()
            self.expect(",")
            b = self.parse_arith()
            self.expect(")")
            return ("mathfn", tok.split(".")[1], a, b)
        if tok in ("ta.crossover", "ta.crossunder"):
            self.expect("(")
            a = self.parse_arith()
            self.expect(",")
            b = self.parse_arith()
            self.expect(")")
            ctype = "cross_above" if tok == "ta.crossover" else "cross_below"
            return ("cross", ctype, a, b)
        if tok == "barstate.isconfirmed":
            return ("true",)
        if tok == "true":
            return ("true",)
        if tok == "false":
            return ("false",)
        if tok == "strategy.position_size":
            return self._maybe_shift(("possize",))
        if tok in self.windows:
            return self._maybe_shift(("timewindow", self.windows[tok]))
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", tok):
            return self._maybe_shift(self.sym.resolve(tok))
        raise _ExprError(f"unexpected token {tok!r}")

    def _maybe_shift(self, node: tuple) -> tuple:
        while self.peek() == "[":
            self.next()
            n_tok = self.next()
            if not re.fullmatch(r"\d+", n_tok):
                raise _ExprError(f"history offset must be a literal int, got {n_tok!r}")
            self.expect("]")
            node = ("shift", node, int(n_tok))
        return node

    def _make_cmp(self, op: str, left: tuple, right: tuple) -> tuple:
        # close[1] <= level  →  (close <= level)[1]: hoist the operand shift to
        # the comparison so the flip idiom (a > x and a[1] <= x) reduces to a
        # cross. Exact when the un-shifted side is frozen within a session
        # (opening-range levels); a one-bar approximation otherwise.
        if left[0] == "shift" and right[0] != "shift":
            return ("shift", self._make_cmp(op, left[1], right), left[2])
        if right[0] == "shift" and left[0] != "shift":
            return ("shift", self._make_cmp(op, left, right[1]), right[2])
        # position_size comparisons → semantic markers
        for a, b, flip in ((left, right, False), (right, left, True)):
            if a[0] == "possize" and b[0] == "num" and b[1] == 0.0:
                o = {"gt": "lt", "lt": "gt", "ge": "le", "le": "ge"}.get(op, op) if flip else op
                if o == "eq":
                    return ("flat",)
                if o == "ne":
                    return ("pos", "nonzero")
                if o == "gt":
                    return ("pos", "long")
                if o == "lt":
                    return ("pos", "short")
                if o == "le":
                    return ("not", ("pos", "long"))
                if o == "ge":
                    return ("not", ("pos", "short"))
            if a[0] == "shift" and a[1][0] == "possize" and b[0] == "num" and b[1] == 0.0:
                return ("pos_prev", op if not flip else op)
            # time(...) != 0 / == 0 — session membership test
            if a[0] == "timewindow" and b[0] == "num" and b[1] == 0.0:
                if op == "ne":
                    return ("session_w", a[1])
                if op == "eq":
                    return ("not", ("session_w", a[1]))
        # session-bar-counter comparisons → warm-up / entry-window markers
        if left[0] == "counter" and right[0] == "num":
            return ("counter_cmp", op, left[1], right[1])
        return ("cmp", op, left, right)


@dataclass
class _Symbols:
    """Symbol table for name resolution inside expressions."""
    inputs: dict[str, float] = field(default_factory=dict)
    bool_inputs: dict[str, bool] = field(default_factory=dict)
    indicators: dict[str, dict] = field(default_factory=dict)   # var → {type, params}
    bools: dict[str, tuple] = field(default_factory=dict)       # var → expr AST
    session_vars: dict[str, dict] = field(default_factory=dict)  # var → window
    session_start_vars: set[str] = field(default_factory=set)
    session_end_vars: set[str] = field(default_factory=set)
    counter_vars: set[str] = field(default_factory=set)
    state_vars: set[str] = field(default_factory=set)           # var float/bool = na/false
    traded_vars: set[str] = field(default_factory=set)          # state bools set true on entry
    avg_entry_aliases: set[str] = field(default_factory=set)

    def resolve(self, name: str) -> tuple:
        if name in self.indicators or name in VALID_SOURCES:
            return ("operand", name)
        if name in self.inputs:
            return ("num", float(self.inputs[name]))
        if name in self.bool_inputs:
            return ("true",) if self.bool_inputs[name] else ("false",)
        if name in self.session_vars:
            return ("session_w", self.session_vars[name])
        if name in self.session_start_vars:
            return ("session_start",)
        if name in self.session_end_vars:
            return ("session_end",)
        if name in self.counter_vars:
            return ("counter", name)
        if name in self.traded_vars:
            return ("traded", name)
        if name in self.bools:
            return self.bools[name]
        if name in self.avg_entry_aliases or name == "strategy.position_avg_price":
            return ("anchor",)
        if name in self.state_vars:
            return ("statevar", name)
        raise _ExprError(f"unknown name {name!r}")


# ────────────────────────── block tree ──────────────────────────


@dataclass
class _Stmt:
    lineno: int
    text: str            # stripped
    indent: int


@dataclass
class _Block:
    """An `if` chain: (condition text, body) branches; else branch last with
    condition None."""
    lineno: int
    branches: list[tuple[str | None, list["._Node"]]] = field(default_factory=list)


_Node = "._Stmt | _Block"


def _build_tree(stmts: list[_Stmt]) -> list[Any]:
    """Group an indented statement stream into nested if-chains."""
    result, _ = _parse_body(stmts, 0, parent_indent=-1)
    return result


def _parse_body(stmts: list[_Stmt], i: int, parent_indent: int) -> tuple[list[Any], int]:
    """Consume statements strictly deeper than parent_indent."""
    out: list[Any] = []
    body_indent: int | None = None
    while i < len(stmts):
        s = stmts[i]
        if s.indent <= parent_indent:
            break
        if body_indent is None:
            body_indent = s.indent
        if s.indent < body_indent:
            break  # dedent below this body but above parent — sibling of an ancestor
        m = IF_RE.match(s.text)
        if m and m.group(1) == "if":
            block = _Block(lineno=s.lineno)
            cond = m.group(2).strip()
            body, i = _parse_body(stmts, i + 1, s.indent)
            block.branches.append((cond, body))
            # else-if / else chain at the same indent
            while i < len(stmts) and stmts[i].indent == s.indent:
                m2 = IF_RE.match(stmts[i].text)
                if not m2 or m2.group(1) == "if":
                    break
                cond2 = m2.group(2).strip() or None
                body2, i = _parse_body(stmts, i + 1, s.indent)
                block.branches.append((cond2 if m2.group(1) == "else if" else None, body2))
            out.append(block)
        else:
            out.append(s)
            i += 1
    return out, i


# ────────────────────────── parser state ──────────────────────────


@dataclass
class _RiskSpec:
    kind: str            # atr_dist | atr_stop | rr_target | ind_level | range | range_rr_target | atr_copy
    data: dict = field(default_factory=dict)


@dataclass
class _State:
    name: str = ""
    pine_version: int | None = None
    initial_capital: float | None = None
    default_qty: float | None = None
    commission_per_contract: float | None = None
    slippage_points: float | None = None

    sym: _Symbols = field(default_factory=_Symbols)
    inputs_meta: list[dict] = field(default_factory=list)

    session_window: dict | None = None      # {"start","end","tz","days"?}
    eod_window: dict | None = None          # short tail window used only to flatten
    session_inputs: dict[str, tuple] = field(default_factory=dict)  # input.session vars
    warmup_bars: int | None = None
    max_entry_bars: int | None = None
    flat_at_end: bool = False
    session_used: bool = False
    max_entries_per_session: int | None = None

    long_cond: tuple | None = None
    short_cond: tuple | None = None

    # frozen risk specs per side
    stop_spec: dict[str, _RiskSpec | None] = field(default_factory=lambda: {"long": None, "short": None})
    target_spec: dict[str, _RiskSpec | None] = field(default_factory=lambda: {"long": None, "short": None})
    stop_points: dict[str, float | None] = field(default_factory=lambda: {"long": None, "short": None})
    target_points: dict[str, float | None] = field(default_factory=lambda: {"long": None, "short": None})

    # risk-expr values carried by vars (block locals AND state vars)
    var_specs: dict[str, _RiskSpec] = field(default_factory=dict)
    # ORB construction: state var → ("high"|"low", bars)
    orb_vars: dict[str, tuple[str, int | None]] = field(default_factory=dict)
    orb_bars: int | None = None
    # bool state vars set true once a session counter passes N ("range complete")
    completion_flags: dict[str, int] = field(default_factory=dict)

    errors: list[ImportError_] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


# ────────────────────────── header + declaration handlers ──────────────────────────


def _kw_str(args: str, key: str) -> str | None:
    m = re.search(rf'{key}\s*=\s*"([^"]+)"', args)
    return m.group(1) if m else None


def _kw_num(args: str, key: str) -> float | None:
    m = re.search(rf"\b{key}\s*=\s*(-?[0-9]+(?:\.[0-9]+)?)", args)
    return float(m.group(1)) if m else None


def _try_strategy_decl(line: str, lineno: int, state: _State) -> bool:
    m = STRATEGY_DECL_RE.match(line)
    if not m:
        return False
    if state.name:
        state.errors.append(ImportError_(lineno, "multiple strategy() declarations"))
    state.name = m.group(1)
    state.initial_capital = _kw_num(line, "initial_capital")
    state.default_qty = _kw_num(line, "default_qty_value")
    if re.search(r"commission_type\s*=\s*strategy\.commission\.cash_per_contract", line):
        state.commission_per_contract = _kw_num(line, "commission_value")
    state.slippage_points = _kw_num(line, "slippage")
    return True


def _resolve_length(tok: str, lineno: int, what: str, state: _State) -> int | None:
    try:
        return int(float(tok))
    except ValueError:
        pass
    if tok in state.sym.inputs:
        return int(state.sym.inputs[tok])
    state.errors.append(ImportError_(
        lineno, f"{what} length {tok!r} is neither a literal int nor a known input.int/float"
    ))
    return None


def _offset_of(suffix: str | None) -> int:
    return int(suffix.strip("[]")) if suffix else 0


def _try_indicator_assign(line: str, lineno: int, state: _State) -> bool:
    sym = state.sym

    def add(var: str, itype: str, params: dict, offset: int = 0) -> None:
        if offset:
            params = dict(params, offset=offset)
        sym.indicators[var] = {"type": itype, "params": params}

    for pat, itype, default_src in ((EMA_RE, "EMA", None), (SMA_RE, "SMA", None), (RSI_RE, "RSI", None)):
        m = pat.match(line)
        if m:
            var, src, length_tok, sfx = m.group(1), m.group(2), m.group(3), m.group(4)
            if src not in VALID_SOURCES:
                state.errors.append(ImportError_(lineno, f"{itype} source {src!r} not recognized"))
                return True
            length = _resolve_length(length_tok, lineno, itype, state)
            if length is None:
                return True
            add(var, itype, {"source": src, "length": length}, _offset_of(sfx))
            return True

    m = ATR_RE.match(line)
    if m:
        var, length_tok, sfx = m.group(1), m.group(2), m.group(3)
        length = _resolve_length(length_tok, lineno, "ATR", state)
        if length is None:
            return True
        add(var, "ATR", {"length": length}, _offset_of(sfx))
        return True

    m = VWAP_RE.match(line)
    if m:
        add(m.group(1), "VWAP", {})
        return True

    for pat, itype, dsrc in ((HIGHEST_RE, "HIGHEST", "high"), (LOWEST_RE, "LOWEST", "low")):
        m = pat.match(line)
        if m:
            var, src, length_tok, sfx = m.group(1), m.group(2), m.group(3), m.group(4)
            if src not in VALID_SOURCES:
                state.errors.append(ImportError_(lineno, f"{itype} source {src!r} not recognized"))
                return True
            length = _resolve_length(length_tok, lineno, itype, state)
            if length is None:
                return True
            add(var, itype, {"source": src, "length": length}, _offset_of(sfx))
            return True

    m = SUPERTREND_RE.match(line)
    if m:
        line_var, dir_var, factor_tok, len_tok = m.groups()
        factor = _num_or_input(factor_tok, state)
        length = _resolve_length(len_tok, lineno, "Supertrend ATR", state)
        if factor is None or length is None:
            state.errors.append(ImportError_(lineno, "ta.supertrend args must be literals or inputs"))
            return True
        add(line_var, "SUPERTREND", {"factor": factor, "atr_length": length, "output": "line"})
        add(dir_var, "SUPERTREND", {"factor": factor, "atr_length": length, "output": "direction"})
        return True

    m = SECURITY_EMA_RE.match(line)
    if m:
        var, tf, src, len_tok = m.groups()
        if src not in VALID_SOURCES:
            state.errors.append(ImportError_(lineno, f"HTF EMA source {src!r} not recognized"))
            return True
        length = _resolve_length(len_tok, lineno, "HTF EMA", state)
        if length is None:
            return True
        tf_norm = tf if not tf.isdigit() else (
            f"{int(tf)}m" if int(tf) < 60 else f"{int(tf) // 60}h"
        )
        add(var, "EMA", {"source": src, "length": length, "timeframe": tf_norm})
        return True

    return False


def _num_or_input(tok: str, state: _State) -> float | None:
    try:
        return float(tok)
    except ValueError:
        return float(state.sym.inputs[tok]) if tok in state.sym.inputs else None


def _classify_session_assign(var: str, expr: tuple, state: _State) -> bool:
    """A bool assign whose value folds to a pure session-membership test makes
    the var a session flag (usable by derived start/end flags and blocks).
    Handles bool-input guards: `not use_sess or <in window>` with default
    true folds to the window test; with default false it folds to constant
    true (filter off), which stays a plain bool."""
    folded = _fold_session(expr)
    if folded is not None and folded[0] == "session_w":
        state.sym.session_vars[var] = folded[1]
        return True
    return False


def _fold_session(expr: tuple) -> tuple | None:
    """Constant-fold true/false/session_w through not/and/or; the two
    session-test shapes (`not na(time)` and `time != 0`) reduce here too.
    Returns the folded node, or None when the expression involves anything
    beyond constants and session tests."""
    t = expr[0]
    if t in ("true", "false", "session_w"):
        return expr
    if t == "na" and expr[1][0] == "timewindow":
        return ("not_session_w", expr[1][1])
    if t == "timewindow":
        return None
    if t == "not":
        sub = _fold_session(expr[1])
        if sub is None:
            return None
        if sub == ("true",):
            return ("false",)
        if sub == ("false",):
            return ("true",)
        if sub[0] == "session_w":
            return ("not_session_w", sub[1])
        if sub[0] == "not_session_w":
            return ("session_w", sub[1])
        return None
    if t in ("and", "or"):
        subs = [_fold_session(c) for c in expr[1]]
        if any(s is None for s in subs):
            return None
        if t == "and":
            if any(s == ("false",) for s in subs):
                return ("false",)
            subs = [s for s in subs if s != ("true",)]
            if not subs:
                return ("true",)
            return subs[0] if len(subs) == 1 else None
        if any(s == ("true",) for s in subs):
            return ("true",)
        subs = [s for s in subs if s != ("false",)]
        if not subs:
            return ("false",)
        return subs[0] if len(subs) == 1 else None
    return None


def _try_derived_session_flag(line: str, state: _State) -> bool:
    """sessionStart = not X[1] and X  /  sessionEnd = not X and X[1]."""
    m = re.match(r"^(\w+)\s*=\s*not\s+(\w+)\s*\[1\]\s*and\s+(\w+)$", line)
    if m and m.group(2) == m.group(3) and m.group(2) in state.sym.session_vars:
        state.sym.session_start_vars.add(m.group(1))
        return True
    m = re.match(r"^(\w+)\s*=\s*not\s+(\w+)\s*and\s+(\w+)\s*\[1\]$", line)
    if m and m.group(2) == m.group(3) and m.group(2) in state.sym.session_vars:
        state.sym.session_end_vars.add(m.group(1))
        return True
    m = re.match(r"^(\w+)\s*=\s*(\w+)\s*\[1\]\s*and\s+not\s+(\w+)$", line)
    if m and m.group(2) == m.group(3) and m.group(2) in state.sym.session_vars:
        state.sym.session_end_vars.add(m.group(1))
        return True
    m = re.match(r"^(\w+)\s*=\s*(\w+)\s*and\s+not\s+(\w+)\s*\[1\]$", line)
    if m and m.group(2) == m.group(3) and m.group(2) in state.sym.session_vars:
        state.sym.session_start_vars.add(m.group(1))
        return True
    # newSession = inRTH and not prevInRTH — session start via a memory var
    # (prevInRTH is a `var bool` updated `prevInRTH := inRTH` each bar)
    m = re.match(r"^(\w+)\s*=\s*(\w+)\s+and\s+not\s+(\w+)$", line)
    if m and m.group(2) in state.sym.session_vars and m.group(3) in state.sym.state_vars:
        state.sym.session_start_vars.add(m.group(1))
        return True
    # barsSinceOpen = ta.barssince(flag) — a 0-based session bar counter when
    # the flag marks session start (derived flag or a short opening window)
    m = re.match(r"^(\w+)\s*=\s*ta\.barssince\s*\(\s*(\w+)\s*\)$", line)
    if m and (m.group(2) in state.sym.session_start_vars or m.group(2) in state.sym.session_vars):
        state.sym.counter_vars.add(m.group(1))
        return True
    return False


# ────────────────────────── risk-expression evaluation ──────────────────────────


def _risk_eval(expr: tuple, state: _State) -> _RiskSpec | None:
    """Evaluate an assignment RHS in 'frozen risk level' context."""
    sym = state.sym
    t = expr[0]
    if t == "operand":
        name = expr[1]
        if name == "close":
            # In frozen-risk context `close` is the entry anchor (fill = signal-bar close)
            return _RiskSpec("anchor", {})
        ind = sym.indicators.get(name)
        if ind and ind["type"] == "ATR" and not ind["params"].get("offset"):
            return _RiskSpec("atr_copy", {"atr_length": ind["params"]["length"]})
        if name in state.orb_vars:
            side, _ = state.orb_vars[name]
            return _RiskSpec("ind_level", {"orb_side": side})
        if ind:
            return _RiskSpec("ind_level", {"indicator_var": name})
        return None
    if t == "statevar":
        return state.var_specs.get(expr[1])
    if t == "num":
        return _RiskSpec("num", {"value": expr[1]})
    if t == "anchor":
        return _RiskSpec("anchor", {})
    if t == "mathfn":
        return None  # handled by the ORB construction pass
    if t == "arith":
        op, a, b = expr[1], expr[2], expr[3]
        sa, sb = _risk_eval(a, state), _risk_eval(b, state)
        if sa is None or sb is None:
            return None
        if op == "*":
            if sb.kind != "num" and sa.kind == "num":
                sa, sb = sb, sa  # multiplication commutes: mult * atr == atr * mult
            if sa.kind == "atr_copy" and sb.kind == "num":
                return _RiskSpec("atr_dist", {"atr_length": sa.data["atr_length"],
                                              "atr_multiplier": sb.data["value"]})
            if sa.kind == "atr_dist" and sb.kind == "num":
                return _RiskSpec("rr_dist", {"ratio": sb.data["value"], "base": sa})
            if sa.kind == "stop_dist" and sb.kind == "num":
                return _RiskSpec("rr_dist", {"ratio": sb.data["value"]})
            if sa.kind == "range" and sb.kind == "num":
                return _RiskSpec("range_dist", {"ratio": sb.data["value"], **sa.data})
            if sa.kind == "num" and sb.kind == "num":
                return _RiskSpec("num", {"value": sa.data["value"] * sb.data["value"]})
        if op in ("+", "-"):
            if op == "-" and sa.kind == "anchor" and sb.kind == "atr_stop":
                # (close - stopPrice) recovers the frozen stop distance
                return _RiskSpec("atr_dist", {"atr_length": sb.data["atr_length"],
                                              "atr_multiplier": sb.data["atr_multiplier"]})
            # (close - stopLevel) / (stopLevel - close): distance to ANY frozen
            # stop level — feeds R-multiple targets anchored on the stop
            stop_like = ("ind_level", "range_level")
            if op == "-" and sa.kind == "anchor" and sb.kind in stop_like:
                return _RiskSpec("stop_dist", {})
            if op == "-" and sa.kind in stop_like and sb.kind == "anchor":
                return _RiskSpec("stop_dist", {})
            if sa.kind == "anchor" and sb.kind == "atr_dist":
                return _RiskSpec("atr_stop", dict(sb.data, sign=op))
            if sa.kind == "anchor" and sb.kind == "rr_dist":
                return _RiskSpec("rr_target", {"ratio": sb.data["ratio"], "sign": op})
            if sa.kind == "anchor" and sb.kind == "range_dist":
                return _RiskSpec("range_rr_target", {"ratio": sb.data["ratio"],
                                                     "upper": sb.data.get("upper"),
                                                     "lower": sb.data.get("lower"),
                                                     "sign": op})
            # orbLow - buf * range  → stop beyond a band boundary
            if sa.kind == "ind_level" and sb.kind == "range_dist":
                return _RiskSpec("range_level", {"level": sa.data,
                                                 "offset_ratio": sb.data["ratio"],
                                                 "upper": sb.data.get("upper"),
                                                 "lower": sb.data.get("lower")})
            if op == "-" and sa.kind == "ind_level" and sb.kind == "ind_level":
                return _RiskSpec("range", {"upper": sa.data, "lower": sb.data})
            if sa.kind == "num" and sb.kind == "num":
                v = sa.data["value"] + sb.data["value"] if op == "+" else sa.data["value"] - sb.data["value"]
                return _RiskSpec("num", {"value": v})
    return None


# ────────────────────────── main entry ──────────────────────────


def import_pine(source: str) -> ImportResult:
    state = _State()
    stmts: list[_Stmt] = []

    for lineno, raw in _preprocess(source):
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            mv = VERSION_RE.match(line)
            if mv:
                state.pine_version = int(mv.group(1))
            continue

        code = _strip_comment(line).rstrip()  # joined lines are already stripped; idempotent
        if not code.strip():
            continue

        unsupported_hit = False
        for pat, msg in UNSUPPORTED_PATTERNS:
            if pat.search(code):
                state.unsupported.append(f"line {lineno}: {msg}")
                unsupported_hit = True
        if unsupported_hit:
            continue

        indent = len(code) - len(code.lstrip())
        stmts.append(_Stmt(lineno=lineno, text=code.strip(), indent=indent))

    tree = _build_tree(stmts)
    for node in tree:
        _handle_top(node, state)

    return _finalize(state)


def _handle_top(node: Any, state: _State) -> None:
    if isinstance(node, _Block):
        _handle_block(node, state, gate=("true",))
        return
    line, lineno = node.text, node.lineno

    if _try_strategy_decl(line, lineno, state):
        return
    m = INDICATOR_DECL_RE.match(line)
    if m:
        state.errors.append(ImportError_(
            lineno,
            "indicator() not supported — paste a strategy() script. Indicators are extracted automatically.",
        ))
        return
    m = INPUT_SESSION_RE.match(line)
    if m:
        var, start, end, days = m.groups()
        state.session_inputs[var] = (start, end, days)
        state.inputs_meta.append({"name": var, "kind": "session",
                                  "default": f"{start}-{end}" + (f":{days}" if days else "")})
        return
    m = INPUT_RE.match(line)
    if m:
        var, kind, raw = m.group(1), m.group(2), m.group(3)
        if kind == "bool":
            val = raw == "true"
            state.sym.bool_inputs[var] = val
            state.inputs_meta.append({"name": var, "kind": "bool", "default": val})
        else:
            value = int(raw) if kind == "int" else float(raw)
            state.sym.inputs[var] = value
            state.inputs_meta.append({"name": var, "kind": kind, "default": value})
        return
    if _try_indicator_assign(line, lineno, state):
        return
    if _try_derived_session_flag(line, state):
        return
    m = VAR_DECL_RE.match(line)
    if m:
        typ, var, init = m.groups()
        if typ == "int":
            state.sym.counter_vars.add(var)
        elif typ == "bool":
            state.sym.state_vars.add(var)  # promoted to traded flag when := true in entry
        else:
            state.sym.state_vars.add(var)
        return
    if IGNORED_RE.match(line):
        return
    m = ENTRY_RE.match(line)
    if m:
        when = ENTRY_WHEN_RE.search(line)
        if when:
            _register_entry(m.group(2), when.group(1), ("true",), lineno, state)
        else:
            state.errors.append(ImportError_(
                lineno, "strategy.entry outside an if-block needs when= (none found)"))
        return
    m = EXIT_RE.match(line)
    if m:
        _register_exit(m.group(1), ("true",), lineno, state)
        return
    m = ASSIGN_RE.match(line)
    if m and m.group(2) == "=":
        var, rhs = m.group(1), m.group(3).strip()
        # alias: avgEntry = strategy.position_avg_price
        if rhs == "strategy.position_avg_price":
            state.sym.avg_entry_aliases.add(var)
            return
        expr = _parse_expr(rhs, state)
        if expr is not None:
            if _classify_session_assign(var, expr, state):
                return
            # a top-level arithmetic over indicators may be a risk quantity
            # (orbRange = orbHigh - orbLow) referenced by later frozen levels
            spec = _risk_eval(expr, state)
            if spec is not None and spec.kind not in ("num", "anchor"):
                state.var_specs[var] = spec
                state.sym.state_vars.add(var)
                return
            state.sym.bools[var] = expr
            return
    elif m and m.group(2) == ":=" and m.group(3).strip() in state.sym.session_vars:
        return  # session-memory update (prevInRTH := inRTH)
    state.warnings.append(f"line {lineno}: skipped (not in subset): {line[:60]}")


def _parse_expr(text: str, state: _State) -> tuple | None:
    # time(...) calls become placeholder atoms carrying their window. The
    # session string is a literal or an input.session variable.
    windows: dict[str, dict] = {}
    bad_time = False

    def _sub(m: re.Match) -> str:
        nonlocal bad_time
        start, end, days, var, tz = m.groups()
        if var is not None:
            sess = state.session_inputs.get(var)
            if sess is None:
                bad_time = True
                return "__badtime__"
            start, end, days = sess
        key = f"__time{len(windows)}__"
        windows[key] = _window_dict(start, end, days, tz)
        return key

    text = TIME_CALL_RE.sub(_sub, text)
    if bad_time:
        return None
    tokens = _tokenize(text)
    if tokens is None:
        return None
    try:
        return _ExprParser(tokens, state.sym, windows).parse_bool()
    except (_ExprError, KeyError):
        return None


# ────────────────────────── block classification ──────────────────────────


def _handle_block(block: _Block, state: _State, gate: tuple) -> None:
    """Classify an if-chain. Recognized shapes: entry blocks, exit blocks,
    close_all blocks, session-bar counters, ORB construction, justEntered
    risk anchoring. Unrecognized blocks degrade to per-line warnings."""
    if _try_stateful_block(block, state):
        return

    for cond_text, body in block.branches:
        cond = _parse_expr(cond_text, state) if cond_text else ("true",)
        if cond is None:
            state.errors.append(ImportError_(
                block.lineno, f"cannot parse condition: {cond_text[:60]!r}"))
            continue
        combined = ("and", [gate, cond])
        _handle_branch_body(body, combined, block.lineno, state)


def _handle_branch_body(body: list[Any], cond: tuple, lineno: int, state: _State) -> None:
    flat = _contains(cond, "pos")
    has_entry = any(isinstance(n, _Stmt) and ENTRY_RE.match(n.text) for n in body)
    just = _just_entered_side(cond)

    for node in body:
        if isinstance(node, _Block):
            _handle_block(node, state, gate=cond)
            continue
        text = node.text
        m = ENTRY_RE.match(text)
        if m:
            when = ENTRY_WHEN_RE.search(text)
            _register_entry(m.group(2), when.group(1) if when else None, cond, node.lineno, state)
            continue
        m = EXIT_RE.match(text)
        if m:
            _register_exit(m.group(1), cond, node.lineno, state)
            continue
        if CLOSE_ALL_RE.match(text):
            if not _handle_close_all(cond, state):
                state.errors.append(ImportError_(
                    node.lineno, "strategy.close_all outside a session-end condition not supported"))
            continue
        if IGNORED_RE.match(text):
            continue
        m = ASSIGN_RE.match(re.sub(r"^(?:int|float|bool)\s+", "", text))
        if m:
            var, op, rhs = m.group(1), m.group(2), m.group(3).strip()
            if op == "+=":
                continue  # counters handled by _try_counter_block
            if rhs == "true" and var in state.sym.state_vars:
                # traded-flag idiom: set inside an entry block → once per session
                if has_entry:
                    state.sym.traded_vars.add(var)
                    state.max_entries_per_session = 1
                continue
            if rhs == "false" and var in state.sym.state_vars:
                continue
            expr = _parse_expr(rhs, state)
            spec = _risk_eval(expr, state) if expr is not None else None
            if spec is not None:
                state.var_specs[var] = spec
                # Block locals join the symbol table so later expressions in the
                # same script (exit stop=/limit= arithmetic) can reference them.
                state.sym.state_vars.add(var)
                if just:
                    _apply_spec_for_side(var, spec, just, state)
                continue
            state.warnings.append(f"line {node.lineno}: skipped (not in subset): {text[:60]}")
            continue
        state.warnings.append(f"line {node.lineno}: skipped (not in subset): {text[:60]}")

    # In a justEntered block, specs were applied above; in entry blocks the
    # entry registration resolves sides. Nothing else to do.
    _ = flat


def _just_entered_side(cond: tuple) -> str | None:
    """justEntered = position_size > 0 and position_size[1] <= 0 (long fill bar)."""
    found = {"long": False, "short": False, "prev": False}

    def walk(n: tuple) -> None:
        if n[0] == "pos":
            if n[1] in ("long", "short"):
                found[n[1]] = True
        elif n[0] == "pos_prev":
            found["prev"] = True
        elif n[0] in ("and", "or"):
            for c in n[1]:
                walk(c)
        elif n[0] == "not":
            walk(n[1])

    walk(cond)
    if found["prev"] and found["long"]:
        return "long"
    if found["prev"] and found["short"]:
        return "short"
    return None


def _contains(expr: tuple, kind: str) -> bool:
    if expr[0] == kind:
        return True
    if expr[0] in ("and", "or"):
        return any(_contains(c, kind) for c in expr[1])
    if expr[0] in ("not", "shift", "na"):
        return _contains(expr[1], kind)
    return False


def _handle_close_all(cond: tuple, state: _State) -> bool:
    """Recognize the flatten-at-session-end shapes:
      if sessionEnd: close_all                      → flat at end of entry window
      if not inSession and position_size != 0: …    → same
      if eodFlag: close_all  (eod = short tail window like 15:55-16:00)
                                                    → entry window truncated to the
                                                      tail's start + flat_at_end
    Returns False if the condition doesn't look session-driven."""
    handled = False

    def walk(n: tuple, negated: bool) -> None:
        nonlocal handled
        if n[0] == "session_end":
            state.flat_at_end = True
            handled = True
        elif n[0] == "session_w":
            if negated:
                if state.session_window is None:
                    state.session_window = n[1]
                state.flat_at_end = True
            else:
                state.eod_window = n[1]
                state.flat_at_end = True
            handled = True
        elif n[0] == "na" and n[1][0] == "timewindow":
            # na(time(...)) == "outside window"
            walk(("session_w", n[1][1]), not negated)
        elif n[0] == "not":
            walk(n[1], not negated)
        elif n[0] in ("and", "or"):
            for c in n[1]:
                walk(c, negated)

    walk(cond, False)
    if handled:
        state.session_used = True
    return handled


ORB_ACC_RE = re.compile(
    r"^(\w+)\s*:=\s*(?:na\s*\(\s*(\w+)\s*\)\s*\?\s*(?:high|low)\s*:\s*)?"
    r"math\.(max|min)\s*\(\s*(\w+)\s*,\s*(high|low)\s*\)$"
)


def _try_stateful_block(block: _Block, state: _State) -> bool:
    """Consume session-state construction blocks: bar counters, opening-range
    seed/accumulate, completion flags, traded-flag resets, session-memory
    updates. A block qualifies only when EVERY statement is a recognized
    stateful mutation and none is a strategy.* call — anything else falls
    through to normal entry/exit handling."""
    actions: list[tuple] = []

    def classify_body(body: list[Any], bound_le: int | None, bound_ge: int | None) -> bool:
        for node in body:
            if isinstance(node, _Block):
                for c2, b2 in node.branches:
                    le, ge = bound_le, bound_ge
                    if c2:
                        e2 = _parse_expr(c2, state)
                        if e2 is None:
                            return False
                        le2, ge2 = _counter_bounds(e2)
                        le = le2 if le2 is not None else le
                        ge = ge2 if ge2 is not None else ge
                    if not classify_body(b2, le, ge):
                        return False
                continue
            if not _classify_stateful_stmt(node.text, state, bound_le, bound_ge, actions):
                return False
        return True

    for cond_text, body in block.branches:
        le = ge = None
        if cond_text:
            cond = _parse_expr(cond_text, state)
            if cond is None:
                return False
            if _contains(cond, "pos"):
                return False  # position-guarded blocks are exit logic, not state
            le, ge = _counter_bounds(cond)
        if not classify_body(body, le, ge):
            return False

    if not actions:
        return False
    meaningful = {"counter", "seed", "accum", "flag_set", "completion"}
    if not any(a[0] in meaningful for a in actions):
        return False

    for action in actions:
        kind = action[0]
        if kind == "accum":
            _, var, side, bars = action
            if var not in state.orb_vars:
                params: dict[str, Any] = {"bars": bars, "side": side}
                state.orb_vars[var] = (side, bars)
                state.sym.indicators[var] = {"type": "ORB", "params": params}
            elif bars is not None:
                state.sym.indicators[var]["params"]["bars"] = bars
                state.orb_vars[var] = (side, bars)
        elif kind == "completion":
            _, var, n = action
            state.completion_flags[var] = n
    state.session_used = True
    return True


def _counter_bounds(cond: tuple) -> tuple[int | None, int | None]:
    """Extract (upper_bound_le, lower_bound_ge) counter comparisons from a
    condition AST (searching and-nodes)."""
    le = ge = None

    def walk(n: tuple) -> None:
        nonlocal le, ge
        if n[0] == "counter_cmp":
            op, _name, v = n[1], n[2], n[3]
            if op == "le":
                le = int(v)
            elif op == "lt":
                le = int(v)  # 0-based `< N` accumulates N bars, like 1-based `<= N`
            elif op in ("ge", "gt"):
                ge = int(v)
        elif n[0] in ("and", "or"):
            for c in n[1]:
                walk(c)
        elif n[0] == "not":
            walk(n[1])

    walk(cond)
    return le, ge


def _classify_stateful_stmt(text: str, state: _State, bound_le: int | None,
                            bound_ge: int | None, actions: list[tuple]) -> bool:
    m = ASSIGN_RE.match(text)
    if not m:
        return False
    var, op, rhs = m.group(1), m.group(2), m.group(3).strip()
    sym = state.sym

    if var in sym.counter_vars:
        if op == "+=" or rhs in ("0", "1") or re.fullmatch(rf"{var}\s*\+\s*1", rhs):
            actions.append(("counter", var))
            return True
        return False

    if op == ":=" and var in sym.state_vars:
        if rhs in ("high", "low", "na"):
            actions.append(("seed", var, rhs))
            return True
        acc = ORB_ACC_RE.match(text)
        if acc and acc.group(1) == acc.group(4) and (acc.group(2) in (None, acc.group(1))):
            side = "high" if acc.group(3) == "max" else "low"
            actions.append(("accum", var, side, bound_le))
            return True
        if rhs == "true":
            if bound_ge is not None:
                actions.append(("completion", var, bound_ge))
            else:
                actions.append(("flag_set", var))
            return True
        if rhs == "false":
            actions.append(("flag_set", var))
            return True
        return False

    # session-memory update: prevInRTH := inRTH
    if op == ":=" and rhs in sym.session_vars:
        actions.append(("session_memory", var))
        return True
    return False


# ────────────────────────── entry / exit registration ──────────────────────────


def _register_entry(side: str, when_var: str | None, cond: tuple, lineno: int, state: _State) -> None:
    full = cond
    if when_var:
        expr = _parse_expr(when_var, state)
        if expr is None:
            state.errors.append(ImportError_(lineno, f"when= signal {when_var!r} is not a known condition"))
            return
        full = ("and", [cond, expr])
    existing = state.long_cond if side == "long" else state.short_cond
    if existing is not None:
        state.errors.append(ImportError_(lineno, f"multiple {side} entries not supported"))
        return
    if side == "long":
        state.long_cond = full
    else:
        state.short_cond = full


def _split_args(args: str) -> list[str]:
    """Split exit() args on top-level commas (parens may nest in exprs)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    in_str: str | None = None
    for ch in args:
        if in_str:
            cur.append(ch)
            if ch == in_str:
                in_str = None
            continue
        if ch in "\"'":
            in_str = ch
            cur.append(ch)
        elif ch in "([":
            depth += 1
            cur.append(ch)
        elif ch in ")]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return parts


def _register_exit(args: str, cond: tuple, lineno: int, state: _State) -> None:
    parts = _split_args(args)
    kwargs: dict[str, str] = {}
    positional: list[str] = []
    for p in parts:
        m = re.match(r"^(\w+)\s*=\s*(.+)$", p)
        if m:
            kwargs[m.group(1)] = m.group(2).strip()
        else:
            positional.append(p)

    # Side: from_entry kwarg, 2nd positional string, or the enclosing
    # position_size guard.
    side: str | None = None
    fe = kwargs.get("from_entry") or (positional[1] if len(positional) > 1 else None)
    if fe:
        fe_l = fe.strip('"').lower()
        if "long" in fe_l:
            side = "long"
        elif "short" in fe_l:
            side = "short"
    if side is None:
        if _contains_pos(cond, "long"):
            side = "long"
        elif _contains_pos(cond, "short"):
            side = "short"

    loss = _to_float(kwargs.get("loss"))
    profit = _to_float(kwargs.get("profit"))
    stop_expr = kwargs.get("stop")
    limit_expr = kwargs.get("limit")

    if stop_expr or limit_expr:
        failed: list[str] = []
        if stop_expr:
            spec = _resolve_exit_expr(stop_expr, state)
            if spec is not None and spec.kind in ("atr_stop", "ind_level", "range_level"):
                _set_spec(state.stop_spec, side, spec)
            else:
                failed.append(f"stop={stop_expr}")
        if limit_expr:
            spec = _resolve_exit_expr(limit_expr, state)
            # atr_stop as a LIMIT is the direct ATR-multiple target idiom
            # (tp := close + atr * mult).
            if spec is not None and spec.kind in ("rr_target", "range_rr_target", "atr_stop"):
                _set_spec(state.target_spec, side, spec)
            else:
                failed.append(f"limit={limit_expr}")
        if failed:
            state.errors.append(ImportError_(
                lineno,
                f"strategy.exit() {', '.join(failed)} — must resolve to a frozen risk "
                "level (ATR distance, indicator level, or R-multiple of a captured "
                "distance/range)",
            ))
        return

    if loss is None and profit is None:
        state.errors.append(ImportError_(
            lineno, "strategy.exit() needs loss=/profit= points or stop=/limit= frozen levels"))
        return
    sides = [side] if side else ["long", "short"]
    if side is None:
        state.warnings.append("strategy.exit without from_entry — applying to both long & short")
    for s in sides:
        if loss is not None:
            state.stop_points[s] = loss
        if profit is not None:
            state.target_points[s] = profit


def _contains_pos(expr: tuple, which: str) -> bool:
    if expr[0] == "pos" and expr[1] == which:
        return True
    if expr[0] in ("and", "or"):
        return any(_contains_pos(c, which) for c in expr[1])
    if expr[0] == "not":
        return False  # `not (pos long)` is not a long guard
    return False


def _to_float(tok: str | None) -> float | None:
    if tok is None:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _resolve_exit_expr(text: str, state: _State) -> _RiskSpec | None:
    """stop=/limit= value: a bare var carrying a frozen spec, or an inline
    anchor-relative expression (avgEntry - stopDist etc.)."""
    text = text.strip()
    if re.fullmatch(r"[A-Za-z_]\w*", text):
        if text in state.var_specs:
            return state.var_specs[text]
        if text in state.orb_vars:
            side, _ = state.orb_vars[text]
            return _RiskSpec("ind_level", {"orb_side": side})
        ind = state.sym.indicators.get(text)
        if ind:
            return _RiskSpec("ind_level", {"indicator_var": text})
        return None
    expr = _parse_expr(text, state)
    if expr is None:
        return None
    return _risk_eval(expr, state)


def _set_spec(specs: dict[str, _RiskSpec | None], side: str | None, spec: _RiskSpec) -> None:
    for s in ([side] if side else ["long", "short"]):
        if specs.get(s) is None:
            specs[s] = spec


def _apply_spec_for_side(var: str, spec: _RiskSpec, side: str, state: _State) -> None:
    if spec.kind == "atr_stop":
        _set_spec(state.stop_spec, side, spec)
    elif spec.kind in ("rr_target", "range_rr_target"):
        _set_spec(state.target_spec, side, spec)


# ────────────────────────── condition → schema tree ──────────────────────────


class _CondError(Exception):
    pass


def _to_schema_tree(expr: tuple, state: _State, id_map: dict[str, str]) -> dict | None:
    """Normalize an expr AST into a schema ConditionTree (or None if the whole
    expression folds away — e.g. pure session/flat gates)."""
    node = _normalize(expr, state)
    if node is None or node == ("true",):
        return None
    return _emit(node, state, id_map)


def _normalize(expr: tuple, state: _State) -> tuple | None:
    t = expr[0]
    if t in ("true", "flat"):
        # flat-only entries are the engine default
        return ("true",)
    if t == "session_w":
        _register_entry_window(expr[1], state)
        return ("true",)
    if t == "false":
        return ("false",)
    if t == "counter_cmp":
        # Session-bar counter comparisons become session-window entry bounds.
        # 0-based counters use >= N, 1-based use > N — both mean "entries allowed
        # from the (N+1)-th session bar", which is the engine's warmup_bars=N.
        op, _name, n = expr[1], expr[2], expr[3]
        state.session_used = True
        if op in ("ge", "gt"):
            state.warmup_bars = max(state.warmup_bars or 0, int(n))
            return ("true",)
        if op in ("le", "lt"):
            bound = int(n) if op == "le" else int(n) - 1
            state.max_entry_bars = bound if state.max_entry_bars is None else min(state.max_entry_bars, bound)
            return ("true",)
        raise _CondError(f"unsupported counter comparison {op}")
    if t == "traded":
        return expr  # meaningful only under `not`; bare use errors at emit
    if t == "statevar":
        name = expr[1]
        if name in state.sym.traded_vars:
            # promotion happens when the entry block is processed, AFTER the
            # condition var was parsed — reclassify at normalize time
            return ("traded", name)
        if name in state.completion_flags:
            # "opening range complete" flag == session warm-up of N bars
            state.session_used = True
            state.warmup_bars = max(state.warmup_bars or 0, state.completion_flags[name])
            return ("true",)
        raise _CondError(f"stateful var {name!r} in a condition")
    if t == "not":
        inner = expr[1]
        if inner[0] == "na":
            if inner[1][0] == "timewindow":
                # not na(time(...)) — session membership test
                _register_entry_window(inner[1][1], state)
                return ("true",)
            # not na(x) — warm-up guard; engine NaN-comparisons are already false
            return ("true",)
        if inner[0] == "session_w":
            raise _CondError("`not inSession` inside an entry condition")
        sub = _normalize(inner, state)
        if sub[0] == "traded":
            state.max_entries_per_session = 1
            return ("true",)
        if sub == ("true",):
            return ("false",)
        if sub == ("false",):
            return ("true",)
        return ("not", sub)
    if t == "na":
        raise _CondError("bare na(x) in a condition")
    if t in ("and", "or"):
        subs = []
        for c in expr[1]:
            s = _normalize(c, state)
            if s is None:
                s = ("true",)
            subs.append(s)
        if t == "and":
            # flip idiom: cmp(x op c) and shift(cmp(x op' c), 1) → cross
            flip = _detect_flip(subs)
            if flip is not None:
                subs = [s for s in subs if s[0] not in ("cmp", "shift")
                        or not _is_flip_member(s, flip)] + [flip["node"]]
            subs = [s for s in subs if s != ("true",)]
            if any(s == ("false",) for s in subs):
                return ("false",)
            if not subs:
                return ("true",)
            return subs[0] if len(subs) == 1 else ("and", subs)
        subs_o = [s for s in subs if s != ("false",)]
        if any(s == ("true",) for s in subs_o):
            return ("true",)
        if not subs_o:
            return ("false",)
        return subs_o[0] if len(subs_o) == 1 else ("or", subs_o)
    if t == "shift":
        inner = expr[1]
        if inner[0] == "session_w":
            # inSession[1] alone (rare) — approximate with in-session
            _register_entry_window(inner[1], state)
            return ("true",)
        return ("shift", _normalize(inner, state), expr[2])
    if t == "cross":
        return expr
    if t == "cmp":
        return expr
    if t in ("pos", "pos_prev"):
        # entry conditions never legitimately reference open-position state
        # beyond flat-gates (already normalized); treat as fold-away
        return ("true",)
    if t == "operand":
        raise _CondError(f"bare operand {expr[1]!r} used as a boolean")
    raise _CondError(f"cannot normalize {t!r} in a condition")


def _register_entry_window(window: dict, state: _State) -> None:
    state.session_used = True
    if state.session_window is None:
        state.session_window = window
    elif state.session_window != window:
        raise _CondError("multiple distinct entry session windows")


def _detect_flip(subs: list[tuple]) -> dict | None:
    """AND(cmp(x,op,c), shift(cmp(x,op',c),1)) with opposite ops → cross."""
    cmps = [s for s in subs if s[0] == "cmp"]
    shifts = [s for s in subs if s[0] == "shift" and s[1] is not None and s[1][0] == "cmp" and s[2] == 1]
    for a in cmps:
        for b in shifts:
            bb = b[1]
            if a[2:] == bb[2:] or (a[2], a[3]) == (bb[2], bb[3]):
                opposites = {("lt", "gt"), ("gt", "lt"), ("lt", "ge"), ("gt", "le"),
                             ("le", "gt"), ("ge", "lt")}
                if (a[1], bb[1]) in opposites:
                    ctype = "cross_below" if a[1] in ("lt", "le") else "cross_above"
                    return {"node": ("cross", ctype, a[2], a[3]), "a": a, "b": b}
    return None


def _is_flip_member(s: tuple, flip: dict) -> bool:
    return s is flip["a"] or s is flip["b"]


def _emit(node: tuple, state: _State, id_map: dict[str, str]) -> dict:
    t = node[0]
    if t == "and" or t == "or":
        return {"type": t, "conditions": [_emit(c, state, id_map) for c in node[1]]}
    if t == "not":
        return {"type": "not", "conditions": [_emit(node[1], state, id_map)]}
    if t == "cross":
        return {"type": node[1], "left": _emit_operand(node[2], state, id_map),
                "right": _emit_operand(node[3], state, id_map)}
    if t == "cmp":
        op, left, right = node[1], node[2], node[3]
        if op in ("gt", "lt", "eq"):
            return {"type": op, "left": _emit_operand(left, state, id_map),
                    "right": _emit_operand(right, state, id_map)}
        if op == "ge":
            return {"type": "not", "conditions": [{
                "type": "lt", "left": _emit_operand(left, state, id_map),
                "right": _emit_operand(right, state, id_map)}]}
        if op == "le":
            return {"type": "not", "conditions": [{
                "type": "gt", "left": _emit_operand(left, state, id_map),
                "right": _emit_operand(right, state, id_map)}]}
        if op == "ne":
            return {"type": "not", "conditions": [{
                "type": "eq", "left": _emit_operand(left, state, id_map),
                "right": _emit_operand(right, state, id_map)}]}
    if t == "shift":
        raise _CondError("history reference [n] survives normalization — unsupported shape")
    raise _CondError(f"cannot emit condition node {t!r}")


def _emit_operand(node: tuple, state: _State, id_map: dict[str, str]) -> Any:
    if node[0] == "num":
        return node[1]
    if node[0] == "operand":
        name = node[1]
        return id_map.get(name, name)
    raise _CondError(f"operand too complex: {node!r} — assign it to an indicator variable first")


# ────────────────────────── finalization ──────────────────────────


def _finalize(state: _State) -> ImportResult:
    errors = list(state.errors)

    if state.pine_version is None:
        errors.append(ImportError_(0, "missing `//@version=5` (or `//@version=6`) directive"))
    elif state.pine_version not in (5, 6):
        errors.append(ImportError_(0, f"unsupported Pine version {state.pine_version}; we support v5 and v6"))

    if not state.name:
        errors.append(ImportError_(0, "no strategy() declaration found"))

    if state.long_cond is None and state.short_cond is None:
        errors.append(ImportError_(0, "no entry signals detected (need strategy.entry inside a condition)"))

    # ORB indicators whose bar count came from a completion flag get it now
    # (the flag is recognized after the accumulation statement).
    for var, (side, bars) in list(state.orb_vars.items()):
        if bars is None:
            n = next(iter(state.completion_flags.values()), None)
            if n is None:
                errors.append(ImportError_(0, f"opening-range var {var!r}: bar count not found"))
            else:
                state.sym.indicators[var]["params"]["bars"] = n
                state.orb_vars[var] = (side, n)

    id_map = {var: _snake(var) for var in state.sym.indicators}

    entry_obj: dict[str, Any] = {}
    for side, cond in (("long", state.long_cond), ("short", state.short_cond)):
        if cond is None:
            continue
        try:
            tree = _to_schema_tree(cond, state, id_map)
        except _CondError as e:
            errors.append(ImportError_(0, f"{side} entry condition: {e}"))
            continue
        if tree is None:
            errors.append(ImportError_(0, f"{side} entry condition folded to nothing — no actionable signal"))
            continue
        entry_obj[side] = tree

    if errors:
        return ImportResult(
            ok=False, strategy_json=None, errors=errors,
            warnings=state.warnings, unsupported_features=state.unsupported,
            pine_version=state.pine_version, detected_name=state.name or None,
        )

    if state.max_entries_per_session is not None:
        entry_obj["max_entries_per_session"] = state.max_entries_per_session

    exit_obj = _exit_json(state, id_map, errors)
    if errors:
        return ImportResult(
            ok=False, strategy_json=None, errors=errors,
            warnings=state.warnings, unsupported_features=state.unsupported,
            pine_version=state.pine_version, detected_name=state.name or None,
        )

    # Entry conditions may never name the session directly (opening-range
    # completion flags carry the session semantics) — fall back to the script's
    # sole session window when session machinery was used.
    if state.session_window is None and state.session_used:
        distinct = {tuple(sorted(w.items(), key=str)): w for w in state.sym.session_vars.values()}
        if len(distinct) == 1:
            state.session_window = next(iter(distinct.values()))

    session_filter: dict[str, Any] = {"type": "always"}
    if state.session_window is not None:
        window = dict(state.session_window)
        if state.eod_window is not None:
            # A close_all tail window (e.g. 15:55-16:00 inside 09:30-16:00) means
            # "be flat from the tail's start" — truncate the session there.
            if state.eod_window.get("tz") == window.get("tz") and \
                    state.eod_window["start"] > window["start"]:
                window["end"] = min(window["end"], state.eod_window["start"])
        session_filter = {"type": "session_window", "windows": [window]}
        if state.flat_at_end:
            session_filter["flat_at_end"] = True
        if state.warmup_bars:
            session_filter["warmup_bars"] = state.warmup_bars
        if state.max_entry_bars:
            session_filter["max_entry_bars"] = state.max_entry_bars

    indicators_json = [
        {"id": id_map[var], "type": meta["type"], "params": meta["params"]}
        for var, meta in state.sym.indicators.items()
    ]

    risk: dict[str, Any]
    if state.default_qty is not None:
        risk = {"model": "fixed_contracts", "fixed_contracts": max(1, int(state.default_qty)),
                "max_concurrent_positions": 1}
    else:
        risk = {"model": "fixed_fractional", "risk_per_trade": 0.01, "max_concurrent_positions": 1}
    if state.commission_per_contract is not None:
        risk["commission_per_contract"] = state.commission_per_contract
    if state.slippage_points is not None:
        risk["slippage_points"] = state.slippage_points

    schema_json = {
        "schema_version": "0.1.0",
        "id": _slugify(state.name),
        "name": state.name,
        "description": f"Imported from Pine v{state.pine_version}",
        "market": {
            "instrument": "MNQ",       # default; user can edit before running
            "instrument_type": "futures",
            "timeframe": "5m",
            "session_filter": session_filter,
        },
        "indicators": indicators_json,
        "entry": entry_obj,
        "exit": exit_obj,
        "risk": risk,
        "metadata": {
            "source_format": "pine_import",
        },
    }

    return ImportResult(
        ok=True, strategy_json=schema_json, errors=[],
        warnings=state.warnings, unsupported_features=state.unsupported,
        pine_version=state.pine_version, detected_name=state.name,
        pine_inputs=state.inputs_meta,
    )


def _exit_json(state: _State, id_map: dict[str, str], errors: list[ImportError_]) -> dict[str, Any]:
    exit_obj: dict[str, Any] = {"signal_exit": "opposite"}

    def orb_id(side_key: str) -> str | None:
        for var, (side, _) in state.orb_vars.items():
            if side == side_key:
                return id_map[var]
        return None

    stop = state.stop_spec["long"] or state.stop_spec["short"]
    target = state.target_spec["long"] or state.target_spec["short"]
    stop_pts = state.stop_points["long"] if state.stop_points["long"] is not None else state.stop_points["short"]
    target_pts = state.target_points["long"] if state.target_points["long"] is not None else state.target_points["short"]

    if stop is not None:
        if stop.kind == "atr_stop":
            exit_obj["stop_loss"] = {"type": "atr", "atr_length": stop.data["atr_length"],
                                     "atr_multiplier": stop.data["atr_multiplier"]}
        elif stop.kind == "ind_level":
            long_spec = state.stop_spec["long"]
            short_spec = state.stop_spec["short"]

            def _level_id(spec: _RiskSpec | None) -> str | None:
                if spec is None or spec.kind != "ind_level":
                    return None
                if "orb_side" in spec.data:
                    return orb_id(spec.data["orb_side"])
                return id_map.get(spec.data["indicator_var"])

            long_id = _level_id(long_spec) or _level_id(short_spec)
            short_id = _level_id(short_spec)
            if long_id is None:
                errors.append(ImportError_(0, "indicator stop level not recognized"))
                return exit_obj
            exit_obj["stop_loss"] = {"type": "indicator", "indicator_id": long_id}
            if short_id and short_id != long_id:
                exit_obj["stop_loss"]["indicator_id_short"] = short_id
        elif stop.kind == "range_level":
            up = stop.data.get("upper") or {}
            lo = stop.data.get("lower") or {}
            up_id = orb_id("high") if "orb_side" in up else id_map.get(up.get("indicator_var", ""))
            lo_id = orb_id("low") if "orb_side" in lo else id_map.get(lo.get("indicator_var", ""))
            if not up_id or not lo_id:
                errors.append(ImportError_(0, "range_level stop band not recognized"))
                return exit_obj
            exit_obj["stop_loss"] = {"type": "range_level", "upper": up_id, "lower": lo_id,
                                     "offset_ratio": stop.data["offset_ratio"]}
    elif stop_pts is not None:
        exit_obj["stop_loss"] = {"type": "fixed_points", "points": stop_pts}

    if target is not None:
        if target.kind == "rr_target":
            exit_obj["take_profit"] = {"type": "rr", "ratio": target.data["ratio"]}
        elif target.kind == "atr_stop":
            # anchor ± atr_dist used as a limit → direct ATR-multiple target
            exit_obj["take_profit"] = {"type": "atr", "atr_length": target.data["atr_length"],
                                       "atr_multiplier": target.data["atr_multiplier"]}
        elif target.kind == "range_rr_target":
            upper = target.data.get("upper") or {}
            lower = target.data.get("lower") or {}
            up_id = orb_id("high") if "orb_side" in upper else id_map.get(upper.get("indicator_var", ""))
            lo_id = orb_id("low") if "orb_side" in lower else id_map.get(lower.get("indicator_var", ""))
            if not up_id or not lo_id:
                errors.append(ImportError_(0, "range R target but band indicators not recognized"))
                return exit_obj
            exit_obj["take_profit"] = {"type": "range_rr", "ratio": target.data["ratio"],
                                       "upper": up_id, "lower": lo_id}
    elif target_pts is not None:
        exit_obj["take_profit"] = {"type": "fixed_points", "points": target_pts}

    return exit_obj


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return s or "imported-strategy"


def _snake(var: str) -> str:
    """camelCase Pine var -> schema-legal snake_case id (^[a-z][a-z0-9_]{0,31}$)."""
    s = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", var).lower()
    s = re.sub(r"[^a-z0-9_]", "_", s).lstrip("_0123456789")
    return (s or "ind")[:32]
