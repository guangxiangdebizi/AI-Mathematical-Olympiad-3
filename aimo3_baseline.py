"""
AIMO3 Inference Script — vLLM + TIR + Majority Voting + Time Budget
Model: Qwen2.5-Math-72B-Instruct

Strategy:
  1. Load Qwen2.5-Math-72B via vLLM (tensor_parallel_size=2 for 2xH100)
  2. Tool-Integrated Reasoning (TIR): model writes Python → sandbox executes → feeds result back
  3. N-sample majority voting: generate N answers per problem, pick most common
  4. Time budget: (9h - startup_reserve) / n_problems per problem
  5. Answer extraction: parse \\boxed{} or last integer → mod 100000

Usage (Kaggle notebook, run as a cell):
    %run aimo3_baseline.py

Usage (Kaggle notebook, formal submit):
    Save Version → Submit to Competition
    (KAGGLE_IS_COMPETITION_RERUN env var triggers serve() automatically)

Usage (local debug, CLI):
    python aimo3_baseline.py --test-path /path/to/test.csv --n-samples 2 --dry-run
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import glob
import io
import math
import multiprocessing
import os
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional

import polars as pl

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MOD = 100_000
TOTAL_BUDGET_SECONDS = 9 * 3600          # 9-hour hard limit from gateway
STARTUP_RESERVE_SECONDS = 600            # 10-min reserve for model load + buffer
N_PROBLEMS = 110                         # hidden test set size (official)
DEFAULT_N_SAMPLES = 8                    # majority-vote samples per problem
# token 上限设极大值（32768），让模型推理过程不被中途截断
# 实际截止靠 per-problem 时间预算控制，不靠 token 数
MAX_NEW_TOKENS = 32768
MAX_CODE_EXEC_SECONDS = 30               # sandbox timeout per code execution（宽松）
MAX_SINGLE_SAMPLE_WEIGHT = 1.65          # 1.0 + boxed(0.35) + tool bonus(<=0.30)

# Kaggle Model path (from the Input panel)
# AWQ 量化版优先（单卡 H100 可用），其次 fallback 到原始版本
# Kaggle 实际挂载路径格式：/kaggle/input/models/<owner>/<model>/...
MODEL_SEARCH_ROOTS = [
    "/kaggle/input/models/shelterw/qwen2.5-math/transformers/qwen2.5-math-72b-instruct-awq/V1",
    "/kaggle/input/models/shelterw/qwen2.5-math/transformers/qwen2.5-math-72b-instruct-awq/1",
    "/kaggle/input/shelterw/qwen2.5-math/transformers/qwen2.5-math-72b-instruct-awq/V1",
    "/kaggle/input/shelterw/qwen2.5-math/transformers/qwen2.5-math-72b-instruct-awq/1",
    "/kaggle/input/qwen-lm/qwen2.5-math/transformers/72b/V1",
    "/kaggle/input/qwen-lm/qwen2.5-math/transformers/72b/1",
    "/kaggle/input/models/qwen-lm/qwen2.5-math/transformers/72b/V1",
    "/kaggle/input/models/qwen-lm/qwen2.5-math/transformers/72b/1",
]


# ──────────────────────────────────────────────
# 0. Path helpers
# ──────────────────────────────────────────────

def _append_sys_path(path: str) -> None:
    if path and path not in sys.path:
        sys.path.insert(0, path)


def find_kaggle_eval_paths() -> tuple[Optional[str], Optional[str]]:
    """Return (comp_root, eval_dir) or (None, None)."""
    patterns = [
        "/kaggle/input/**/kaggle_evaluation/aimo_3_inference_server.py",
        "/kaggle/input/competitions/**/kaggle_evaluation/aimo_3_inference_server.py",
    ]
    hits: list[str] = []
    for pat in patterns:
        hits.extend(glob.glob(pat, recursive=True))
    if not hits:
        return None, None
    server_py = hits[0]
    eval_dir = str(Path(server_py).parent)
    comp_root = str(Path(eval_dir).parent)
    return comp_root, eval_dir


def import_inference_server(eval_dir: str, comp_root: str):
    _append_sys_path(comp_root)
    _append_sys_path(eval_dir)
    from aimo_3_inference_server import AIMO3InferenceServer  # pylint: disable=import-error
    return AIMO3InferenceServer


def find_model_path() -> str:
    """Locate model weights dir — AWQ quantized version preferred."""
    # 1) Check explicit roots first (AWQ listed before non-quantized)
    for root in MODEL_SEARCH_ROOTS:
        if os.path.isdir(root) and os.path.exists(os.path.join(root, "config.json")):
            print(f"[INFO] model found at: {root}")
            return root

    # 2) Glob fallback: prefer AWQ/GPTQ quantized models
    all_hits = glob.glob("/kaggle/input/**/config.json", recursive=True)
    awq_hits = [h for h in all_hits if "awq" in h.lower() or "gptq" in h.lower()]
    math_hits = [h for h in all_hits if "qwen" in h.lower() and "math" in h.lower()]

    for hits in [awq_hits, math_hits]:
        if hits:
            p = str(Path(hits[0]).parent)
            print(f"[INFO] model found (glob fallback) at: {p}")
            return p

    raise FileNotFoundError(
        "Cannot locate Qwen2.5-Math model weights.\n"
        "In Kaggle notebook: Add Input → Models → search 'qwen2.5-math-72b-instruct-awq'."
    )


def resolve_test_path(explicit: Optional[str], comp_root: Optional[str]) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    if comp_root:
        c = os.path.join(comp_root, "test.csv")
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError("Cannot find test.csv.")


# ──────────────────────────────────────────────
# 1. Context Engineering
# ──────────────────────────────────────────────

# System prompt: role + tool guidance only, NO modulo (handled in code)
SYSTEM_PROMPT = """\
You are an expert mathematician specializing in mathematical olympiad problems \
(algebra, number theory, combinatorics, and geometry at IMO level).

You may write Python code in ```python ... ``` blocks to assist with computations. \
The code will be executed and the output returned to you. \
Use libraries such as sympy, math, itertools, fractions, and collections freely.

Always reason step by step. At the very end of your solution, \
write your final integer answer inside \\boxed{} — for example \\boxed{42}.\
"""

# ── Topic classifier ──────────────────────────

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "geometry": [
        "triangle", "circle", "circumcircle", "incircle", "tangent", "chord",
        "angle", "perpendicular", "polygon", "quadrilateral", "line segment",
        "parallelogram", "trapezoid", "inscribed", "collinear", "concyclic",
        r"\angle", r"\triangle", r"\overline", "radius", "diameter",
    ],
    "number_theory": [
        "prime", "divisor", "divisible", "remainder", "modulo", "gcd", "lcm",
        "congruent", "floor", "ceil", r"\lfloor", r"\lceil", "digit",
        "integer", "base", "factori", "p-adic", "coprime", "totient",
        "arithmetic progression", "Fibonacci",
    ],
    "combinatorics": [
        "permutation", "combination", "counting", "arrangement", "tournament",
        "graph", "coloring", "path", "sequence", "subset", "partition",
        "choose", r"\binom", "probability", "expected", "ways", "ordering",
        "binary", "tree", "grid",
    ],
    "algebra": [
        "function", "polynomial", "equation", "inequality", "real number",
        "complex", "root", r"\sum", r"\prod", "matrix", "determinant",
        "minimum", "maximum", "optimize", "series", "limit", "continuous",
        "functional equation", r"\mathbb{R}", r"\mathbb{Z}",
    ],
}


def classify_problem(problem_text: str) -> str:
    """Return one of: geometry | number_theory | combinatorics | algebra | unknown."""
    text_lower = problem_text.lower()
    scores: dict[str, int] = {t: 0 for t in _TOPIC_KEYWORDS}
    for topic, kws in _TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text_lower:
                scores[topic] += 1
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "unknown"


# ── Topic-specific strategy hints ─────────────

_TOPIC_HINTS: dict[str, str] = {
    "geometry": (
        "This is a geometry problem. Strategies to consider:\n"
        "- Introduce coordinates if lengths/angles are involved.\n"
        "- Use sympy.geometry or sympy trigonometric/algebraic tools.\n"
        "- Look for cyclic quadrilaterals, power of a point, or radical axes.\n"
        "- For olympiad geometry, chase angles or use inversion when circles appear.\n"
        "- State any key lemmas before using them."
    ),
    "number_theory": (
        "This is a number theory problem. Strategies to consider:\n"
        "- Factor integers and analyze prime factorizations with sympy.factorint.\n"
        "- Use modular arithmetic: Python's pow(a, b, m) for fast modular exponentiation.\n"
        "- For floor/ceiling sums, try to find closed forms or telescoping.\n"
        "- For divisibility, look for lifting-the-exponent (LTE) or p-adic valuations.\n"
        "- Verify conjectures computationally before proving."
    ),
    "combinatorics": (
        "This is a combinatorics problem. Strategies to consider:\n"
        "- Use itertools or dynamic programming to count small cases first.\n"
        "- Look for bijections, generating functions, or recursive structure.\n"
        "- For counting orderings/permutations, consider symmetry and Burnside's lemma.\n"
        "- Build up from small examples to find patterns, then prove the pattern.\n"
        "- Estimate the answer order of magnitude before computing."
    ),
    "algebra": (
        "This is an algebra problem. Strategies to consider:\n"
        "- For functional equations, try substituting special values (0, 1, x=y, etc.).\n"
        "- For polynomial problems, use sympy.solve or sympy.factor.\n"
        "- For inequalities, try AM-GM, Cauchy-Schwarz, or SOS decomposition.\n"
        "- For optimization, check boundary cases and use calculus or Lagrange multipliers.\n"
        "- Verify all solutions satisfy the original constraints."
    ),
    "unknown": (
        "Carefully read the problem to identify its type (algebra, number theory, "
        "combinatorics, or geometry), then choose a suitable strategy."
    ),
}

# ── Few-shot examples (one per topic, from reference.csv) ─────────────────────
# Each example shows: problem → brief solution sketch → answer
# Purpose: show the model the expected reasoning style and answer format.

_FEW_SHOT_EXAMPLES: dict[str, tuple[str, str]] = {
    "algebra": (
        # 92ba6a — Alice and Bob ages (simplest reference problem)
        "Alice and Bob are each holding some integer number of sweets. "
        "Alice says: 'If we each added the number of sweets we're holding to our "
        "(positive integer) age, my answer would be double yours. If we took the "
        "product, then my answer would be four times yours.' "
        "Bob replies: 'Why don't you give me five of your sweets because then both "
        "our sum and product would be equal.' "
        "What is the product of Alice and Bob's ages?",
        # solution sketch
        """\
Let Alice have sweets $a$, age $p$; Bob have sweets $b$, age $q$.
From the conditions:
  (a + p) = 2(b + q)          ... (1)
  (a * p) = 4(b * q)          ... (2)
  After transfer: (a-5) + (b+5) = sum equal, and (a-5)(b+5) = product equal => (a-5) = (b+5), so a = b+10.

```python
from sympy import symbols, solve, Integer
a, b, p, q = symbols('a b p q', positive=True, integer=True)
sols = solve([
    a + p - 2*(b + q),
    a*p - 4*(b*q),
    (a - 5) - (b + 5),         # a-5 == b+5
    (a - 5)*(b + 5) - (a - 5 + b + 5)**2 // 4  # sum==product after transfer
], [a, b, p, q], dict=True)
print(sols)
```

After solving: p=10, q=5, so the product of ages = 10*5 = 50.
\\boxed{50}""",
    ),
    "number_theory": (
        # 42d360 — Ken blackboard base representation
        r"On a blackboard, Ken starts off by writing a positive integer $n$ and then "
        r"applies the following move until he first reaches $1$. Given that the number "
        r"on the board is $m$, he chooses a base $b$, where $2 \leq b \leq m$, and "
        r"replaces $m$ with the sum of its base-$b$ digits. "
        r"Across all choices of $1 \leq n \leq 10^{10^5}$, the largest possible number "
        r"of moves Ken could make is $M$. What is the remainder when $M$ is divided by $10^5$?",
        # solution sketch
        r"""\
Key insight: each move reduces $m$ drastically. To maximize moves, we want to keep $m$
large as long as possible. The optimal strategy uses base $b=2$ repeatedly.

```python
def digit_sum(m, b):
    s = 0
    while m:
        s += m % b
        m //= b
    return s

def max_moves(start):
    m, moves = start, 0
    while m > 1:
        # try all bases, pick the one giving the largest next value
        best = 0
        for b in range(2, m+1):
            best = max(best, digit_sum(m, b))
        m = best
        moves += 1
    return moves

# Check small cases to find the pattern
for n in range(2, 20):
    print(n, max_moves(n))
```

After analysis, $M \equiv 32193 \pmod{10^5}$.
\\boxed{32193}""",
    ),
    "combinatorics": (
        # a295e9 — 500x500 square divided into rectangles
        r"A $500 \times 500$ square is divided into $k$ rectangles, each having integer "
        r"side lengths. Given that no two of these rectangles have the same perimeter, "
        r"the largest possible value of $k$ is $\mathcal{K}$. "
        r"What is the remainder when $\mathcal{K}$ is divided by $10^5$?",
        # solution sketch
        r"""\
Each rectangle has integer sides $a \times b$ with perimeter $2(a+b)$.
We need all perimeters distinct, and rectangles must tile the $500 \times 500$ square.

```python
# Perimeters of integer-sided rectangles fitting inside 500x500:
# perimeter = 2(a+b), with 1<=a<=500, 1<=b<=500
# Minimum perimeter = 2*(1+1)=4, max = 2*(500+500)=2000
# Number of distinct possible perimeters:
perimeters = set()
for a in range(1, 501):
    for b in range(1, 501):
        perimeters.add(2*(a+b))
print(len(perimeters))  # upper bound on k
```

After careful analysis of tiling constraints, $\mathcal{K} \equiv 520 \pmod{10^5}$.
\\boxed{520}""",
    ),
    "geometry": (
        # 0e644e — ABC triangle with circles
        r"Let $ABC$ be an acute-angled triangle with integer side lengths and $AB < AC$. "
        r"Points $D$ and $E$ lie on segments $BC$ and $AC$, respectively, such that "
        r"$AD = AE = AB$. Line $DE$ intersects $AB$ at $X$. "
        r"Circles $BXD$ and $CED$ intersect for the second time at $Y \neq D$. "
        r"Suppose that $Y$ lies on line $AD$. "
        r"There is a unique such triangle with minimal perimeter. "
        r"This triangle has side lengths $a=BC$, $b=CA$, and $c=AB$. "
        r"Find the remainder when $abc$ is divided by $10^5$.",
        # solution sketch
        r"""\
Let $c = AB$, $b = CA$, $a = BC$ with $c < b$ and all integers.
Since $AD = AE = AB = c$, point $E$ is on $AC$ with $AE = c$, so $EC = b - c$.

```python
from math import gcd
results = []
for c in range(1, 300):
    for b in range(c+1, 300):       # b > c since AB < AC
        for a in range(abs(b-c)+1, b+c):  # triangle inequality
            # Check acute: a^2+b^2>c^2, a^2+c^2>b^2, b^2+c^2>a^2
            if a**2+b**2 <= c**2: continue
            if a**2+c**2 <= b**2: continue
            if b**2+c**2 <= a**2: continue
            # Y on line AD condition (derived algebraically)
            # ... [olympiad condition checked symbolically]
            pass
# After checking the geometric condition: minimal perimeter gives abc % 10^5 = 336
print(336)
```

\\boxed{336}""",
    ),
}


def build_messages(problem_text: str, topic: str) -> list[dict]:
    """
    Build the full message list for a problem:
      [system] → [few-shot user] → [few-shot assistant] → [actual user]
    The few-shot pair shows the model the expected solution style for this topic.
    """
    hint = _TOPIC_HINTS.get(topic, _TOPIC_HINTS["unknown"])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Insert one few-shot example for the detected topic
    if topic in _FEW_SHOT_EXAMPLES:
        ex_problem, ex_solution = _FEW_SHOT_EXAMPLES[topic]
        messages.append({
            "role": "user",
            "content": (
                f"[Example {topic} problem]\n{ex_problem}\n\n"
                f"[Strategy hint]\n{hint}"
            ),
        })
        messages.append({
            "role": "assistant",
            "content": ex_solution,
        })

    # Actual problem
    messages.append({
        "role": "user",
        "content": (
            f"Now solve this problem:\n\n{problem_text}\n\n"
            f"[Strategy hint]\n{hint}\n\n"
            "Show your reasoning step by step, use Python code where helpful, "
            "and give your final answer as \\boxed{<integer>}."
        ),
    })

    return messages


# ── Context length protection ─────────────────

def _trim_messages_to_fit(
    messages: list[dict],
    tokenizer,
    max_context_tokens: int = 3000,
) -> list[dict]:
    """
    If the rendered prompt exceeds max_context_tokens, progressively remove
    the oldest tool output (user messages containing ```output```) from the
    middle of the conversation, keeping system + first user + latest turns intact.
    """
    while True:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        if len(prompt) <= max_context_tokens:
            break
        # Find oldest trimable message (tool output injected during TIR)
        trimmed = False
        for i in range(1, len(messages) - 1):
            if messages[i]["role"] == "user" and "```output" in messages[i].get("content", ""):
                messages[i] = {
                    "role": "user",
                    "content": "[previous code output truncated to save context]",
                }
                trimmed = True
                break
        if not trimmed:
            break  # nothing left to trim safely
    return messages


# ──────────────────────────────────────────────
# 2. Safe code execution sandbox
# ──────────────────────────────────────────────

def _exec_worker(code: str, result_queue: multiprocessing.Queue) -> None:
    """Run in a separate process to isolate crashes and enforce timeout."""
    buf = io.StringIO()
    allowed_globals = {
        "__builtins__": {
            "print": print,
            "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "abs": abs, "min": min, "max": max, "sum": sum, "sorted": sorted,
            "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
            "round": round, "divmod": divmod, "pow": pow, "hex": hex,
            "bin": bin, "oct": oct, "bool": bool, "type": type,
            "__import__": __import__,
        }
    }
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, allowed_globals)  # nosec - isolated process
        result_queue.put(("ok", buf.getvalue()[:2000]))
    except Exception as e:
        result_queue.put(("err", f"{type(e).__name__}: {e}"))


def safe_exec(code: str, timeout: float = MAX_CODE_EXEC_SECONDS) -> str:
    """Execute code in a sandboxed subprocess, return stdout or error string."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_exec_worker, args=(code, q), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return "[Execution timed out]"
    if not q.empty():
        status, output = q.get()
        return output if status == "ok" else f"[Error] {output}"
    return "[No output]"


# ──────────────────────────────────────────────
# 3. Answer extraction
# ──────────────────────────────────────────────

def extract_answer(text: str) -> Optional[int]:
    """
    Parse final integer answer from model output.
    Priority: \\boxed{N} → last bare integer → None
    """
    # 1) \boxed{...}
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    for raw in reversed(boxed_matches):
        raw = raw.strip().replace(",", "")
        # Handle expressions like \boxed{336}
        try:
            val = int(raw)
            return val % MOD
        except ValueError:
            # Try to eval simple expressions like 3*5+2
            try:
                val = int(eval(raw, {"__builtins__": {}}, {}))  # nosec
                return val % MOD
            except Exception:
                pass

    # 2) Last integer on a line starting with "Answer:" or "= N"
    for pat in [
        r"[Aa]nswer[:\s]+(-?\d+)",
        r"=\s*(-?\d+)\s*$",
        r"(?:therefore|thus|so)[,\s]+.*?(-?\d+)\s*$",
    ]:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            try:
                return int(m.group(1)) % MOD
            except ValueError:
                pass

    # 3) Last standalone integer in the text
    ints = re.findall(r"\b(\d{1,6})\b", text)
    if ints:
        return int(ints[-1]) % MOD

    return None


# ──────────────────────────────────────────────
# 4. TIR (Tool-Integrated Reasoning) engine
# ──────────────────────────────────────────────

def run_tir_sample(
    llm,
    sampling_params_fn,
    problem_text: str,
    deadline: float,
    topic: str = "unknown",
) -> tuple[Optional[int], float]:
    """
    Single TIR sample — 以时间预算为唯一截止条件，不限轮次。
    上下文工程：
      - 题型识别 → 针对性 strategy hint + few-shot 示例
      - 多轮 TIR：model 生成 → python 沙盒执行 → 结果追回对话
      - 超 context 时自动裁剪旧的 tool output，避免截断崩溃
    Returns:
      (answer, weight)
      - answer: extracted integer answer or None
      - weight: confidence-like vote weight for weighted majority voting
    """
    tokenizer = llm.get_tokenizer()

    # 构建带 few-shot + strategy hint 的完整初始 messages
    messages = build_messages(problem_text, topic)

    full_response = ""
    round_idx = 0
    executed_tool_rounds = 0

    while True:
        if time.time() >= deadline:
            print(f"[TIR] deadline at round {round_idx}, extracting best answer so far")
            break

        # 超长保护：裁剪旧的 tool output，避免超出 max_model_len
        messages = _trim_messages_to_fit(messages, tokenizer, max_context_tokens=3500)

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = llm.generate([prompt], sampling_params_fn())
        text = outputs[0].outputs[0].text
        full_response += text
        messages.append({"role": "assistant", "content": text})
        round_idx += 1

        code_blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
        has_answer = "\\boxed{" in full_response

        if not code_blocks:
            # 模型没有生成代码，认为推理完成
            break

        # 执行最后一个代码块并把结果反馈回去
        code = code_blocks[-1].strip()
        exec_output = safe_exec(code, timeout=MAX_CODE_EXEC_SECONDS)
        tool_msg = f"```output\n{exec_output}\n```"
        messages.append({"role": "user", "content": tool_msg})
        executed_tool_rounds += 1

        # 已有答案 + 代码执行完毕 → 可以停了
        if has_answer:
            break

    answer = extract_answer(full_response)
    has_boxed = "\\boxed{" in full_response
    # Weighted vote signal:
    # - base 1.0
    # - +0.35 if explicit boxed final answer exists
    # - +0.10 per tool round, capped at +0.30
    weight = 1.0
    if has_boxed:
        weight += 0.35
    weight += min(0.30, 0.10 * executed_tool_rounds)
    return answer, weight


# ──────────────────────────────────────────────
# 5. Majority voting
# ──────────────────────────────────────────────

def majority_vote(sample_results: list[tuple[Optional[int], float]]) -> int:
    """
    Weighted majority vote.
    Tie-break order: higher weighted score -> higher raw count -> smaller answer.
    """
    weighted_scores: dict[int, float] = {}
    raw_counts: Counter[int] = Counter()
    for ans, w in sample_results:
        if ans is None:
            continue
        raw_counts[ans] += 1
        weighted_scores[ans] = weighted_scores.get(ans, 0.0) + float(w)

    if not weighted_scores:
        return 0

    # max by (weighted_score, raw_count, -answer) to keep deterministic tie break
    best_answer = max(
        weighted_scores.keys(),
        key=lambda a: (weighted_scores[a], raw_counts[a], -a),
    )
    return int(best_answer)


# ──────────────────────────────────────────────
# 6. Main solver (called per problem)
# ──────────────────────────────────────────────

class Solver:
    def __init__(
        self,
        model_path: str,
        n_samples: int = DEFAULT_N_SAMPLES,
        dry_run: bool = False,
    ):
        self.n_samples = n_samples
        self.dry_run = dry_run
        self.llm = None
        self.model_path = model_path
        # 每题时间预算 = (总时间 - 启动预留) / 题目数，再除以采样数
        # 即每个 sample 独立分到一份时间，各自有独立 deadline
        self.sample_budget_seconds = (
            TOTAL_BUDGET_SECONDS - STARTUP_RESERVE_SECONDS
        ) / N_PROBLEMS / n_samples

    def load_model(self) -> None:
        if self.dry_run:
            print("[DRY-RUN] skipping model load")
            return
        try:
            from vllm import LLM  # noqa: lazy import
        except ImportError:
            raise ImportError(
                "vLLM is not installed. "
                "In Kaggle notebook install first:\n"
                "  !pip install vllm -q"
            )

        import torch
        n_gpus = torch.cuda.device_count()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU count: {n_gpus}, VRAM per GPU: {vram_gb:.1f} GB")
        print(f"[INFO] Loading model: {self.model_path}")

        # 自动检测是否为量化模型（AWQ/GPTQ），根据配置选参数
        config_path = os.path.join(self.model_path, "config.json")
        is_quantized = False
        quant_type = None
        if os.path.exists(config_path):
            import json
            with open(config_path) as f:
                cfg = json.load(f)
            quant_type = cfg.get("quantization_config", {}).get("quant_type") or \
                         cfg.get("quantization_config", {}).get("quant_method")
            is_quantized = quant_type is not None
            if is_quantized:
                print(f"[INFO] Detected quantization: {quant_type}")

        # tensor_parallel_size：量化模型单卡就够，大模型多卡
        total_vram = vram_gb * n_gpus
        model_path_lower = self.model_path.lower()
        is_large = any(tag in model_path_lower for tag in ["72b", "70b", "120b"])

        if is_quantized or not is_large:
            tp = 1   # 量化模型 / 小模型，单卡
        else:
            # 非量化大模型：按显存估算需要几张卡
            # 72B BF16 ≈ 144GB，每张 H100 80GB → 需要 2 张
            tp = min(n_gpus, max(1, int(math.ceil(144 / vram_gb))))
            tp = min(tp, n_gpus)

        print(f"[INFO] tensor_parallel_size: {tp}, quantized: {is_quantized}")

        # max_model_len: never exceed model config limit.
        # For this AWQ model, max_position_embeddings is 4096.
        cfg_max_ctx = cfg.get("max_position_embeddings") if os.path.exists(config_path) else None
        if isinstance(cfg_max_ctx, int) and cfg_max_ctx > 0:
            max_ctx = cfg_max_ctx
        else:
            max_ctx = 4096

        # AWQ/GPTQ quantized models on vLLM require float16 dtype.
        dtype_for_model = "float16" if is_quantized else "auto"

        load_kwargs = dict(
            model=self.model_path,
            tensor_parallel_size=tp,
            dtype=dtype_for_model,
            trust_remote_code=True,
            max_model_len=max_ctx,
            gpu_memory_utilization=0.92,
        )
        # 显式指定 quantization 参数（部分量化模型需要）
        if quant_type and quant_type.lower() in ("awq", "awq_marlin"):
            load_kwargs["quantization"] = "awq"
        elif quant_type and quant_type.lower() in ("gptq", "gptq_marlin"):
            load_kwargs["quantization"] = "gptq"

        self.llm = LLM(**load_kwargs)
        print("[INFO] Model loaded successfully.")

    def _sampling_params(self):
        from vllm import SamplingParams  # noqa: lazy import
        return SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=MAX_NEW_TOKENS,   # 极大上限，让模型算完整，靠时间预算截止
        )

    def solve(self, problem_text: str) -> int:
        if self.dry_run or self.llm is None:
            return 0

        topic = classify_problem(problem_text)
        print(f"[SOLVER] detected topic: {topic}")

        # 每道题的总截止时间
        problem_deadline = time.time() + self.sample_budget_seconds * self.n_samples
        sample_results: list[tuple[Optional[int], float]] = []
        weighted_scores: dict[int, float] = {}
        raw_counts: Counter[int] = Counter()

        for i in range(self.n_samples):
            remaining = problem_deadline - time.time()
            if remaining <= 0:
                print(f"[WARN] Problem time budget exhausted after {i} samples")
                break
            # 动态分配：剩余时间 / 剩余 sample 数，让每个 sample 都有机会跑完
            sample_deadline = time.time() + remaining / (self.n_samples - i)
            try:
                ans, weight = run_tir_sample(
                    self.llm,
                    self._sampling_params,
                    problem_text,
                    deadline=sample_deadline,
                    topic=topic,
                )
                sample_results.append((ans, weight))
                if ans is not None:
                    raw_counts[ans] += 1
                    weighted_scores[ans] = weighted_scores.get(ans, 0.0) + weight
                print(f"[SOLVER] sample {i}: answer={ans}, weight={weight:.2f}")

                # Early stop 1: strict raw majority already achieved
                if ans is not None and raw_counts[ans] > self.n_samples // 2:
                    print(f"[SOLVER] early-stop: raw majority reached by answer={ans}")
                    break

                # Early stop 2: weighted lead cannot be overtaken
                remaining_samples = self.n_samples - (i + 1)
                if weighted_scores and remaining_samples > 0:
                    ranked = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
                    best_ans, best_score = ranked[0]
                    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
                    max_second_possible = second_score + remaining_samples * MAX_SINGLE_SAMPLE_WEIGHT
                    if best_score > max_second_possible:
                        print(
                            "[SOLVER] early-stop: weighted winner locked "
                            f"(best={best_ans}, score={best_score:.2f}, "
                            f"max_second={max_second_possible:.2f})"
                        )
                        break
            except Exception as e:
                print(f"[WARN] Sample {i} failed: {e}")
                sample_results.append((None, 0.0))

        result = majority_vote(sample_results)
        print(f"[SOLVER] final vote: {sample_results} → {result}")
        return result


# ──────────────────────────────────────────────
# 7. predict() — Kaggle gateway interface
# ──────────────────────────────────────────────

def _to_problem_list(problem_input) -> list[str]:
    if isinstance(problem_input, pl.Series):
        return [str(x) for x in problem_input.to_list()]
    if isinstance(problem_input, pl.DataFrame):
        col = "problem" if "problem" in problem_input.columns else problem_input.columns[0]
        return [str(x) for x in problem_input[col].to_list()]
    if isinstance(problem_input, list):
        return [str(x) for x in problem_input]
    return [str(problem_input)]


def make_predict(solver: Solver):
    def predict(id_, problem):
        problems = _to_problem_list(problem)
        answers = []
        for p in problems:
            t0 = time.time()
            ans = solver.solve(p)
            elapsed = time.time() - t0
            print(f"[PREDICT] answer={ans}, time={elapsed:.1f}s")
            answers.append(int(ans) % MOD)
        return pl.DataFrame({"answer": answers}, schema={"answer": pl.Int64})
    return predict


# ──────────────────────────────────────────────
# 8. Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help="Majority-vote samples per problem (default 8).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip model load; return 0 for all answers (test the pipeline).")
    # parse_known_args: ignore Jupyter/Colab kernel args like -f kernel.json
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[INFO] Ignoring notebook runtime args: {unknown}")

    # ── locate competition evaluation package ──
    comp_root, eval_dir = find_kaggle_eval_paths()
    if not eval_dir or not comp_root:
        raise FileNotFoundError(
            "Cannot find aimo_3_inference_server.py.\n"
            "Attach the competition input: ai-mathematical-olympiad-progress-prize-3"
        )
    AIMO3InferenceServer = import_inference_server(eval_dir, comp_root)
    print(f"[INFO] comp_root : {comp_root}")
    print(f"[INFO] eval_dir  : {eval_dir}")

    # ── locate model ──
    model_path = "" if args.dry_run else find_model_path()

    # ── build solver & load model ──
    solver = Solver(
        model_path=model_path,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
    )
    solver.load_model()

    # ── register predict and start server ──
    predict = make_predict(solver)
    inference_server = AIMO3InferenceServer(predict)

    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        # Official rerun: serve() blocks until all predictions done
        inference_server.serve()
        return

    # ── local / public debug ──
    test_path = resolve_test_path(args.test_path, comp_root)
    print(f"[INFO] test path : {test_path}")
    inference_server.run_local_gateway(data_paths=(test_path,))

    if os.path.exists("submission.parquet"):
        sub = pl.read_parquet("submission.parquet")
        print(sub)
        sub.write_csv("submission.csv")
        print("[INFO] saved submission.csv")
    else:
        print("[WARN] submission.parquet not found")


def _ensure_vllm() -> None:
    """Auto-install vLLM and enforce protobuf compatibility."""
    import subprocess
    try:
        import vllm  # noqa
    except ImportError:
        print("[INFO] vLLM not found, installing... (takes ~2 min)")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "vllm", "-q"],
            check=True,
        )
        print("[INFO] vLLM installed successfully.")
    # Kaggle evaluation's grpc stubs are incompatible with protobuf 6.x.
    # Pin protobuf to 5.x to avoid:
    # AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
    try:
        import google.protobuf  # noqa
        from google.protobuf import __version__ as pb_ver
        major = int(pb_ver.split(".")[0])
    except Exception:
        major = 0
    if major >= 6:
        print(f"[INFO] protobuf {major}.x detected, downgrading to 5.x for kaggle_evaluation compatibility...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "protobuf>=5.26.1,<6", "-q"],
            check=True,
        )
        print("[INFO] protobuf downgraded to 5.x.")


def _patch_protobuf_compat() -> None:
    """
    vLLM 安装会把 protobuf 升级到 6.x，导致 kaggle_evaluation 的 grpc 生成代码
    调用 MessageFactory.GetPrototype 时报 AttributeError（该方法在 4.x+ 被移除）。
    这里打一个向后兼容的 monkey patch，用新 API GetMessageClass 代替。
    必须在 import kaggle_evaluation 之前调用。
    """
    try:
        import google.protobuf.message_factory as _mf
        import google.protobuf.symbol_database as _sdb
        if not hasattr(_mf.MessageFactory, "GetPrototype"):
            def _get_prototype(self, descriptor):
                return _mf.GetMessageClass(descriptor)
            _mf.MessageFactory.GetPrototype = _get_prototype
        if not hasattr(_sdb.SymbolDatabase, "GetPrototype"):
            def _symdb_get_prototype(self, descriptor):
                return _mf.GetMessageClass(descriptor)
            _sdb.SymbolDatabase.GetPrototype = _symdb_get_prototype
        print("[INFO] protobuf compat patch applied (GetPrototype → GetMessageClass)")
    except Exception as e:
        print(f"[WARN] protobuf patch failed (may be OK): {e}")


# ── Auto-run: works both as notebook cell and as CLI script ──
# When pasted into a Kaggle notebook cell and Run All is clicked,
# __name__ is NOT "__main__", so we call main() unconditionally here.
_ensure_vllm()
_patch_protobuf_compat()
main()
