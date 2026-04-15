"""
Top-level orchestrator for Homework_2026.

Runs every exercise sub-package's ``main`` module sequentially and then
compiles the LaTeX report under ``report/`` to produce ``report/main.pdf``.

Usage
-----
    python -m main                        # from inside Homework_2026/
    python main.py                        # equivalent
    python main.py --skip-exercises       # only compile the LaTeX report
    python main.py --skip-latex           # only run the Python exercises
    python main.py --only 1 3 5           # run a subset of exercises
    python main.py --latex-engine xelatex # change the TeX engine

Exit code is 0 on full success, 1 if any stage fails (the script keeps going
after a failure, then reports a summary).
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Repository root = the folder this file lives in.
ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "report"
REPORT_TEX = "main.tex"
REPORT_PDF = "main.pdf"

EXERCISES = (1, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def _banner(title: str, char: str = "=") -> None:
    bar = char * 78
    print(f"\n{bar}\n{title}\n{bar}", flush=True)


# ---------------------------------------------------------------------------
# Run a single exercise
# ---------------------------------------------------------------------------
def run_exercise(n: int) -> tuple[bool, float, str]:
    """Import and execute ``exerciseN.main.main()``.

    Returns
    -------
    (success, elapsed_seconds, message)
    """
    module_name = f"exercise{n}.main"
    _banner(f"Exercise {n}  ({module_name})")
    t0 = time.perf_counter()
    try:
        # Make sure ROOT is on sys.path so the package import works regardless
        # of the directory the script was launched from.
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))

        mod = importlib.import_module(module_name)
        if hasattr(mod, "main") and callable(mod.main):
            mod.main()
        # If the module runs everything at import time it's still fine.
    except Exception as exc:                                # noqa: BLE001
        elapsed = time.perf_counter() - t0
        msg = f"FAILED ({type(exc).__name__}: {exc})"
        print(f"\n[exercise{n}] {msg}", file=sys.stderr, flush=True)
        return False, elapsed, msg

    elapsed = time.perf_counter() - t0
    return True, elapsed, "ok"


# ---------------------------------------------------------------------------
# Compile the LaTeX report
# ---------------------------------------------------------------------------
def compile_latex(engine: str = "pdflatex", passes: int = 2) -> tuple[bool, float, str]:
    """Compile ``report/main.tex`` to ``report/main.pdf``.

    Two passes are run by default so cross-references and the table of contents
    stabilise. Output is captured and only printed on failure (or in verbose
    mode if the user re-runs manually).
    """
    _banner(f"LaTeX report  ({engine}, {passes} passes)")
    t0 = time.perf_counter()

    if shutil.which(engine) is None:
        msg = f"{engine!r} not found in PATH"
        print(f"[latex] {msg}", file=sys.stderr, flush=True)
        return False, time.perf_counter() - t0, msg

    if not (REPORT_DIR / REPORT_TEX).exists():
        msg = f"{REPORT_DIR / REPORT_TEX} does not exist"
        print(f"[latex] {msg}", file=sys.stderr, flush=True)
        return False, time.perf_counter() - t0, msg

    cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", REPORT_TEX]
    last_stdout = ""
    for k in range(1, passes + 1):
        print(f"[latex] pass {k}/{passes}: {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=REPORT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )
        last_stdout = proc.stdout
        if proc.returncode != 0:
            sys.stderr.write(proc.stdout[-4000:])
            sys.stderr.write(proc.stderr[-2000:])
            return (
                False,
                time.perf_counter() - t0,
                f"{engine} exited with code {proc.returncode}",
            )

    elapsed = time.perf_counter() - t0
    pdf_path = REPORT_DIR / REPORT_PDF
    pdf_size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0.0
    print(
        f"[latex] OK  -> {pdf_path}  ({pdf_size_kb:,.0f} KB)\n"
        f"[latex] last lines:\n{last_stdout.splitlines()[-3:]}",
        flush=True,
    )
    return True, elapsed, "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run every PS2 exercise and compile the LaTeX report.",
    )
    p.add_argument(
        "--only",
        nargs="+",
        type=int,
        metavar="N",
        choices=EXERCISES,
        help="Run only this subset of exercises (e.g. --only 1 3).",
    )
    p.add_argument(
        "--skip-exercises",
        action="store_true",
        help="Do not run any Python exercise; only compile the LaTeX report.",
    )
    p.add_argument(
        "--skip-latex",
        action="store_true",
        help="Do not compile the LaTeX report; only run the Python exercises.",
    )
    p.add_argument(
        "--latex-engine",
        default="pdflatex",
        help="LaTeX engine to use (default: pdflatex; e.g. xelatex, lualatex).",
    )
    p.add_argument(
        "--latex-passes",
        type=int,
        default=2,
        help="Number of LaTeX passes (default: 2 to settle cross-refs).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exercises = tuple(args.only) if args.only else EXERCISES

    summary: list[tuple[str, bool, float, str]] = []

    if not args.skip_exercises:
        for n in exercises:
            ok, dt, msg = run_exercise(n)
            summary.append((f"exercise{n}", ok, dt, msg))
    else:
        print("[main] --skip-exercises set: skipping Python stages.", flush=True)

    if not args.skip_latex:
        ok, dt, msg = compile_latex(
            engine=args.latex_engine,
            passes=args.latex_passes,
        )
        summary.append(("latex", ok, dt, msg))
    else:
        print("[main] --skip-latex set: skipping LaTeX compilation.", flush=True)

    # ---------- Summary ----------
    _banner("Run summary")
    width = max((len(name) for name, *_ in summary), default=0)
    all_ok = True
    for name, ok, dt, msg in summary:
        flag = "OK " if ok else "ERR"
        all_ok &= ok
        print(f"  [{flag}] {name.ljust(width)}   {dt:7.2f} s   {msg}")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
