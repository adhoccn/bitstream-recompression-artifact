#!/usr/bin/env python
r"""Deterministic four-pass build for the manuscripts of this project.

BibTeX MUST run between the pdflatex passes. With a pdflatex-only build the
``.bbl`` is never written and every ``\cite`` silently renders as ``[?]``; fonts
and page count can also be wrong, so page limits cannot be checked from such a
build. This script fixes the order and asserts the outcome:

    pdflatex -> bibtex -> pdflatex -> pdflatex

then requires, for the requested stem,
    * zero LaTeX errors,
    * zero undefined citations and zero undefined references,
    * a non-empty ``.bbl``,
and prints the page count read from the log.

It is archived so that a reader can reproduce the deliverables exactly rather
than guessing the pass order. It is deliberately dependency-free: only a TeX
distribution with ``pdflatex``/``bibtex`` on PATH is assumed.

Usage
-----
    python scripts/build.py                       # the combined manuscript
    python scripts/build.py --stem paper_theory_part1
    python scripts/build.py --stem paper_theory_part2
    python scripts/build.py --stem supplement_part1
    python scripts/build.py --zh                  # Chinese mirror (xelatex)
    python scripts/build.py --check-only          # audit existing PDF/log
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_STEM = "paper_companion_theory"


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def which(name: str) -> bool:
    rc, _ = run([name, "--version"])
    return rc == 0


def build_pass(engine: str, stem: str) -> tuple[int, str]:
    """One non-interactive pass. Never use -halt-on-error: we need the log."""
    return run([engine, "-interaction=nonstopmode", stem + ".tex"])


def check_log(stem: str) -> list[str]:
    """Return a list of problems found in the most recent log."""
    path = os.path.join(ROOT, stem + ".log")
    if not os.path.exists(path):
        return ["no log produced for %s" % stem]
    with open(path, encoding="utf-8", errors="replace") as f:
        log = f.read()
    problems = []
    n_undef_cite = len(re.findall(r"undefined citation", log, re.I))
    n_undef_ref = len(re.findall(r"undefined reference", log, re.I))
    if n_undef_cite:
        problems.append("%d undefined citation(s)" % n_undef_cite)
    if n_undef_ref:
        problems.append("%d undefined reference(s)" % n_undef_ref)
    bbl = os.path.join(ROOT, stem + ".bbl")
    if not os.path.exists(bbl) or os.path.getsize(bbl) == 0:
        problems.append("empty or missing %s.bbl (was bibtex run?)" % stem)
    return problems


def report_pages(stem: str) -> None:
    """Print the page count as reported by TeX itself."""
    path = os.path.join(ROOT, stem + ".log")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Output written on (\S+) \((\d+) pages", line)
            if m:
                print("  pages: %s -> %s" % (m.group(1), m.group(2)))
                return


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default=DEFAULT_STEM,
                    help="manuscript stem without extension "
                         "(default: %s)" % DEFAULT_STEM)
    ap.add_argument("--zh", action="store_true",
                    help="Chinese mirror, built with xelatex")
    ap.add_argument("--check-only", action="store_true",
                    help="do not build; audit the existing log/PDF")
    args = ap.parse_args(argv)

    stem = (args.stem + "_zh") if args.zh else args.stem
    engine = "xelatex" if args.zh else "pdflatex"

    print("stem   : %s" % stem)
    print("engine : %s" % engine)
    print("root   : %s" % ROOT)

    if not which(engine):
        print("ERROR: %s not found on PATH" % engine)
        return 2

    if args.check_only:
        problems = check_log(stem)
        report_pages(stem)
        for p in problems:
            print("FAIL: %s" % p)
        print("OK: no problems" if not problems else "FAILED")
        return 1 if problems else 0

    # ── the four passes, in the only order that works ──────────────────────
    # 1st pass: writes .aux (and collects citation keys)
    rc, _ = build_pass(engine, stem)
    if rc != 0:
        print("ERROR: first %s pass failed (rc=%d)" % (engine, rc))
        check_log(stem)
        return 1
    print("  pass 1/4 ok")

    # bibtex pass: turns .aux keys into a .bbl. Skipping this is exactly what
    # leaves every citation unresolved.
    rc, out = run(["bibtex", stem])
    if rc != 0:
        print("ERROR: bibtex failed (rc=%d)" % rc)
        print(out[-800:])
        return 1
    print("  pass 2/4 bibtex ok")

    # 2nd and 3rd latex passes: resolve references and settle page breaking.
    for i in (3, 4):
        rc, _ = build_pass(engine, stem)
        if rc != 0:
            print("ERROR: %s pass %d failed (rc=%d)" % (engine, i, rc))
            return 1
        print("  pass %d/4 ok" % i)

    problems = check_log(stem)
    report_pages(stem)
    if problems:
        for p in problems:
            print("FAIL: %s" % p)
        return 1
    print("OK: %s built cleanly" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
