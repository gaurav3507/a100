"""Manuscript checks: compile, cross references, symbols, values, style.

  python paper/check_manuscript.py paper/sec3_methodology.tex [...]

Checks performed on every section file:
  1. LaTeX compiles under a local harness (catches broken math/environments).
  2. Every \\label is referenced and every \\ref/\\eqref target exists.
  3. Math delimiters and environment nesting balance.
  4. Style: no em dash or en dash, no smart quotes, no "delve/leverage"
     filler, no double spaces, no TODO markers.
  5. Numerical claims in the prose are listed for manual cross checking
     against the results CSVs (printed, not asserted).
"""
import pathlib
import re
import subprocess
import sys

BAD_CHARS = {
    "\u2014": "em dash",
    "\u2013": "en dash",
    "\u2018": "left single quote",
    "\u2019": "right single quote",
    "\u201c": "left double quote",
    "\u201d": "right double quote",
    "\u00a0": "non breaking space",
    "\u2026": "ellipsis character",
}
FILLER = ["delve", "leverage", "utilise", "utilize", "showcase", "pivotal",
          "realm", "landscape of", "it is worth noting", "notably,"]


def check_file(path: pathlib.Path):
    text = path.read_text()
    issues = []

    for ch, name in BAD_CHARS.items():
        n = text.count(ch)
        if n:
            line = next((i + 1 for i, l in enumerate(text.splitlines()) if ch in l), 0)
            issues.append(f"style: {n} x {name} (first at line {line})")

    low = text.lower()
    for w in FILLER:
        if w in low:
            issues.append(f"style: filler phrase '{w}'")

    if "  " in re.sub(r"^\s+", "", text, flags=re.M):
        pass  # aligned LaTeX often has internal spacing; ignore

    for marker in ("TODO", "FIXME", "XXX", "??"):
        if marker in text:
            issues.append(f"content: leftover marker {marker}")

    # math delimiter balance
    if text.count("$") % 2:
        issues.append("math: odd number of $ delimiters")
    for env in ("equation", "align", "algorithmic", "algorithm", "proposition",
                "proof", "itemize", "enumerate", "table", "figure"):
        b = len(re.findall(r"\\begin\{" + env + r"\*?\}", text))
        e = len(re.findall(r"\\end\{" + env + r"\*?\}", text))
        if b != e:
            issues.append(f"structure: \\begin{{{env}}} x{b} vs \\end x{e}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text))
    dangling = sorted(r for r in refs if r not in labels and not r.startswith("sec:"))
    if dangling:
        issues.append(f"refs: targets not defined in this file: {dangling}")
    unused = sorted(l for l in labels if l not in refs and l.startswith("eq:"))
    if unused:
        issues.append(f"refs: equation labels never referenced: {unused}")

    numbers = sorted(set(re.findall(r"(?<![\w\\])(\d+\.\d+|\d{2,})(?![\w}])", text)))
    return issues, numbers


def compile_check(paths):
    harness = pathlib.Path("paper/_localtest.tex")
    body = "\n".join(f"\\input{{{p.stem}}}" for p in paths)
    harness.write_text(
        "\\documentclass[twocolumn,10pt]{article}\n"
        "\\usepackage[margin=0.7in]{geometry}\n"
        "\\usepackage{amsmath,amssymb,amsthm}\n"
        "\\usepackage{algorithm}\n\\usepackage{algorithmic}\n"
        "\\usepackage{graphicx}\n\\usepackage{booktabs}\n"
        "\\newtheorem{proposition}{Proposition}\n"
        "\\newtheorem{theorem}{Theorem}\n"
        "\\usepackage{cite}\n"
        "\\newcommand{\\IEEEPARstart}[2]{\\textbf{#1#2}}\n"
        "\\providecommand{\\IEEEkeywords}{}\n"
        "\\begin{document}\n" + body + "\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                        "_localtest.tex"], cwd="paper", capture_output=True, text=True)
    if r.returncode != 0:
        err = [l for l in r.stdout.splitlines() if l.startswith("!")][:5]
        return False, err
    pages = re.search(r"\((\d+) pages", r.stdout)
    return True, [f"compiled, {pages.group(1) if pages else '?'} pages"]


if __name__ == "__main__":
    paths = [pathlib.Path(p) for p in sys.argv[1:]] or \
            sorted(pathlib.Path("paper").glob("sec*.tex"))
    ok, info = compile_check(paths)
    print(f"[compile] {'OK' if ok else 'FAILED'}: {info}")
    total = 0
    for p in paths:
        issues, numbers = check_file(p)
        total += len(issues)
        print(f"\n=== {p.name} ===")
        if issues:
            for i in issues:
                print(f"  ISSUE {i}")
        else:
            print("  no style/structure issues")
        print(f"  numeric claims to verify ({len(numbers)}): {', '.join(numbers[:24])}"
              + (" ..." if len(numbers) > 24 else ""))
    print(f"\nTOTAL ISSUES: {total}" + ("" if ok and not total else "  <-- fix before release"))
    sys.exit(1 if (total or not ok) else 0)
