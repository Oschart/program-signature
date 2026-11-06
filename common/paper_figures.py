"""Save a matplotlib figure as usual, plus a LaTeX copy in the paper repo.

Every plotting script saves through `save_figure(fig, path, **savefig_kwargs)` instead of `fig.savefig`. Besides
`path` (the PNG), it writes the figure as a pgfpicture (matplotlib's PGF backend) to

    <paper repo>/figures/generated/<path relative to this repo, suffix .tex>

e.g. plots/ablations_50pct/sig_vs_logsig.png -> figures/generated/plots/ablations_50pct/sig_vs_logsig.tex (a path
outside this repo keeps only its file name). The paper includes it with
`\\plotinput[<width>]{figures/generated/...}` (defined in neurips_2026.tex): LaTeX sets the text in the paper's fonts
and the picture is scaled to <width>, as the PNG would be, so a figure drawn at its placement width (NeurIPS
\\textwidth = 5.5 in) keeps its font sizes. A figure too heavy for pdflatex as vectors (dense scatter / path plots)
gets its data layer rasterized into PNGs next to the .tex; text and axes stay LaTeX.

The paper repo is $PROGSIG_PAPER_DIR; unset or "" (the default) skips the LaTeX copy, and a missing directory skips
it with a warning. Only the LaTeX copy needs pdflatex on PATH: matplotlib runs it to measure the text, so a label LaTeX
cannot typeset fails here, not in the paper build.
"""
import os
import re
import sys
from pathlib import Path

import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_DIR = ""
GENERATED_DIR = Path("figures/generated")

# Text is measured in the paper's fonts (neurips_2026.sty: Times roman, Helvetica sans), so layouts match the build.
PGF_RC = {
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage[T1]{fontenc}\renewcommand{\rmdefault}{ptm}\renewcommand{\sfdefault}{phv}",
}
# Plain-text characters that LaTeX treats specially or pdflatex cannot set (the PGF backend handles % and ^ itself).
TEX_PLAIN_ESCAPES = {"_": r"\_", "&": r"\&", "#": r"\#", "∞": r"\ensuremath{\infty}"}
# pdflatex's fixed main memory overflows on ~2.4 MB of PGF (24-panel latent_view figure); above this, rasterize data.
MAX_VECTOR_BYTES = 1_000_000
RASTER_DPI = 300
TEXT_WIDTH_IN = 5.5  # NeurIPS \textwidth: draw at this width so \plotinput[\textwidth]{...} keeps the font sizes


def escape_plain_text(text):
    """Escape TEX_PLAIN_ESCAPES outside $...$; math is LaTeX already and passes through."""
    parts = re.split(r"(?<!\\)\$", text)  # even indices are outside math
    parts[::2] = ["".join(TEX_PLAIN_ESCAPES.get(c, c) for c in p) for p in parts[::2]]
    return "$".join(parts)


def paper_dir():
    """The paper repo to write LaTeX copies to, or None if there is none."""
    paper = os.environ.get("PROGSIG_PAPER_DIR", DEFAULT_PAPER_DIR)
    if not paper:
        return None
    paper = Path(paper).expanduser()
    if not paper.is_dir():
        print(f"[paper_figures] WARNING: paper repo {paper} not found, no LaTeX copies (set PROGSIG_PAPER_DIR)")
        return None
    return paper


def repo_relative(path):
    """`path` relative to this repo, or just its file name if it lies outside."""
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else Path(path.name)


def write_pgf(fig, tex_path, savefig_kwargs, rasterize_data=False):
    """fig as a pgfpicture at tex_path; with rasterize_data, every line / collection / patch of every axes is a
    bitmap (one PNG per run of consecutive data artists, written next to tex_path by matplotlib)."""
    # Every string the PGF backend writes goes through backend_pgf._tex_escape (private, matplotlib 3.9), which
    # only handles the minus sign; escape plain text on top of it for this save only.
    from matplotlib.backends import backend_pgf
    tex_escape = backend_pgf._tex_escape
    backend_pgf._tex_escape = lambda s: tex_escape(escape_plain_text(s))
    data = [a for ax in fig.axes for a in (*ax.lines, *ax.collections, *ax.patches) if not a.get_rasterized()]
    for artist in data if rasterize_data else []:
        artist.set_rasterized(True)
    try:
        with mpl.rc_context(PGF_RC):
            fig.savefig(tex_path, format="pgf", **savefig_kwargs)
    finally:
        backend_pgf._tex_escape = tex_escape
        for artist in data:
            artist.set_rasterized(False)


def save_figure(fig, path, **savefig_kwargs):
    """fig.savefig(path, **savefig_kwargs), plus its LaTeX copy in the paper repo (see the module docstring)."""
    fig.savefig(path, **savefig_kwargs)
    paper = paper_dir()
    if paper is None:
        return
    tex_path = paper / GENERATED_DIR / repo_relative(path).with_suffix(".tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    for stale in tex_path.parent.glob(f"{tex_path.stem}-img*.png"):
        stale.unlink()
    write_pgf(fig, tex_path, savefig_kwargs)
    pgf = tex_path.read_text()
    if len(pgf) > MAX_VECTOR_BYTES:
        print(f"[paper_figures] {tex_path.name}: {len(pgf) / 1e6:.1f} MB as vectors, too big for pdflatex; "
              f"rasterizing its data at {RASTER_DPI} dpi")
        write_pgf(fig, tex_path, {**savefig_kwargs, "dpi": RASTER_DPI}, rasterize_data=True)
        # matplotlib references the bitmaps by bare file name; the paper builds from its root.
        img_dir = tex_path.parent.relative_to(paper).as_posix()
        pgf = re.sub(r"\]\{(%s-img\d+\.png)\}" % re.escape(tex_path.stem), rf"]{{{img_dir}/\1}}", tex_path.read_text())
        if len(pgf) > MAX_VECTOR_BYTES:
            print(f"[paper_figures] WARNING: {tex_path.name} is still {len(pgf) / 1e6:.1f} MB; pdflatex may run out "
                  "of memory on it")
    script = Path(sys.argv[0]).resolve()
    script = script.relative_to(REPO_ROOT) if script.is_relative_to(REPO_ROOT) else script.name
    cmd = " ".join(["python", str(script), *sys.argv[1:]])
    header = (f"%% Generated in the program-signature code repo by `{cmd}` (LaTeX copy of {Path(path).name}).\n"
              "%% Do not edit: rerun that command. Include with \\plotinput[<width>]{<this file>}.\n")
    tex_path.write_text(header + pgf)
    print(f"Wrote {tex_path}")
