#!/usr/bin/env python3
"""
build_revealjs.py — Generate a self-contained Reveal.js HTML slide deck
from the thesis defense outline.

Usage:
    .venv/bin/python scripts/build_revealjs.py
    .venv/bin/python scripts/build_revealjs.py --validate
    .venv/bin/python scripts/build_revealjs.py --no-vendor-check
"""
from __future__ import annotations

import argparse
import base64
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTLINE_PATH = REPO_ROOT / "docs" / "slides" / "thesis_defense_outline.md"
OUTPUT_PATH = REPO_ROOT / "docs" / "slides" / "thesis_defense.html"
PLOTS_DIR = REPO_ROOT / "results" / "report" / "plots"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
REVEALJS_DIR = VENDOR_DIR / "revealjs"
KATEX_DIR = VENDOR_DIR / "katex"

# ---------------------------------------------------------------------------
# Vendor file manifest: (local relative path, CDN URL)
# ---------------------------------------------------------------------------
VENDOR_FILES = [
    (REVEALJS_DIR / "dist" / "reveal.css",
     "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css"),
    (REVEALJS_DIR / "dist" / "theme" / "white.css",
     "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/white.css"),
    (REVEALJS_DIR / "dist" / "reveal.js",
     "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"),
    (REVEALJS_DIR / "plugin" / "math" / "math.js",
     "https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/math/math.js"),
    (KATEX_DIR / "dist" / "katex.min.css",
     "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"),
    (KATEX_DIR / "dist" / "katex.min.js",
     "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"),
    (KATEX_DIR / "dist" / "contrib" / "auto-render.min.js",
     "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"),
]

# ---------------------------------------------------------------------------
# Known image mappings: slide-id (str) → list[filename]
# (Kept in sync with build_pptx.py SLIDE_IMAGES registry)
# ---------------------------------------------------------------------------
SLIDE_IMAGES: dict[str, list[str]] = {
    "1":  ["dataset_snapshots_grid.png"],
    "5":  ["dataset_snapshots_grid.png"],
    "6":  ["edge_growth_density.png"],
    "7":  ["topology_map_2d.png"],
    "15": ["auc_comparison.png", "ap_comparison.png"],
    "16": ["ranking_heatmap.png"],
    "17": ["learning_curves_collegemsg.png", "learning_curves_mooc_actions.png"],
    "18": ["topology_map_2d_with_winners.png"],
    "19": ["beta_sensitivity.png"],
    "20": ["runtime_comparison.png"],
    "22": ["dataset_snapshots_grid.png"],
    "A2": [
        "learning_curves_bitcoinotc.png",
        "learning_curves_eut.png",
        "learning_curves_lastfm.png",
        "learning_curves_wikipedia.png",
    ],
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
class SlideSpec(NamedTuple):
    slide_id: str        # "1".."22", "A1".."A6"
    title: str           # Raw title (LaTeX $...$ preserved for KaTeX)
    bullets: list[str]   # Raw bullet text (LaTeX preserved)
    images: list[Path]   # Resolved image paths


# ---------------------------------------------------------------------------
# Parser (adapted from build_pptx.py, LaTeX NOT normalized for Reveal.js)
# ---------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^##\s+Slide\s+(.+?)(?:\s+—\s+(.+))?$")
_IMAGE_RE = re.compile(r"^-\s+(?:IMAGE:\s*|(?:\[Khung trống cho\b[^\]]*\]\s*))(.+\.png)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^-\s+(.+)$")


def _clean_bullet(text: str) -> str:
    """Minimal cleaning: strip bold markdown, NFC normalize. Keep LaTeX $...$."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = unicodedata.normalize("NFC", text)
    return text


def _clean_title(text: str) -> str:
    """Same as _clean_bullet for titles."""
    return _clean_bullet(text)


def parse_outline(path: Path) -> list[SlideSpec]:
    """Parse the markdown outline into SlideSpec objects (LaTeX preserved)."""
    slides: list[SlideSpec] = []
    current_id: str | None = None
    current_title: str = ""
    current_bullets: list[str] = []
    current_images: list[Path] = []

    def flush() -> None:
        if current_id is None:
            return
        # Use registry-based images if available, else inline-parsed
        reg_images = SLIDE_IMAGES.get(current_id, [])
        resolved: list[Path] = []
        if reg_images:
            for fname in reg_images:
                p = PLOTS_DIR / fname
                if p.exists():
                    resolved.append(p)
        else:
            resolved = list(current_images)

        slides.append(SlideSpec(
            slide_id=current_id,
            title=_clean_title(current_title),
            bullets=[_clean_bullet(b) for b in current_bullets],
            images=resolved,
        ))

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if not line or line.startswith("---"):
            continue

        m = _HEADING_RE.match(line)
        if m:
            flush()
            raw_id_part = m.group(1).strip()
            raw_title_part = m.group(2).strip() if m.group(2) else raw_id_part
            current_id = raw_id_part.split()[0]
            current_title = raw_title_part
            current_bullets = []
            current_images = []
            continue

        if current_id is None:
            continue

        # IMAGE lines
        img_m = _IMAGE_RE.match(line)
        if img_m:
            img_path_str = img_m.group(1).strip()
            p = (REPO_ROOT / img_path_str) if "/" in img_path_str else (PLOTS_DIR / img_path_str)
            p = p.resolve()
            if p.exists():
                current_images.append(p)
            continue

        # Regular bullets
        bul_m = _BULLET_RE.match(line)
        if bul_m:
            text = bul_m.group(1).strip()
            if re.match(r"\[Khung trống", text):
                continue
            current_bullets.append(text)

    flush()
    return slides


# ---------------------------------------------------------------------------
# Vendor management
# ---------------------------------------------------------------------------

def ensure_vendor(skip: bool = False) -> None:
    """Download missing vendor files if skip=False."""
    if skip:
        print("  --no-vendor-check: skipping vendor download.")
        return

    missing = [entry for entry in VENDOR_FILES if not entry[0].exists()]
    if not missing:
        print("  Vendor files present.")
        return

    print(f"  Downloading {len(missing)} missing vendor file(s)...")
    for local_path, url in missing:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"    curl → {local_path.name} from {url}")
        result = subprocess.run(
            ["curl", "-L", "-s", "-o", str(local_path), url],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    ERROR downloading {url}: {result.stderr}", file=sys.stderr)
            sys.exit(1)
        size = local_path.stat().st_size
        if size < 100:
            print(f"    ERROR: {local_path.name} is suspiciously small ({size} bytes)", file=sys.stderr)
            sys.exit(1)
        print(f"    OK  {local_path.name} ({size:,} bytes)")


def read_vendor(local_path: Path) -> str:
    """Read a vendor file as text (UTF-8)."""
    return local_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(img_path: Path) -> str:
    """Return a data URI for the given PNG."""
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# HTML escaping
# ---------------------------------------------------------------------------

def he(text: str) -> str:
    """Minimal HTML escape — only &, <, > (not quotes, since we use tag content not attributes)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Slide HTML builders
# ---------------------------------------------------------------------------

def _bullet_html(bullets: list[str]) -> str:
    """Build a <ul> from bullet list."""
    if not bullets:
        return ""
    items = "\n".join(f"      <li>{he(b)}</li>" for b in bullets)
    return f"    <ul>\n{items}\n    </ul>"


def _img_tag(img_path: Path, css_class: str = "plot") -> str:
    uri = encode_image(img_path)
    alt = he(img_path.stem.replace("_", " "))
    return f'<img src="{uri}" alt="{alt}" class="{css_class}">'


def build_title_section(spec: SlideSpec) -> str:
    """Slide 1: centered title layout with subtitle bullets."""
    subtitle_lines = spec.bullets[:4]  # author, supervisor, date, etc.
    subtitle_html = ""
    if subtitle_lines:
        items = "\n".join(f"      <li>{he(b)}</li>" for b in subtitle_lines)
        subtitle_html = f"""
    <ul class="subtitle-list">
{items}
    </ul>"""

    img_html = ""
    if spec.images:
        uri = encode_image(spec.images[0])
        alt = he(spec.images[0].stem)
        img_html = f'\n    <img src="{uri}" alt="{alt}" class="title-bg">'

    return f"""  <section class="title-slide" data-slide-id="{spec.slide_id}">
    <h1>{he(spec.title)}</h1>
{subtitle_html}{img_html}
  </section>"""


def build_content_section(spec: SlideSpec) -> str:
    """Standard content slide."""
    images = spec.images
    bullets = spec.bullets
    n_imgs = len(images)

    body_parts: list[str] = []

    # Appendix marker
    is_appendix = spec.slide_id.startswith("A")
    appendix_marker = ""
    if is_appendix:
        appendix_marker = '\n    <span class="appendix-marker">Phụ lục</span>'

    # Content layout
    if n_imgs == 0:
        # Text only
        body_parts.append(_bullet_html(bullets))
    elif n_imgs == 1:
        if bullets:
            # Split: bullets left, image right
            body_parts.append(f'    <div class="split-layout">')
            body_parts.append(f'      <div class="split-text">')
            body_parts.append(_bullet_html(bullets))
            body_parts.append(f'      </div>')
            body_parts.append(f'      <div class="split-img">')
            body_parts.append(f'        {_img_tag(images[0], "plot")}')
            body_parts.append(f'      </div>')
            body_parts.append(f'    </div>')
        else:
            # Image only, centered
            body_parts.append(f'    {_img_tag(images[0], "plot")}')
    elif n_imgs == 2:
        if bullets:
            # Bullets above, 2-image grid below
            body_parts.append(_bullet_html(bullets))
            body_parts.append(f'    <div class="image-grid">')
            for img in images:
                body_parts.append(f'      {_img_tag(img, "")}')
            body_parts.append(f'    </div>')
        else:
            # 2-image grid only
            body_parts.append(f'    <div class="image-grid">')
            for img in images:
                body_parts.append(f'      {_img_tag(img, "")}')
            body_parts.append(f'    </div>')
    elif n_imgs >= 3:
        # Grid (2×2 for 4, etc.) — no bullets expected
        grid_class = "image-grid image-grid-2x2" if n_imgs == 4 else "image-grid"
        if bullets:
            body_parts.append(_bullet_html(bullets))
        body_parts.append(f'    <div class="{grid_class}">')
        for img in images:
            body_parts.append(f'      {_img_tag(img, "")}')
        body_parts.append(f'    </div>')

    body_html = "\n".join(body_parts)

    return f"""  <section data-slide-id="{spec.slide_id}">{appendix_marker}
    <h2>{he(spec.title)}</h2>
{body_html}
  </section>"""


def build_sections(slides: list[SlideSpec]) -> str:
    """Build all <section> elements."""
    parts: list[str] = []
    for i, spec in enumerate(slides):
        if i == 0:
            parts.append(build_title_section(spec))
        else:
            parts.append(build_content_section(spec))
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CUSTOM_CSS = """\
/* ── Academic tweaks ─────────────────────────────── */
.reveal section {
  text-align: left;
  padding: 0 2em;
}
.reveal .title-slide {
  text-align: center;
  padding-top: 3em;
}
.reveal .title-slide h1 {
  font-size: 1.7em;
  color: #1F3564;
  margin-bottom: 0.6em;
  line-height: 1.3;
}
.reveal .title-slide .subtitle-list {
  list-style: none;
  margin: 0;
  padding: 0;
  font-size: 0.9em;
  color: #444;
  line-height: 1.8;
}
.reveal .title-slide .title-bg {
  max-height: 28vh;
  margin-top: 1em;
  opacity: 0.55;
  display: block;
  margin-left: auto;
  margin-right: auto;
}
.reveal h2 {
  font-size: 1.4em;
  color: #1F3564;
  margin-bottom: 0.5em;
  border-bottom: 3px solid #2672B6;
  padding-bottom: 0.2em;
}
.reveal ul {
  margin: 0.3em 0 0.3em 1.2em;
}
.reveal ul li {
  margin: 0.35em 0;
  font-size: 0.85em;
  line-height: 1.5;
}
.reveal img.plot {
  max-height: 60vh;
  max-width: 92%;
  display: block;
  margin: 0.5em auto 0;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.reveal .image-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.6em;
  margin-top: 0.4em;
}
.reveal .image-grid-2x2 {
  grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
}
.reveal .image-grid img {
  width: 100%;
  max-height: 42vh;
  object-fit: contain;
  border-radius: 3px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.12);
}
.reveal .split-layout {
  display: grid;
  grid-template-columns: 45% 55%;
  gap: 1em;
  align-items: start;
}
.reveal .split-text ul {
  margin-top: 0;
}
.reveal .split-img img.plot {
  margin: 0;
  max-height: 55vh;
  max-width: 100%;
}
.reveal .appendix-marker {
  color: #aaa;
  font-size: 0.65em;
  font-style: italic;
  float: right;
  margin-top: 0.2em;
}
/* Slide number */
.reveal .slide-number {
  font-size: 14px;
  color: #999;
  background-color: transparent;
}
"""


def build_html(slides: list[SlideSpec]) -> str:
    """Assemble the full self-contained HTML file."""
    # Vendor assets
    reveal_css = read_vendor(REVEALJS_DIR / "dist" / "reveal.css")
    white_css = read_vendor(REVEALJS_DIR / "dist" / "theme" / "white.css")
    katex_css = read_vendor(KATEX_DIR / "dist" / "katex.min.css")
    reveal_js = read_vendor(REVEALJS_DIR / "dist" / "reveal.js")
    math_plugin_js = read_vendor(REVEALJS_DIR / "plugin" / "math" / "math.js")
    katex_js = read_vendor(KATEX_DIR / "dist" / "katex.min.js")
    auto_render_js = read_vendor(KATEX_DIR / "dist" / "contrib" / "auto-render.min.js")

    sections_html = build_sections(slides)
    n_slides = len(slides)

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Tái hiện và phân tích so sánh GCN_MA — Bảo vệ luận văn</title>
  <!--[Reveal.js v5 — reveal.css]-->
  <style>
{reveal_css}
  </style>
  <!--[Reveal.js v5 — white theme]-->
  <style>
{white_css}
  </style>
  <!--[KaTeX 0.16.9 — katex.min.css]-->
  <style>
{katex_css}
  </style>
  <!--[Custom academic tweaks]-->
  <style>
{CUSTOM_CSS}
  </style>
</head>
<body>
  <!--
    Thesis defense slide deck — {n_slides} slides (22 main + 6 appendix)
    Built by scripts/build_revealjs.py
    PDF export: open this file and append ?print-pdf to the URL, then Ctrl+P → Save as PDF
  -->
  <div class="reveal">
    <div class="slides">

{sections_html}

    </div>
  </div>

  <!--[KaTeX 0.16.9 — katex.min.js]-->
  <script>
{katex_js}
  </script>
  <!--[KaTeX 0.16.9 — auto-render.min.js]-->
  <script>
{auto_render_js}
  </script>
  <!--[Reveal.js v5 — reveal.js]-->
  <script>
{reveal_js}
  </script>
  <!--[Reveal.js v5 — math plugin]-->
  <script>
{math_plugin_js}
  </script>

  <script>
    // The math plugin tries to load KaTeX from CDN; we patch it to use
    // the already-loaded global instead.
    (function() {{
      // KaTeX is already loaded inline above. Patch the plugin so it
      // does NOT fetch any external scripts, and calls renderMathInElement
      // directly using the already-available globals.
      const _origInit = RevealMath.KaTeX().init;
      Reveal.initialize({{
        width: 1280,
        height: 720,
        margin: 0.04,
        hash: true,
        slideNumber: 'c/t',
        controls: true,
        progress: true,
        history: true,
        center: false,
        transition: 'slide',
        backgroundTransition: 'fade',
        plugins: [],   // no math plugin — we render manually below
      }});

      // Manual KaTeX rendering after Reveal is ready
      Reveal.on('ready', function() {{
        renderMathInElement(document.querySelector('.slides'), {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$',  right: '$',  display: false}},
            {{left: '\\\\(', right: '\\\\)', display: false}},
            {{left: '\\\\[', right: '\\\\]', display: true}},
          ],
          throwOnError: false,
          ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        }});
        Reveal.layout();
      }});

      // Re-render when slide changes (lazy safety net)
      Reveal.on('slidechanged', function(event) {{
        renderMathInElement(event.currentSlide, {{
          delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '$',  right: '$',  display: false}},
          ],
          throwOnError: false,
        }});
      }});
    }})();
  </script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------

def validate(slides: list[SlideSpec]) -> None:
    print(f"Total slides: {len(slides)}\n")
    for spec in slides:
        img_note = f"  [{len(spec.images)} image(s)]" if spec.images else ""
        has_math = " [math]" if "$" in spec.title or any("$" in b for b in spec.bullets) else ""
        print(f"  Slide {spec.slide_id}: {spec.title}{img_note}{has_math}")


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_html_deck(slides: list[SlideSpec]) -> None:
    print(f"  Building HTML for {len(slides)} slides...")
    html = build_html(slides)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")

    size_bytes = OUTPUT_PATH.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Quick sanity checks
    section_count = html.count("<section")
    starts_ok = html.startswith("<!DOCTYPE html>")
    ends_ok = html.rstrip().endswith("</html>")

    print(f"\n  Saved: {OUTPUT_PATH}")
    print(f"  File size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"  <section> count: {section_count}")
    print(f"  Starts with <!DOCTYPE html>: {starts_ok}")
    print(f"  Ends with </html>: {ends_ok}")

    if section_count < 28:
        print(f"  WARNING: expected ≥28 <section> elements, got {section_count}", file=sys.stderr)
    if not (2 * 1024 * 1024 <= size_bytes <= 8 * 1024 * 1024):
        print(f"  WARNING: file size {size_mb:.2f} MB is outside expected 2–8 MB range", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build self-contained Reveal.js HTML deck from thesis_defense_outline.md"
    )
    parser.add_argument("--validate", action="store_true",
                        help="Print slide list only, do not generate HTML")
    parser.add_argument("--no-vendor-check", action="store_true",
                        help="Skip downloading vendor files (assume already present)")
    args = parser.parse_args()

    print(f"Parsing outline: {OUTLINE_PATH}")
    slides = parse_outline(OUTLINE_PATH)
    print(f"Found {len(slides)} slides\n")

    if args.validate:
        validate(slides)
        return

    print("Checking vendor files...")
    ensure_vendor(skip=args.no_vendor_check)
    print()

    build_html_deck(slides)
    print("\nDone. Open in browser with:")
    print(f"  file://{OUTPUT_PATH}")
    print("For PDF export, append ?print-pdf to the URL and use browser Print → Save as PDF.")


if __name__ == "__main__":
    main()
