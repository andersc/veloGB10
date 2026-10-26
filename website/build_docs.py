#!/usr/bin/env python3
"""Build the veloengine.ai documentation section from Markdown sources in content/.

Usage:  python3 build_docs.py            (from the website/ directory)

The landing page (index.html) and the docs hub (docs/index.html) are
hand-written and untouched by this script. To add or rename a docs page:
edit MANIFEST below, add/edit content/<name>.md, re-run, then link the page
from docs/index.html. sitemap.xml is regenerated on every run.
"""
import datetime
import os
import re
import markdown

ROOT = os.path.dirname(os.path.abspath(__file__))
CONTENT = os.path.join(ROOT, "content")
OUT = os.path.join(ROOT, "docs")
BASE = "https://veloengine.ai"

MANIFEST = [
    ("index", "Overview"),
    ("setup-qwen27b", "Qwen3.8-27B setup"),
    ("setup-hy3", "Hy3 setup"),
    ("cache", "TP cache management"),
    ("endurance", "Endurance report"),
    ("changelog", "Changelog"),
]

FAVICON = ('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
           '<path d="M3 4l9 16 9-16" fill="none" stroke="%235b8cff" stroke-width="2.6" '
           'stroke-linecap="round"/><path d="M8.5 4l3.5 6.2L15.5 4" fill="none" '
           'stroke="%238392ff" stroke-width="2.2" stroke-linecap="round"/></svg>')

LOGO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none">'
        '<path d="M3 4l9 16 9-16" stroke="#5b8cff" stroke-width="2.6" stroke-linecap="round" '
        'stroke-linejoin="round"/><path d="M8.5 4l3.5 6.2L15.5 4" stroke="#8392ff" '
        'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>')

GH = ('<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
      '2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-'
      '.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
      '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 '
      '2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 '
      '2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>')

REPO = "https://github.com/sf-stav/veloGB10"


def nav(prefix, active="docs"):
    def cls(name):
        return ' class="active"' if name == active else ""

    return f"""<nav class="nav">
  <div class="container">
    <a class="brand" href="{prefix}index.html">{LOGO} veloGB10 <span class="sub">by veloEngine</span></a>
    <div class="nav-links">
      <a href="{prefix}index.html"{cls("home")}>Overview</a>
      <a href="{prefix}index.html#engineering"{cls("engineering")}>Engineering</a>
      <a href="{prefix}index.html#traces"{cls("traces")}>Throughput</a>
      <a href="{prefix}docs/index.html"{cls("docs")}>Docs</a>
      <a href="https://velobenchmark.com" rel="noopener">VeloBenchmark</a>
      <a class="nav-cta" href="{REPO}" target="_blank" rel="noopener">{GH} GitHub</a>
    </div>
  </div>
</nav>"""


def footer(prefix):
    return f"""<footer>
  <div class="container">
    <div class="foot-grid">
      <div>
        <a class="brand" href="{prefix}index.html" style="margin-bottom:14px">{LOGO} veloGB10 <span class="sub">by veloEngine</span></a>
        <p>A from-scratch Rust + CUDA inference engine for NVIDIA GB10 systems —
           DGX Spark and compatible OEM boxes. Built in the open, measured on real
           hardware.</p>
      </div>
      <div>
        <h4>Product</h4>
        <a href="{prefix}index.html#engineering">Engineering</a>
        <a href="{prefix}index.html#traces">Throughput traces</a>
        <a href="{prefix}index.html#models">Supported models</a>
        <a href="{prefix}index.html#setup">Getting it running</a>
      </div>
      <div>
        <h4>Documentation</h4>
        <a href="{prefix}docs/index.html">Docs home</a>
        <a href="{prefix}docs/index.html">Overview</a>
        <a href="{prefix}docs/setup-qwen27b.html">Qwen3.8-27B setup</a>
        <a href="{prefix}docs/endurance.html">Endurance report</a>
        <a href="{prefix}docs/changelog.html">Changelog</a>
      </div>
      <div>
        <h4>Source</h4>
        <a href="{REPO}" target="_blank" rel="noopener">veloGB10 (Apache-2.0)</a>
        <a href="https://velobenchmark.com" rel="noopener">VeloBenchmark (AGPL-3.0)</a>
        <a href="mailto:stav@scalarflow.co.uk">stav@scalarflow.co.uk</a>
      </div>
    </div>
    <div class="foot-base">
      <span>&copy; 2026 Stav Katsoulis</span>
      <span>veloGB10 Apache-2.0 &middot; veloengine.ai</span>
    </div>
  </div>
</footer>"""


def sidebar(current):
    out = ['<aside class="docs-side">', "<h4>veloGB10</h4>"]
    for name, title in MANIFEST:
        act = ' class="active"' if name == current else ""
        out.append(f'<a href="{name}.html"{act}>{title}</a>')
    out.append('<h4>Related</h4>')
    out.append('<a href="https://velobenchmark.com">VeloBenchmark</a>')
    out.append(f'<a href="{REPO}" target="_blank" rel="noopener">Repository</a>')
    out.append("</aside>")
    return "\n".join(out)


def render(name, title):
    src = os.path.join(CONTENT, name + ".md")
    md = open(src, encoding="utf-8").read()
    md = re.sub(r"^# .*\n", "", md, count=1)  # title comes from the manifest
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "attr_list", "toc"])
    i = [n for (n, _) in MANIFEST].index(name)
    prev = MANIFEST[i - 1] if i > 0 else None
    nxt = MANIFEST[i + 1] if i + 1 < len(MANIFEST) else None
    pager = '<div class="pager">'
    pager += (f'<a href="{prev[0]}.html"><span class="dir">&larr; Previous</span>{prev[1]}</a>'
              if prev else '<span></span>')
    pager += (f'<a class="next" href="{nxt[0]}.html"><span class="dir">Next &rarr;</span>{nxt[1]}</a>'
              if nxt else '<span></span>')
    pager += "</div>"
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — veloGB10 docs</title>
<meta name="description" content="{title} — veloGB10 documentation at veloengine.ai: a from-scratch Rust + CUDA inference engine for NVIDIA GB10 systems.">
<link rel="canonical" href="{BASE}/docs/{name}.html">
<meta property="og:site_name" content="veloGB10 by veloEngine">
<meta property="og:type" content="article">
<meta property="og:title" content="{title} — veloGB10 docs">
<meta property="og:url" content="{BASE}/docs/{name}.html">
<meta property="og:image" content="{BASE}/assets/img/velogb10/tp4_min_max.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="stylesheet" href="../assets/css/site.css">
<link rel="icon" type="image/svg+xml" href='{FAVICON}'>
</head>
<body>
{nav("../")}
<div class="container docs-shell">
{sidebar(name)}
<article class="docs-body">
<h1>{title}</h1>
{body}
{pager}
</article>
</div>
{footer("../")}
</body>
</html>
"""
    dst = os.path.join(OUT, name + ".html")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8").write(html)
    print("built", os.path.relpath(dst, ROOT))


def main():
    for name, title in MANIFEST:
        render(name, title)
    urls = ["/", "/docs/", "/docs/index.html"]
    urls += [f"/docs/{n}.html" for (n, _) in MANIFEST]
    today = datetime.date.today().isoformat()
    body = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    seen = set()
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        body.append(f"  <url><loc>{BASE}{u}</loc><lastmod>{today}</lastmod></url>")
    body.append("</urlset>")
    open(os.path.join(ROOT, "sitemap.xml"), "w", encoding="utf-8").write("\n".join(body) + "\n")
    print("built sitemap.xml (%d urls)" % len(seen))


if __name__ == "__main__":
    main()
