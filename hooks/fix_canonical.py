"""Rewrite the homepage canonical and sitemap entry from `<site>/index.html` to `<site>/`.

We keep `use_directory_urls: false` (notes use raw HTML `<img src="../assets/...">` paths
that depend on the .html URL structure), but Google Search Console flags the homepage as
"Alternate page with proper canonical tag" because `/MLSys-Learning-Notes/index.html` is
declared canonical while crawlers reach `/MLSys-Learning-Notes/`. Fix the homepage only.
"""

from pathlib import Path


def on_post_page(output, page, config, **kwargs):
    if page.url != "index.html":
        return output
    site_url = config["site_url"]
    return output.replace(
        f'<link rel="canonical" href="{site_url}index.html">',
        f'<link rel="canonical" href="{site_url}">',
    )


def on_post_build(config, **kwargs):
    sitemap = Path(config["site_dir"]) / "sitemap.xml"
    if not sitemap.exists():
        return
    site_url = config["site_url"]
    text = sitemap.read_text(encoding="utf-8")
    sitemap.write_text(
        text.replace(f"{site_url}index.html", site_url), encoding="utf-8"
    )
