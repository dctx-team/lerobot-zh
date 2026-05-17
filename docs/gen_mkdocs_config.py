"""Generate mkdocs.yml nav from _toctree.yml."""

import sys
import yaml
from pathlib import Path


def toctree_to_nav(toctree):
    nav = []
    for section in toctree:
        title = section.get("title", "")
        children = []
        for item in section.get("sections", []):
            children.append({item["title"]: item["local"] + ".md"})
        nav.append({title: children})
    return nav


def main():
    lang = sys.argv[1]  # "zh" or "en"
    toctree_path = Path(f"docs/source/{lang}/_toctree.yml")
    toctree = yaml.safe_load(toctree_path.read_text(encoding="utf-8"))
    nav = toctree_to_nav(toctree)

    config = {
        "site_name": "LeRobot 中文文档" if lang == "zh" else "LeRobot Documentation",
        "site_url": f"https://dctx-team.github.io/lerobot-zh/{lang}/",
        "docs_dir": f"docs/_md/{lang}",
        "site_dir": f"docs/_build/{lang}",
        "theme": {
            "name": "material",
            "language": "zh" if lang == "zh" else "en",
            "features": ["navigation.tabs", "navigation.sections", "navigation.top", "search.highlight"],
            "palette": {"scheme": "default", "primary": "indigo"},
        },
        "markdown_extensions": [
            "admonition",
            "pymdownx.superfences",
            {"pymdownx.tabbed": {"alternate_style": True}},
        ],
        "nav": nav,
    }

    out = Path(f"mkdocs_{lang}.yml")
    out.write_text(yaml.dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
