"""Convert HuggingFace MDX files to MkDocs-compatible Markdown."""

import re
import sys
from pathlib import Path


def convert(text: str) -> str:
    # <hfoptions>/<hfoption> → MkDocs tabs
    def replace_hfoptions(m):
        inner = m.group(1)
        tabs = re.findall(r'<hfoption id="([^"]+)">(.*?)</hfoption>', inner, re.DOTALL)
        if not tabs:
            return inner
        parts = []
        for tab_id, tab_content in tabs:
            parts.append(
                f'=== "{tab_id}"\n' + "\n".join("    " + l for l in tab_content.strip().splitlines())
            )
        return "\n\n" + "\n\n".join(parts) + "\n"

    text = re.sub(r"<hfoptions[^>]*>(.*?)</hfoptions>", replace_hfoptions, text, flags=re.DOTALL)

    # <Tip warning={true}> → !!! warning
    text = re.sub(
        r"<Tip warning=\{true\}>(.*?)</Tip>",
        lambda m: "!!! warning\n" + "\n".join("    " + l for l in m.group(1).strip().splitlines()),
        text,
        flags=re.DOTALL,
    )

    # <Tip> content </Tip> (multiline)
    text = re.sub(
        r"<Tip>(.*?)</Tip>",
        lambda m: "!!! tip\n" + "\n".join("    " + l for l in m.group(1).strip().splitlines()),
        text,
        flags=re.DOTALL,
    )

    # Remove remaining unknown JSX-style components
    text = re.sub(r"<(?:Youtube|DocNotebook\w*|frameworkcontent|TokenizersDoc|PipelineTag)[^/]*/>", "", text)
    text = re.sub(
        r"<(?:Youtube|DocNotebook\w*|frameworkcontent|TokenizersDoc|PipelineTag)[^>]*>.*?</\w+>",
        "",
        text,
        flags=re.DOTALL,
    )

    # Fix self-closing img tags: <img ... ></img> or <img ... />
    text = re.sub(r"<img([^>]*?)>\s*</img>", r"<img\1>", text)

    # Fix .mdx links → .md first, then fix bare relative links
    text = text.replace(".mdx)", ".md)").replace(".mdx#", ".md#")

    # Fix relative links: ./foo) or ./foo#anchor) → ./foo.md) (skip already suffixed)
    text = re.sub(r"\((\./[^)#\s]+?)(?<!\.md)(?<!\.html)(#[^)]*?)?\)", _fix_link, text)

    return text


def _fix_link(m):
    path, anchor = m.group(1), m.group(2) or ""
    # Skip external links and already-suffixed paths
    if path.startswith("./") and not any(
        path.endswith(s) for s in (".md", ".html", ".png", ".jpg", ".gif", ".svg")
    ):
        return f"({path}.md{anchor})"
    return m.group(0)


def main():
    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_dir.glob("*.mdx"):
        dst = dst_dir / (src.stem + ".md")
        dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    # Copy .md files directly
    for src in src_dir.glob("*.md"):
        (dst_dir / src.name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
