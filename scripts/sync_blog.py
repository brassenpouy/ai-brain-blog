#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    os.environ.get(
        "BLOG_SOURCE",
        "/Users/chad/Library/Mobile Documents/iCloud~md~obsidian/Documents/AI-Brain/Blog",
    )
)
CONTENT = ROOT / "content"
POSTS = CONTENT / "posts"
IMAGES = ROOT / "static" / "images"
MANIFEST = ROOT / ".sync-posts.json"


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "~"}:
        return None
    return value


def parse_front_matter(text: str) -> tuple[dict, str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing opening frontmatter delimiter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as exc:
        raise ValueError("missing closing frontmatter delimiter") from exc

    data: dict = {}
    current_key = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw[:1].isspace() and raw.strip().startswith("- "):
            if current_key is None:
                raise ValueError("list item without a key")
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(parse_scalar(raw.strip()[2:]))
            continue
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if not value:
            data[key] = []
            current_key = key
        elif value.startswith("[") and value.endswith("]"):
            data[key] = [
                parse_scalar(item) for item in value[1:-1].split(",") if item.strip()
            ]
            current_key = key
        else:
            data[key] = parse_scalar(value)
            current_key = key

    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return data, body


def as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value).strip()]
    return []


def slugify(stem: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", stem.strip().lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug or "post"


def yaml_string(value) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def render_post(data: dict, body: str, path: Path) -> str:
    title = str(data.get("title") or path.stem)
    date = str(data.get("date") or datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d"))
    description = str(data.get("description") or "")
    tags = as_list(data.get("tags"))
    slug = slugify(str(data.get("slug") or title))

    lines = [
        "---",
        f"title: {yaml_string(title)}",
        f"date: {yaml_string(date)}",
        f"description: {yaml_string(description)}",
        "tags:",
    ]
    for tag in tags:
        lines.append(f"  - {yaml_string(tag)}")
    lines.extend(
        [
            "draft: false",
            f"slug: {yaml_string(slug)}",
            "---",
            "",
            body.strip(),
            "",
        ]
    )
    return "\n".join(lines)


def render_home(data: dict, body: str) -> str:
    title = str(data.get("title") or "硅基深潜")
    description = str(data.get("description") or "Connecting Silicon with AI")
    return "\n".join(
        [
            "---",
            f"title: {yaml_string(title)}",
            f"description: {yaml_string(description)}",
            "---",
            "",
            body.strip(),
            "",
        ]
    )


def convert_images(body: str) -> str:
    pattern = re.compile(r"!\[\[([^\]]+)\]\]")

    def replace(match: re.Match) -> str:
        inner = match.group(1)
        parts = inner.split("|", 1)
        target = parts[0].strip()
        alt = target
        if len(parts) > 1 and parts[1].strip() and not parts[1].strip().isdigit():
            alt = parts[1].strip()
        return f"![{alt}](/images/{quote(target, safe='/')})"

    return pattern.sub(replace, body)


def load_manifest() -> list[str]:
    if not MANIFEST.exists():
        return []
    try:
        data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return [str(item) for item in data if str(item).strip()]


def save_manifest(slugs: list[str]) -> None:
    MANIFEST.write_text(
        json.dumps(sorted(set(slugs)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sync(source: Path, check: bool) -> int:
    if not source.is_dir():
        print(f"source not found: {source}", file=sys.stderr)
        return 1

    home_path = source / "index.md"
    if not home_path.exists():
        print(f"homepage not found: {home_path}", file=sys.stderr)
        return 1

    attachments = source / "attachments"
    image_count = 0
    if attachments.is_dir():
        for item in attachments.iterdir():
            if item.is_file() and not item.name.startswith("."):
                if not check:
                    IMAGES.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, IMAGES / item.name)
                image_count += 1

    home_data, home_body = parse_front_matter(home_path.read_text(encoding="utf-8"))
    home_output = render_home(home_data, home_body)

    old_slugs = load_manifest()
    new_slugs: list[str] = []
    errors: list[str] = []
    posts: list[tuple[str, str]] = []

    for path in sorted(source.glob("*.md")):
        if path.name == "index.md" or path.name.startswith("."):
            continue
        data, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if data.get("publish") is not True:
            continue
        body = convert_images(body)
        remaining = re.findall(r"\[\[([^\]]+)\]\]", body)
        if remaining:
            errors.append(f"{path.name}: unresolved wikilinks: {', '.join(remaining)}")
            continue
        slug = slugify(str(data.get("slug") or data.get("title") or path.stem))
        if slug in new_slugs:
            errors.append(f"{path.name}: duplicate slug: {slug}")
            continue
        new_slugs.append(slug)
        posts.append((slug, render_post(data, body, path)))

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if check:
        print(f"check ok: {len(posts)} posts, {image_count} attachments")
        return 0

    CONTENT.mkdir(parents=True, exist_ok=True)
    POSTS.mkdir(parents=True, exist_ok=True)
    (CONTENT / "_index.md").write_text(home_output, encoding="utf-8")

    for slug in old_slugs:
        if slug in new_slugs:
            continue
        target = POSTS / slug
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    for slug, output in posts:
        target = POSTS / slug
        target.mkdir(parents=True, exist_ok=True)
        (target / "index.md").write_text(output, encoding="utf-8")

    save_manifest(new_slugs)
    print(f"synced: {len(posts)} posts, {image_count} attachments")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return sync(args.source, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
