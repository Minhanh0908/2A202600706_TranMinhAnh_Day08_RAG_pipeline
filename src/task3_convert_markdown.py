"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:  
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"

    if not legal_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {legal_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    md = MarkItDown()

    files = [f for f in legal_dir.iterdir() if f.suffix.lower() in LEGAL_EXTENSIONS]
    if not files:
        print("  ⚠ Không tìm thấy file PDF/DOCX nào.")
        return

    ok, fail = 0, 0
    for filepath in sorted(files):
        print(f"  Converting: {filepath.name}")
        try:
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            print(f"    ✓ Saved: {output_path.name}  ({len(result.text_content):,} chars)")
            ok += 1
        except Exception as exc:
            print(f"    ✗ Lỗi khi convert {filepath.name}: {exc}")
            fail += 1

    print(f"  → Legal docs: {ok} thành công, {fail} thất bại")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"

    if not news_dir.exists():
        print(f"  ⚠ Thư mục không tồn tại: {news_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Bỏ qua file summary tổng hợp
    files = [
        f for f in sorted(news_dir.iterdir())
        if f.suffix.lower() == ".json" and not f.name.startswith("_")
    ]
    if not files:
        print("  ⚠ Không tìm thấy file JSON nào.")
        return

    ok, fail, skipped = 0, 0, 0
    for filepath in files:
        print(f"  Converting: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            # Bỏ qua bài crawl thất bại (không có nội dung)
            if not data.get("crawl_success", True):
                print(f"    ⚠ Skipped (crawl_success=false): {data.get('error', '')}")
                skipped += 1
                continue

            content_md = data.get("content_markdown", "").strip()
            if not content_md:
                print("    ⚠ Skipped (content rỗng)")
                skipped += 1
                continue

            # Header YAML-style để downstream pipeline dễ parse
            header = (
                f"# {data.get('title', 'Unknown')}\n\n"
                f"**Source:** {data.get('url', 'N/A')}  \n"
                f"**Crawled:** {data.get('date_crawled', 'N/A')}  \n"
                f"**Words:** {data.get('word_count', 'N/A')}  \n\n"
                "---\n\n"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(header + content_md, encoding="utf-8")
            print(f"    ✓ Saved: {output_path.name}  ({data.get('word_count', '?')} words)")
            ok += 1

        except json.JSONDecodeError as exc:
            print(f"    ✗ JSON không hợp lệ: {exc}")
            fail += 1
        except Exception as exc:
            print(f"    ✗ Lỗi: {exc}")
            fail += 1

    print(f"  → News articles: {ok} thành công, {skipped} bỏ qua, {fail} thất bại")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)
    print(f"Input : {LANDING_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print(f"\n✓ Done! Output tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()