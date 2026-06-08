"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    crawl4ai-setup   # cài Playwright browsers lần đầu
"""

import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES = 3
RETRY_DELAY = 2          # giây, tăng gấp đôi sau mỗi lần thất bại
DELAY_BETWEEN_URLS = 1.5 # giây, tránh bị rate-limit


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output dir: {DATA_DIR.resolve()}")


ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-long-nhat-son-ngoc-minh-bi-bat-vi-lien-quan-ma-tuy-5060857.html",
    "https://vnexpress.net/anh-em-ca-si-chi-dan-ru-nhieu-nguoi-choi-ma-tuy-nhu-the-nao-4929804.html",
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://thanhnien.vn/miu-le-va-loi-xin-loi-muon-mang-cua-loat-sao-viet-vuong-vao-ma-tuy-18526051513021689.htm",
    "https://tuoitre.vn/vien-kiem-sat-tp-hcm-nhieu-nghe-si-nguoi-noi-tieng-bi-khoi-to-do-lien-quan-ma-tuy-20251209142132042.htm",
]


# ── Helper: trích title từ markdown nếu metadata rỗng ────────────────────────
def _extract_title_from_markdown(markdown: str) -> str:
    """Lấy heading đầu tiên từ nội dung markdown làm title dự phòng."""
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip()
    return "Unknown"


def _slugify_url(url: str) -> str:
    """Tạo tên file an toàn từ URL (dùng khi đặt tên theo URL)."""
    name = re.sub(r"https?://", "", url)
    name = re.sub(r"[^\w]", "_", name)
    return name[:80]


# ── Core crawl function ───────────────────────────────────────────────────────
async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str  (ISO 8601),
            "content_markdown": str,
            "word_count": int,
            "crawl_success": bool,
            "error": str | None,
        }
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    # Cấu hình trình duyệt — headless, tiếng Việt
    browser_cfg = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--lang=vi-VN", "--disable-blink-features=AutomationControlled"],
    )

    # Lọc nội dung: bỏ nav, sidebar, quảng cáo; giữ phần thân bài
    content_filter = PruningContentFilter(
        threshold=0.45,
        threshold_type="fixed",
        min_word_threshold=10,
    )
    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    run_cfg = CrawlerRunConfig(
        markdown_generator=md_generator,
        wait_until="domcontentloaded",
        page_timeout=30_000,        # 30s timeout
        delay_before_return_html=1.5,  # chờ JS render xong
        remove_overlay_elements=True,
        magic=True,                 # auto-bypass các anti-bot đơn giản
    )

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with AsyncWebCrawler(config=browser_cfg) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)

            if not result.success:
                raise RuntimeError(f"Crawl failed: {result.error_message}")

            # Ưu tiên filtered markdown (bỏ boilerplate), fallback raw markdown
            content = (
                result.markdown.fit_markdown
                if hasattr(result.markdown, "fit_markdown") and result.markdown.fit_markdown
                else result.markdown.raw_markdown
                if hasattr(result.markdown, "raw_markdown")
                else str(result.markdown)
            )

            # Title: metadata → <title> tag → heading trong markdown
            title = (
                (result.metadata or {}).get("title")
                or (result.metadata or {}).get("og:title")
                or _extract_title_from_markdown(content)
                or "Unknown"
            )
            # Bỏ suffix tên trang báo ở cuối title (vd: " - VnExpress")
            title = re.sub(r"\s*[-|]\s*(VnExpress|Thanh Niên|Tuổi Trẻ).*$", "", title, flags=re.IGNORECASE).strip()

            word_count = len(content.split())

            return {
                "url": url,
                "title": title,
                "date_crawled": datetime.now().isoformat(),
                "content_markdown": content,
                "word_count": word_count,
                "crawl_success": True,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                print(f"    ⚠ Attempt {attempt} failed ({exc}). Retry in {wait}s...")
                await asyncio.sleep(wait)
            else:
                print(f"    ✗ All {MAX_RETRIES} attempts failed for: {url}")

    # Trả về bản ghi lỗi thay vì raise — để không block các URL còn lại
    return {
        "url": url,
        "title": "Unknown",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": "",
        "word_count": 0,
        "crawl_success": False,
        "error": last_error,
    }


# ── Main orchestrator ─────────────────────────────────────────────────────────
async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS tuần tự (tránh bị block)."""
    setup_directory()

    results_summary = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"\n[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        article = await crawl_article(url)

        # Đặt tên file theo index
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        status = "✓" if article["crawl_success"] else "✗"
        print(f"  {status} Saved: {filepath.name}  |  title: {article['title'][:60]}  |  words: {article['word_count']}")

        results_summary.append({
            "file": filename,
            "url": url,
            "title": article["title"],
            "success": article["crawl_success"],
            "word_count": article["word_count"],
            "error": article["error"],
        })

        # Nghỉ giữa các request để không bị rate-limit
        if i < len(ARTICLE_URLS):
            await asyncio.sleep(DELAY_BETWEEN_URLS)

    # Lưu summary
    summary_path = DATA_DIR / "_crawl_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "crawled_at": datetime.now().isoformat(),
                "total": len(ARTICLE_URLS),
                "success": sum(1 for r in results_summary if r["success"]),
                "articles": results_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # In tóm tắt cuối
    success_count = sum(1 for r in results_summary if r["success"])
    print(f"\n{'='*60}")
    print(f"Hoàn tất: {success_count}/{len(ARTICLE_URLS)} bài crawl thành công")
    print(f"Summary : {summary_path}")
    if success_count < len(ARTICLE_URLS):
        print("Các URL thất bại:")
        for r in results_summary:
            if not r["success"]:
                print(f"  - {r['url']}")
                print(f"    Error: {r['error']}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())