"""
Task 6 — Lexical Search Module (BM25).

Cài đặt:
    pip install rank-bm25 underthesea

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

# =============================================================================
# CONFIGURATION
# =============================================================================

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Cache path — tránh build lại index mỗi lần chạy
BM25_CACHE_PATH = Path(__file__).parent.parent / "data" / "bm25_index.pkl"

# Tokenization strategy:
#   "simple"      → split() theo khoảng trắng, nhanh, không cần cài thêm
#   "underthesea" → word tokenizer tiếng Việt, chính xác hơn nhưng chậm hơn
#
# Lý do cung cấp cả hai:
#   - "simple" phù hợp khi văn bản đã được chuẩn hóa tốt (mỗi token cách nhau space)
#   - "underthesea" cần thiết khi query/document có từ ghép tiếng Việt
#     ví dụ: "ma tuý" vs "matuý", "tàng trữ" vs "tàngtrữ"
TOKENIZE_METHOD = "simple"  # hoặc "underthesea"

# =============================================================================
# CORPUS & INDEX (module-level singleton — chỉ load 1 lần)
# =============================================================================

CORPUS: list[dict] = []   # [{'content': str, 'metadata': dict}, ...]
_bm25_index: BM25Okapi | None = None


# =============================================================================
# TOKENIZER
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt.

    - simple: lowercase + split — đủ dùng cho corpus đã chuẩn hóa.
    - underthesea: word_tokenize — tách từ ghép chính xác hơn.
      Ví dụ: "tàng trữ trái phép" → ["tàng_trữ", "trái_phép"]
      giúp BM25 không bị mismatch giữa "tàng trữ" và "tàng" + "trữ" riêng lẻ.
    """
    text = text.lower().strip()

    if TOKENIZE_METHOD == "underthesea":
        try:
            from underthesea import word_tokenize
            # join="_" để từ ghép thành 1 token; BM25 so khớp chính xác hơn
            return word_tokenize(text, format="text").replace(" ", "_").split("_")
        except ImportError:
            # Fallback nhẹ nhàng nếu underthesea chưa được cài
            pass

    return text.split()


# =============================================================================
# LOAD CORPUS
# =============================================================================

def load_corpus() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Trả về list[dict] với keys:
        - content  : nội dung văn bản
        - metadata : source, source_path, type
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {STANDARDIZED_DIR}")

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")

    corpus = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        doc_type = "legal" if "legal" in md_file.parts else "news"
        corpus.append({
            "content": content,
            "metadata": {
                "source":      md_file.name,
                "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type":        doc_type,
            },
        })

    print(f"  Corpus loaded: {len(corpus)} documents")
    return corpus


# =============================================================================
# BUILD BM25 INDEX
# =============================================================================

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Dùng BM25Okapi với tham số mặc định (k1=1.5, b=0.75):
        - k1=1.5: term saturation — điểm không tăng vô hạn khi từ lặp nhiều lần.
          Phù hợp văn bản pháp lý vì tên điều khoản ("Điều 248") lặp đi lặp lại.
        - b=0.75: length normalization — tài liệu dài không bị penalize quá mạnh.
          Vì BLHS có điều khoản ngắn lẫn dài, cần cân bằng.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance đã được fit trên corpus.
    """
    print(f"  Tokenizing {len(corpus)} documents (method={TOKENIZE_METHOD}) ...")
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]

    print("  Building BM25 index ...")
    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  ✓ BM25 index built — vocab size: {len(bm25.idf):,} terms")
    return bm25


# =============================================================================
# CACHE HELPERS
# =============================================================================

def _save_cache(corpus: list[dict], bm25: BM25Okapi) -> None:
    """Lưu corpus + BM25 index ra file để tái sử dụng."""
    BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump({"corpus": corpus, "bm25": bm25}, f)
    print(f"  ✓ Cache saved → {BM25_CACHE_PATH}")


def _load_cache() -> tuple[list[dict], BM25Okapi] | None:
    """Load corpus + BM25 index từ cache nếu có."""
    if not BM25_CACHE_PATH.exists():
        return None
    with open(BM25_CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"  ✓ Cache loaded ← {BM25_CACHE_PATH}")
    return data["corpus"], data["bm25"]


# =============================================================================
# INITIALIZER
# =============================================================================

def _ensure_index(use_cache: bool = True) -> None:
    """
    Đảm bảo CORPUS và _bm25_index đã được khởi tạo (singleton pattern).

    Args:
        use_cache: Nếu True, thử load từ cache trước; nếu False, build lại.
    """
    global CORPUS, _bm25_index

    if _bm25_index is not None:
        return  # Đã khởi tạo rồi, bỏ qua

    if use_cache:
        cached = _load_cache()
        if cached is not None:
            CORPUS, _bm25_index = cached
            return

    # Build fresh
    CORPUS = load_corpus()
    _bm25_index = build_bm25_index(CORPUS)
    if use_cache:
        _save_cache(CORPUS, _bm25_index)


# =============================================================================
# LEXICAL SEARCH
# =============================================================================

def lexical_search(
    query: str,
    top_k: int = 10,
    score_threshold: float = 0.0,
    use_cache: bool = True,
) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Workflow:
        1. Tokenize query (cùng method với lúc index)
        2. BM25.get_scores() → array điểm cho mỗi document
        3. argsort descending → lấy top_k
        4. Filter score > threshold (tránh trả về kết quả không liên quan)

    Args:
        query           : Câu truy vấn (tiếng Việt)
        top_k           : Số kết quả tối đa trả về
        score_threshold : Chỉ trả về chunk có score > ngưỡng này.
                          Mặc định 0.0 — BM25 score = 0 nghĩa là không có
                          từ nào trong query xuất hiện trong document.
        use_cache       : Dùng cached index nếu có

    Returns:
        List[dict] sorted by score descending, mỗi phần tử gồm:
            - content  : str    — nội dung chunk
            - score    : float  — BM25 score (không có upper bound cố định)
            - metadata : dict   — source, type, v.v.
    """
    _ensure_index(use_cache=use_cache)

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores: np.ndarray = _bm25_index.get_scores(tokenized_query)

    # Lấy top_k indices theo thứ tự score giảm dần
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        s = float(scores[idx])
        if s <= score_threshold:
            # BM25 score = 0 → không có term nào khớp → bỏ qua
            continue
        results.append({
            "content":  CORPUS[idx]["content"],
            "score":    s,
            "metadata": CORPUS[idx]["metadata"],
        })

    return results


# =============================================================================
# REBUILD HELPER (dùng khi corpus thay đổi)
# =============================================================================

def rebuild_index() -> None:
    """
    Xóa cache và build lại index từ đầu.
    Dùng khi thêm/sửa/xóa document trong data/standardized/.
    """
    global CORPUS, _bm25_index
    CORPUS = []
    _bm25_index = None
    if BM25_CACHE_PATH.exists():
        BM25_CACHE_PATH.unlink()
        print(f"  Cache deleted: {BM25_CACHE_PATH}")
    _ensure_index(use_cache=True)


# =============================================================================
# MAIN — DEMO / QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("Task 6: Lexical Search (BM25)")
    print(f"  Tokenize method : {TOKENIZE_METHOD}")
    print("=" * 55)

    TEST_QUERIES = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "hình phạt buôn bán heroin",
        "bắt giữ đối tượng vận chuyển ma túy qua biên giới",
    ]

    for query in TEST_QUERIES:
        print(f"\nQuery: \"{query}\"")
        print("-" * 50)
        results = lexical_search(query, top_k=3)
        if not results:
            print("  (Không có kết quả)")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            preview = r["content"][:120].replace("\n", " ")
            print(f"  [{i}] score={r['score']:.4f} | {meta['source']} ({meta['type']})")
            print(f"       {preview}...")