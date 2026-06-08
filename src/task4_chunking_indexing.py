"""
Task 4 — Chunking & Indexing vào Vector Store.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client python-dotenv

Cấu hình .env:
    WEAVIATE_URL=https://xxx.weaviate.network
    WEAVIATE_API_KEY=your-api-key
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking strategy: "markdown_header" → "recursive" (two-pass)
#
# Lý do chọn kết hợp hai method thay vì chỉ dùng một:
#   - Pass 1 (MarkdownHeaderTextSplitter): tách văn bản theo cấu trúc heading
#     (H1/H2/H3), giúp mỗi chunk biết mình thuộc phần nào của tài liệu.
#     Quan trọng với văn bản pháp lý vì mỗi Chương/Điều/Khoản là một đơn vị
#     ngữ nghĩa độc lập — không nên bị cắt ngang.
#   - Pass 2 (RecursiveCharacterTextSplitter): cắt nhỏ các section vẫn còn
#     quá dài sau pass 1, đảm bảo không có chunk nào vượt giới hạn context
#     window của embedding model.
#   - Dùng kết hợp vì dữ liệu hỗn hợp: legal docs có heading rõ ràng, còn
#     news articles là văn xuôi tự do — recursive xử lý tốt cả hai.

CHUNK_SIZE = 512
# Vì sao chọn 512?
#   - BGE-M3 xử lý tốt nhất đoạn văn khoảng 256–512 token. Với tiếng Việt,
#     1 từ ≈ 3–4 ký tự, nên 512 ký tự ≈ 128–170 token — nằm trong vùng tối ưu.
#   - Đủ dài để giữ ngữ cảnh 1 khoản pháp lý hoặc 1 đoạn bài báo hoàn chỉnh,
#     không quá dài làm loãng vector embedding.

CHUNK_OVERLAP = 64
# Vì sao chọn 64?
#   - 64/512 ≈ 12.5% overlap — đủ để không mất ngữ cảnh ở ranh giới chunk.
#     Ví dụ: chủ ngữ hoặc điều kiện ở câu trước có thể quyết định nghĩa câu sau.
#   - Giữ overlap thấp để tránh 2 chunk liền nhau có vector quá giống nhau,
#     gây trùng lặp kết quả khi retrieve.

CHUNKING_METHOD = "markdown_header+recursive"

# Embedding model: BAAI/bge-m3
# Lý do chọn:
#   - Multilingual: được train trên corpus đa ngôn ngữ bao gồm tiếng Việt,
#     hiểu ngữ nghĩa tiếng Việt tốt hơn các model chỉ train trên tiếng Anh.
#   - 1024 dim: độ chính xác cao hơn MiniLM-L6 (384 dim) mà chi phí lưu trữ
#     vẫn thấp hơn OpenAI text-embedding-3-small (1536 dim).
#   - Chạy local hoàn toàn, miễn phí, không cần API key.
#   - Hỗ trợ hybrid retrieval (dense + sparse) natively — khớp với Weaviate.

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Batch size khi encode — tránh OOM trên máy không có GPU mạnh.
# Tăng lên 64 nếu RAM/VRAM đủ để encode nhanh hơn.
EMBEDDING_BATCH_SIZE = 32

# Vector store: Weaviate Cloud
# Lý do chọn Weaviate:
#   - Hỗ trợ hybrid search (BM25 + dense vector) built-in: cần thiết với văn
#     bản pháp lý vì tên điều khoản kiểu "Điều 194 BLHS" hay mã văn bản cần
#     exact match — dense search thường bỏ sót những trường hợp này.
#   - Schema property rõ ràng, dễ filter theo doc_type (legal/news) hoặc
#     theo heading (h1/h2/h3) khi cần thu hẹp phạm vi tìm kiếm.
#   - Hỗ trợ cả local (Docker) lẫn Weaviate Cloud — linh hoạt khi chuyển
#     từ dev sang production.

VECTOR_STORE = "weaviate"
COLLECTION_NAME = "DrugLawDocs"

# Đọc credentials từ .env — không hardcode vào source code
WEAVIATE_URL     = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """Đọc toàn bộ markdown files từ data/standardized/."""
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {STANDARDIZED_DIR}")

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        raise RuntimeError(f"Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")

    documents = []
    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "source_path": str(md_file.relative_to(STANDARDIZED_DIR)),
                "type": doc_type,
            },
        })
        print(f"  Loaded [{doc_type}]: {md_file.name} ({len(content):,} chars)")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Two-pass chunking: MarkdownHeader → Recursive."""
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Pass 1: tách theo cấu trúc heading markdown.
    # strip_headers=False để giữ heading trong nội dung chunk —
    # embedding model sẽ hiểu chunk thuộc Chương/Điều nào.
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )

    # Pass 2: cắt nhỏ các section còn quá dài.
    # separators theo thứ tự ưu tiên: đoạn văn → dòng → câu → từ → ký tự.
    # Thêm "。" để xử lý văn bản có dấu câu kiểu CJK đôi khi xuất hiện
    # trong nội dung crawl từ các trang báo.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ". ", "! ", "? ", " ", ""],
    )

    chunks = []
    for doc in documents:
        for hchunk in header_splitter.split_text(doc["content"]):
            # Merge metadata gốc với heading context từ pass 1
            section_meta = {**doc["metadata"], **hchunk.metadata}
            for i, text in enumerate(recursive_splitter.split_text(hchunk.page_content)):
                text = text.strip()
                if text:
                    chunks.append({
                        "content": text,
                        "metadata": {**section_meta, "chunk_index": i, "chunk_char_len": len(text)},
                    })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks bằng BAAI/bge-m3, xử lý theo batch."""
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = [c["content"] for c in chunks]

    all_embeddings = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        # normalize_embeddings=True: cosine similarity = dot product sau normalize,
        # Weaviate tính nhanh hơn và không cần config thêm distance metric.
        embs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_embeddings.extend(embs.tolist())
        print(f"  Embedded {min(start + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} chunks", end="\r")

    print()
    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


def _get_client():
    """Kết nối tới Weaviate Cloud dùng URL + API key từ .env."""
    import weaviate
    from weaviate.auth import AuthApiKey

    print(f"  Kết nối Weaviate Cloud: {WEAVIATE_URL} ...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )
    assert client.is_ready(), "Weaviate chưa sẵn sàng — kiểm tra URL và API key"
    print("  ✓ Kết nối thành công")
    return client


def _ensure_collection(client):
    """Tạo collection nếu chưa có, dùng lại nếu đã tồn tại."""
    import weaviate.classes.config as wc

    # Kiểm tra trước khi tạo — chạy lại script nhiều lần không bị lỗi duplicate
    if client.collections.exists(COLLECTION_NAME):
        print(f"  Collection '{COLLECTION_NAME}' đã tồn tại — dùng lại.")
        return client.collections.get(COLLECTION_NAME)

    collection = client.collections.create(
        name=COLLECTION_NAME,
        # vectorizer=none vì ta tự embed bằng BGE-M3, không dùng built-in vectorizer
        vectorizer_config=wc.Configure.Vectorizer.none(),
        # Bật BM25 index để hỗ trợ hybrid search:
        #   bm25_b=0.75: mức chuẩn, cân bằng giữa TF và độ dài tài liệu
        #   bm25_k1=1.2: saturation parameter, phù hợp văn bản pháp lý dài
        inverted_index_config=wc.Configure.inverted_index(bm25_b=0.75, bm25_k1=1.2),
        properties=[
            wc.Property(name="content",      data_type=wc.DataType.TEXT),
            wc.Property(name="source",       data_type=wc.DataType.TEXT),
            wc.Property(name="source_path",  data_type=wc.DataType.TEXT),
            wc.Property(name="doc_type",     data_type=wc.DataType.TEXT),
            wc.Property(name="chunk_index",  data_type=wc.DataType.INT),
            # Lưu heading context để có thể filter/retrieve theo cấu trúc tài liệu
            wc.Property(name="h1",           data_type=wc.DataType.TEXT),
            wc.Property(name="h2",           data_type=wc.DataType.TEXT),
            wc.Property(name="h3",           data_type=wc.DataType.TEXT),
        ],
    )
    print(f"  ✓ Đã tạo collection '{COLLECTION_NAME}'")
    return collection


def index_to_vectorstore(chunks: list[dict]):
    """Insert chunks vào Weaviate Cloud theo batch."""
    client = _get_client()
    try:
        collection = _ensure_collection(client)

        ok, fail = 0, 0
        # batch.dynamic(): Weaviate tự điều chỉnh batch size dựa trên response time
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                meta = chunk["metadata"]
                try:
                    batch.add_object(
                        properties={
                            "content":     chunk["content"],
                            "source":      meta.get("source", ""),
                            "source_path": meta.get("source_path", ""),
                            "doc_type":    meta.get("type", ""),
                            "chunk_index": meta.get("chunk_index", 0),
                            # h1/h2/h3 từ MarkdownHeaderTextSplitter — rỗng nếu không có heading
                            "h1":          meta.get("h1", ""),
                            "h2":          meta.get("h2", ""),
                            "h3":          meta.get("h3", ""),
                        },
                        vector=chunk["embedding"],
                    )
                    ok += 1
                except Exception as exc:
                    print(f"\n  ✗ Lỗi insert chunk: {exc}")
                    fail += 1

        print(f"  ✓ Indexed {ok} chunks  |  {fail} lỗi")
    finally:
        # Đảm bảo connection luôn được đóng dù có lỗi hay không
        client.close()


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    print("=" * 55)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : Weaviate Cloud → collection={COLLECTION_NAME}")
    print("=" * 55)

    print("\n[1/4] Loading documents ...")
    docs = load_documents()
    print(f"  → {len(docs)} documents loaded")

    print("\n[2/4] Chunking ...")
    chunks = chunk_documents(docs)
    avg_len = sum(c["metadata"]["chunk_char_len"] for c in chunks) / max(len(chunks), 1)
    print(f"  → {len(chunks)} chunks  |  avg {avg_len:.0f} chars")

    print("\n[3/4] Embedding ...")
    chunks = embed_chunks(chunks)
    print(f"  → {len(chunks)} embeddings (dim={EMBEDDING_DIM})")

    print("\n[4/4] Indexing to Weaviate Cloud ...")
    index_to_vectorstore(chunks)

    print("\n✓ Pipeline hoàn tất!")


if __name__ == "__main__":
    run_pipeline()