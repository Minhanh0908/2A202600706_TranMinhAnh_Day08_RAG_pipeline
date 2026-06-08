# """
# Task 4 — Chunking & Indexing vào Vector Store.

# Hướng dẫn:
#     1. Đọc toàn bộ markdown files từ data/standardized/
#     2. Chọn 1 chunking strategy (giải thích lý do)
#     3. Chọn 1 embedding model (giải thích lý do)
#     4. Index vào vector store (Weaviate khuyến cáo)

# Chunking options (langchain-text-splitters):
#     - RecursiveCharacterTextSplitter: an toàn, phổ biến
#     - MarkdownHeaderTextSplitter: tốt cho file có heading
#     - SemanticChunker: dùng embedding để tách (nâng cao)

# Embedding model options:
#     - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
#     - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
#     - OpenAI text-embedding-3-small (1536 dim, API)

# Vector store options:
#     - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
#     - ChromaDB (đơn giản, local)
#     - FAISS (chỉ dense search)

# Cài đặt:
#     pip install langchain-text-splitters sentence-transformers weaviate-client
# """

# from pathlib import Path

# STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# # =============================================================================
# # CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# # =============================================================================

# # TODO: Chọn chunking strategy và giải thích vì sao
# CHUNK_SIZE = 500        # Vì sao chọn 500? ...
# CHUNK_OVERLAP = 50      # Vì sao chọn 50? ...
# CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# # TODO: Chọn embedding model và giải thích
# EMBEDDING_MODEL = "BAAI/bge-m3"  # Vì sao? Multilingual, tốt cho tiếng Việt
# EMBEDDING_DIM = 1024

# # TODO: Chọn vector store
# VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"


# # =============================================================================
# # IMPLEMENTATION
# # =============================================================================

# def load_documents() -> list[dict]:
#     """
#     Đọc toàn bộ markdown files từ data/standardized/.

#     Returns:
#         List of {'content': str, 'metadata': {'source': str, 'type': str}}
#     """
#     # TODO: Iterate qua STANDARDIZED_DIR, đọc .md files
#     # documents = []
#     # for md_file in STANDARDIZED_DIR.rglob("*.md"):
#     #     content = md_file.read_text(encoding="utf-8")
#     #     doc_type = "legal" if "legal" in str(md_file) else "news"
#     #     documents.append({
#     #         "content": content,
#     #         "metadata": {"source": md_file.name, "type": doc_type}
#     #     })
#     # return documents
#     raise NotImplementedError("Implement load_documents")


# def chunk_documents(documents: list[dict]) -> list[dict]:
#     """
#     Chunk documents theo strategy đã chọn.

#     Returns:
#         List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
#     """
#     # TODO: Implement chunking
#     #
#     # Ví dụ với RecursiveCharacterTextSplitter:
#     # from langchain_text_splitters import RecursiveCharacterTextSplitter
#     #
#     # splitter = RecursiveCharacterTextSplitter(
#     #     chunk_size=CHUNK_SIZE,
#     #     chunk_overlap=CHUNK_OVERLAP,
#     #     separators=["\n\n", "\n", ". ", " ", ""]
#     # )
#     # chunks = []
#     # for doc in documents:
#     #     splits = splitter.split_text(doc["content"])
#     #     for i, chunk_text in enumerate(splits):
#     #         chunks.append({
#     #             "content": chunk_text,
#     #             "metadata": {**doc["metadata"], "chunk_index": i}
#     #         })
#     # return chunks
#     raise NotImplementedError("Implement chunk_documents")


# def embed_chunks(chunks: list[dict]) -> list[dict]:
#     """
#     Embed toàn bộ chunks bằng model đã chọn.

#     Returns:
#         Mỗi chunk dict được thêm key 'embedding': list[float]
#     """
#     # TODO: Implement embedding
#     #
#     # Ví dụ với sentence-transformers:
#     # from sentence_transformers import SentenceTransformer
#     #
#     # model = SentenceTransformer(EMBEDDING_MODEL)
#     # texts = [c["content"] for c in chunks]
#     # embeddings = model.encode(texts, show_progress_bar=True)
#     # for chunk, emb in zip(chunks, embeddings):
#     #     chunk["embedding"] = emb.tolist()
#     # return chunks
#     raise NotImplementedError("Implement embed_chunks")


# def index_to_vectorstore(chunks: list[dict]):
#     """
#     Lưu chunks vào vector store đã chọn.
#     """
#     # TODO: Implement indexing
#     #
#     # Ví dụ với Weaviate:
#     # import weaviate
#     # from weaviate.classes.config import Configure, Property, DataType
#     #
#     # client = weaviate.connect_to_local()  # hoặc connect_to_weaviate_cloud()
#     #
#     # # Tạo collection
#     # collection = client.collections.create(
#     #     name="DrugLawDocs",
#     #     vectorizer_config=Configure.Vectorizer.none(),
#     #     properties=[
#     #         Property(name="content", data_type=DataType.TEXT),
#     #         Property(name="source", data_type=DataType.TEXT),
#     #         Property(name="doc_type", data_type=DataType.TEXT),
#     #     ]
#     # )
#     #
#     # # Insert chunks
#     # with collection.batch.dynamic() as batch:
#     #     for chunk in chunks:
#     #         batch.add_object(
#     #             properties={"content": chunk["content"], ...},
#     #             vector=chunk["embedding"]
#     #         )
#     raise NotImplementedError("Implement index_to_vectorstore")


# def run_pipeline():
#     """Chạy toàn bộ pipeline: load → chunk → embed → index."""
#     print("=" * 50)
#     print("Task 4: Chunking & Indexing")
#     print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
#     print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
#     print(f"  Vector Store: {VECTOR_STORE}")
#     print("=" * 50)

#     docs = load_documents()
#     print(f"\n✓ Loaded {len(docs)} documents")

#     chunks = chunk_documents(docs)
#     print(f"✓ Created {len(chunks)} chunks")

#     chunks = embed_chunks(chunks)
#     print(f"✓ Embedded {len(chunks)} chunks")

#     index_to_vectorstore(chunks)
#     print("✓ Indexed to vector store")


# if __name__ == "__main__":
#     run_pipeline()

"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking strategy: "markdown_header" → "recursive" (two-pass)
# - Pass 1 (MarkdownHeaderTextSplitter): giữ nguyên ngữ nghĩa theo heading,
#   giúp mỗi chunk biết mình thuộc phần nào của văn bản.
# - Pass 2 (RecursiveCharacterTextSplitter): cắt nhỏ các section quá dài,
#   tránh chunk vượt context window của embedding model.
# Dùng kết hợp thay vì chỉ dùng recursive vì dữ liệu có cấu trúc heading rõ
# (legal docs) lẫn văn xuôi tự do (news articles).

CHUNK_SIZE = 512
# 512 tokens ≈ giới hạn tối ưu của BGE-M3; đủ để giữ ngữ cảnh 1 đoạn pháp lý
# mà không cắt ngang câu quan trọng.

CHUNK_OVERLAP = 64
# ~12% overlap — đủ để không mất ngữ cảnh ở ranh giới chunk (ví dụ: chủ ngữ
# câu trước quan trọng cho câu sau), nhưng không quá lớn gây trùng lặp vector.

CHUNKING_METHOD = "markdown_header+recursive"

# Embedding model: BAAI/bge-m3
# - Multilingual (hỗ trợ tiếng Việt tốt, được train trên dữ liệu đa ngôn ngữ)
# - 1024 dim — cân bằng giữa độ chính xác và bộ nhớ
# - Hỗ trợ hybrid retrieval (dense + sparse) natively → khớp với Weaviate
# - Miễn phí, chạy local, không cần API key
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 32   # batch để tránh OOM trên máy yếu

# Vector store: Weaviate
# - Hybrid search (BM25 + dense vector) built-in: quan trọng với pháp lý Việt
#   vì tên điều luật, số điều khoản cần exact match mà dense search bỏ sót.
# - Collection schema rõ ràng, dễ filter theo doc_type / source
VECTOR_STORE = "weaviate"
WEAVIATE_HOST = "localhost"
WEAVIATE_PORT = 8080
COLLECTION_NAME = "DrugLawDocs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {STANDARDIZED_DIR}")

    documents = []
    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))

    if not md_files:
        raise RuntimeError(f"Không tìm thấy file .md nào trong {STANDARDIZED_DIR}")

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        # Xác định loại tài liệu theo cấu trúc thư mục
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
    """
    Two-pass chunking:
      Pass 1 — MarkdownHeaderTextSplitter: tách theo heading (H1/H2/H3)
      Pass 2 — RecursiveCharacterTextSplitter: cắt nhỏ section quá dài

    Returns:
        List of {'content': str, 'metadata': dict}
    """
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    # Pass 1: tách theo heading markdown
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ],
        strip_headers=False,   # giữ heading trong chunk để embedding nắm context
    )

    # Pass 2: cắt nhỏ nếu section vẫn quá dài
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ". ", "! ", "? ", " ", ""],
        # Thêm dấu câu tiếng Việt để tách tự nhiên hơn
    )

    chunks = []
    for doc in documents:
        base_meta = doc["metadata"]

        # Pass 1
        header_chunks = header_splitter.split_text(doc["content"])

        # Pass 2 — áp dụng lên mỗi header-chunk
        for hchunk in header_chunks:
            # hchunk là Document(page_content=..., metadata={h1:..., h2:...})
            section_meta = {
                **base_meta,
                **hchunk.metadata,   # h1, h2, h3 nếu có
            }
            sub_splits = recursive_splitter.split_text(hchunk.page_content)
            for i, text in enumerate(sub_splits):
                text = text.strip()
                if not text:
                    continue
                chunks.append({
                    "content": text,
                    "metadata": {
                        **section_meta,
                        "chunk_index": i,
                        "chunk_char_len": len(text),
                    },
                })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng BAAI/bge-m3.
    Xử lý theo batch để tránh OOM.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    all_embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        batch_embs = model.encode(
            batch,
            normalize_embeddings=True,   # cosine similarity = dot product sau normalize
            show_progress_bar=False,
        )
        all_embeddings.extend(batch_embs.tolist())
        print(f"  Embedded {min(start + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} chunks", end="\r")

    print()  # newline sau progress

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb

    return chunks


def _ensure_collection(client):
    """Tạo Weaviate collection nếu chưa có, bỏ qua nếu đã tồn tại."""
    import weaviate.classes.config as wc

    existing = [c.name for c in client.collections.list_all().values()]
    if COLLECTION_NAME in existing:
        print(f"  Collection '{COLLECTION_NAME}' đã tồn tại — dùng lại.")
        return client.collections.get(COLLECTION_NAME)

    collection = client.collections.create(
        name=COLLECTION_NAME,
        # Không dùng built-in vectorizer vì ta tự embed
        vectorizer_config=wc.Configure.Vectorizer.none(),
        # Bật BM25 index để hỗ trợ hybrid search
        inverted_index_config=wc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
        ),
        properties=[
            wc.Property(name="content",      data_type=wc.DataType.TEXT),
            wc.Property(name="source",       data_type=wc.DataType.TEXT),
            wc.Property(name="source_path",  data_type=wc.DataType.TEXT),
            wc.Property(name="doc_type",     data_type=wc.DataType.TEXT),
            wc.Property(name="chunk_index",  data_type=wc.DataType.INT),
            wc.Property(name="h1",           data_type=wc.DataType.TEXT),
            wc.Property(name="h2",           data_type=wc.DataType.TEXT),
            wc.Property(name="h3",           data_type=wc.DataType.TEXT),
        ],
    )
    print(f"  ✓ Đã tạo collection '{COLLECTION_NAME}'")
    return collection


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào Weaviate với vector tự embed (vectorizer=none).
    Dùng batch insert để tối ưu throughput.
    """
    import weaviate

    print(f"  Kết nối Weaviate tại {WEAVIATE_HOST}:{WEAVIATE_PORT} ...")
    client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)

    try:
        collection = _ensure_collection(client)

        ok, fail = 0, 0
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
        client.close()


# =============================================================================
# PIPELINE
# =============================================================================

def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 55)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking : {CHUNKING_METHOD}")
    print(f"             size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} → collection={COLLECTION_NAME}")
    print("=" * 55)

    print("\n[1/4] Loading documents ...")
    docs = load_documents()
    print(f"  → {len(docs)} documents loaded")

    print("\n[2/4] Chunking ...")
    chunks = chunk_documents(docs)
    print(f"  → {len(chunks)} chunks created")
    avg_len = sum(c["metadata"]["chunk_char_len"] for c in chunks) / max(len(chunks), 1)
    print(f"  → Avg chunk length: {avg_len:.0f} chars")

    print("\n[3/4] Embedding ...")
    chunks = embed_chunks(chunks)
    print(f"  → {len(chunks)} embeddings generated (dim={EMBEDDING_DIM})")

    print("\n[4/4] Indexing to Weaviate ...")
    index_to_vectorstore(chunks)

    print("\n✓ Pipeline hoàn tất!")


if __name__ == "__main__":
    run_pipeline()