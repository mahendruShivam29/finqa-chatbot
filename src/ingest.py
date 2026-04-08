import json
import pickle
import time
import uuid
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import faiss
import tiktoken
from langchain_classic.retrievers import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.stores import InMemoryStore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


DATA_DIR = Path("data")
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
DOCSTORE_FILE = DATA_DIR / "docstore.pkl"
TABLE_SUMMARIES_CHECKPOINT = DATA_DIR / "table_summaries_checkpoint.json"
DATASET_FILES = ("train.json", "dev.json", "test.json")
ENCODING = tiktoken.get_encoding("cl100k_base")
RRF_K = 60
TABLE_SUMMARY_REQUEST_TIMEOUT_S = 60
TABLE_SUMMARY_MAX_RETRIES = 3
TABLE_SUMMARY_INITIAL_BACKOFF_S = 2
TABLE_SUMMARY_CHECKPOINT_EVERY = 10
TABLE_SUMMARY_PROGRESS_EVERY = 5


@dataclass
class HybridRetriever:
    bm25: BM25Okapi
    bm25_docs: list[Document]
    multi_vector_retriever: MultiVectorRetriever
    parent_lookup: dict[str, Document]

    def invoke(self, query: str, k: int = 3) -> list[Document]:
        return hybrid_retrieve(
            query=query,
            k=k,
            bm25=self.bm25,
            bm25_docs=self.bm25_docs,
            multi_vector_retriever=self.multi_vector_retriever,
            parent_lookup=self.parent_lookup,
        )


def ensure_data_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_samples(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    samples: list[dict[str, Any]] = []
    for filename in DATASET_FILES:
        file_path = data_dir / filename
        if not file_path.exists():
            continue
        samples.extend(load_json_file(file_path))
    return samples


def table_to_markdown(table: list[list[Any]]) -> str:
    if not table:
        return ""

    normalized_rows = [
        ["" if cell is None else str(cell).strip() for cell in row]
        for row in table
    ]
    width = max(len(row) for row in normalized_rows)
    padded_rows = [row + [""] * (width - len(row)) for row in normalized_rows]

    header = padded_rows[0]
    separator = ["---"] * width
    body = padded_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _paragraph_documents(paragraphs: list[str], sample_id: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    documents: list[Document] = []

    for index, paragraph in enumerate(paragraphs):
        text = paragraph.strip()
        if not text:
            continue

        if len(ENCODING.encode(text)) <= 512:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"type": "text", "source": sample_id, "chunk_index": index},
                )
            )
            continue

        for chunk in splitter.split_text(text):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"type": "text", "source": sample_id, "chunk_index": index},
                )
            )
    return documents


def preprocess_sample(sample: dict[str, Any]) -> list[Document]:
    sample_id = str(sample["id"])
    paragraphs = [*sample.get("pre_text", []), *sample.get("post_text", [])]
    documents = _paragraph_documents(paragraphs, sample_id)
    documents.append(
        Document(
            page_content=table_to_markdown(sample.get("table", [])),
            metadata={"type": "table", "source": sample_id, "chunk_index": 0},
        )
    )
    return documents


def should_skip_ingestion(index_dir: str | Path = FAISS_INDEX_DIR) -> bool:
    index_dir = Path(index_dir)
    return index_dir.exists() and FAISS_INDEX_FILE.exists() and DOCSTORE_FILE.exists()


def load_table_summary_checkpoint(
    checkpoint_path: str | Path = TABLE_SUMMARIES_CHECKPOINT,
) -> dict[str, str]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return {}
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_table_summary_checkpoint(
    summaries: dict[str, str],
    checkpoint_path: str | Path = TABLE_SUMMARIES_CHECKPOINT,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=True, indent=2)


def print_table_summarization_estimate(table_count: int) -> None:
    print(f"Table summarization: ~{table_count:,} tables x ~300 tokens/call")
    print("Estimated cost: ~$0.36 (gpt-4o-mini @ $0.15/M input + $0.60/M output)")
    print("Proceed? [Y/n]")


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=max(int(seconds), 0)))


def summarize_table_with_retry(
    markdown_table: str,
    summarizer: Callable[[str], str],
    table_key: str,
    max_retries: int = TABLE_SUMMARY_MAX_RETRIES,
    initial_backoff_s: int = TABLE_SUMMARY_INITIAL_BACKOFF_S,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return summarizer(markdown_table)
        except Exception as exc:  # pragma: no cover - runtime API failures
            last_error = exc
            print(
                f"[table-summary] {table_key} failed on attempt {attempt}/{max_retries}: {exc}",
                flush=True,
            )
            if attempt == max_retries:
                break
            backoff = initial_backoff_s * (2 ** (attempt - 1))
            print(f"[table-summary] retrying {table_key} in {backoff}s", flush=True)
            time.sleep(backoff)
    raise RuntimeError("Table summarization failed after retries") from last_error


def checkpoint_table_summaries(
    table_documents: list[Document],
    summarizer: Callable[[str], str],
    checkpoint_every: int = TABLE_SUMMARY_CHECKPOINT_EVERY,
    checkpoint_path: str | Path = TABLE_SUMMARIES_CHECKPOINT,
) -> dict[str, str]:
    summaries = load_table_summary_checkpoint(checkpoint_path)
    completed = len(summaries)
    already_completed = completed
    total = len(table_documents)
    pending = total - completed
    start_time = time.time()

    if completed:
        print(
            f"[table-summary] resuming from checkpoint: {completed:,}/{total:,} completed, "
            f"{pending:,} remaining",
            flush=True,
        )
    else:
        print(
            f"[table-summary] starting new run: {total:,} tables to summarize",
            flush=True,
        )

    for document in table_documents:
        source = str(document.metadata.get("source", ""))
        chunk_index = int(document.metadata.get("chunk_index", 0))
        key = f"{source}:table:{chunk_index}"
        if key in summaries:
            continue

        item_start = time.time()
        print(
            f"[table-summary] processing {completed + 1:,}/{total:,} -> {key}",
            flush=True,
        )
        summaries[key] = summarize_table_with_retry(
            document.page_content,
            summarizer,
            table_key=key,
        )
        completed += 1
        item_duration = time.time() - item_start

        if completed % TABLE_SUMMARY_PROGRESS_EVERY == 0 or completed == total:
            elapsed = time.time() - start_time
            finished_this_run = max(completed - already_completed, 1)
            rate = finished_this_run / elapsed if elapsed > 0 else 0.0
            remaining = total - completed
            eta_seconds = remaining / rate if rate > 0 else 0.0
            print(
                f"[table-summary] progress {completed:,}/{total:,} "
                f"({completed / total:.1%}) | last={item_duration:.1f}s | "
                f"rate={rate:.2f} tables/s | eta={_format_duration(eta_seconds)}",
                flush=True,
            )

        if completed % checkpoint_every == 0:
            save_table_summary_checkpoint(summaries, checkpoint_path)
            print(
                f"[table-summary] checkpoint saved at {completed:,}/{total:,}",
                flush=True,
            )

    save_table_summary_checkpoint(summaries, checkpoint_path)
    print(f"[table-summary] checkpoint saved at {completed:,}/{total:,}", flush=True)
    return summaries


def get_embedding_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


def get_table_summarizer() -> BaseChatModel:
    return ChatOpenAI(
        model="gpt-4o-mini",
        timeout=TABLE_SUMMARY_REQUEST_TIMEOUT_S,
        max_retries=0,
    )


def _default_table_summary(markdown_table: str, llm: BaseChatModel | None = None) -> str:
    llm = llm or get_table_summarizer()
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Summarize the following financial table in plain English, "
                    "highlighting key numerical values and their meaning."
                )
            ),
            HumanMessage(content=markdown_table),
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)


def _document_key(document: Document) -> tuple[str, str, int]:
    return (
        str(document.metadata.get("source", "")),
        str(document.metadata.get("type", "")),
        int(document.metadata.get("chunk_index", 0)),
    )


def _tokenize_for_bm25(text: str) -> list[str]:
    return text.lower().split()


def _build_vectorstore(child_documents: list[Document], embeddings: Embeddings) -> FAISS:
    if not child_documents:
        raise ValueError("Cannot build a vector store with no child documents.")
    return FAISS.from_documents(child_documents, embeddings)


def build_hybrid_retriever(
    documents: list[Document],
    *,
    embeddings: Embeddings | None = None,
    table_summaries: dict[str, str] | None = None,
    use_table_summaries: bool = True,
) -> HybridRetriever:
    embeddings = embeddings or get_embedding_model()
    parent_store = InMemoryStore()
    child_documents: list[Document] = []
    bm25_documents: list[Document] = []
    parent_lookup: dict[str, Document] = {}

    for document in documents:
        metadata = dict(document.metadata)
        parent_id = str(uuid.uuid4())
        parent_document = Document(page_content=document.page_content, metadata=metadata)
        parent_lookup[parent_id] = parent_document
        parent_store.mset([(parent_id, parent_document)])

        if metadata["type"] == "table":
            summary_key = f"{metadata['source']}:table:{metadata['chunk_index']}"
            summary_text = (
                table_summaries.get(summary_key, document.page_content)
                if table_summaries
                else document.page_content
            )
            if not use_table_summaries:
                summary_text = document.page_content

            child_documents.append(
                Document(
                    page_content=summary_text,
                    metadata={
                        "type": "table_summary" if use_table_summaries else "table",
                        "source": metadata["source"],
                        "chunk_index": metadata["chunk_index"],
                        "doc_id": parent_id,
                    },
                )
            )
            bm25_documents.append(
                Document(
                    page_content=summary_text,
                    metadata={
                        "type": "table_summary",
                        "source": metadata["source"],
                        "chunk_index": metadata["chunk_index"],
                        "parent_id": parent_id,
                    },
                )
            )
            bm25_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        "type": "table_raw_bm25",
                        "source": metadata["source"],
                        "chunk_index": metadata["chunk_index"],
                        "parent_id": parent_id,
                    },
                )
            )
            continue

        child_documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    "type": metadata["type"],
                    "source": metadata["source"],
                    "chunk_index": metadata["chunk_index"],
                    "doc_id": parent_id,
                },
            )
        )
        bm25_documents.append(parent_document)

    vectorstore = _build_vectorstore(child_documents, embeddings)
    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        id_key="doc_id",
    )
    bm25_corpus = [_tokenize_for_bm25(doc.page_content) for doc in bm25_documents]
    bm25 = BM25Okapi(bm25_corpus if bm25_corpus else [["placeholder"]])

    return HybridRetriever(
        bm25=bm25,
        bm25_docs=bm25_documents if bm25_documents else [],
        multi_vector_retriever=multi_vector_retriever,
        parent_lookup=parent_lookup,
    )


def _resolve_bm25_hits(
    ranked_hits: list[tuple[int, Document]],
    parent_lookup: dict[str, Document],
) -> list[Document]:
    best_by_key: dict[tuple[str, str, int], tuple[int, Document]] = {}

    for rank, document in ranked_hits:
        doc_type = str(document.metadata.get("type", ""))
        if doc_type in {"table_summary", "table_raw_bm25"}:
            parent_id = str(document.metadata["parent_id"])
            resolved = parent_lookup[parent_id]
        else:
            resolved = document

        key = _document_key(resolved)
        current = best_by_key.get(key)
        if current is None or rank < current[0]:
            best_by_key[key] = (rank, resolved)

    return [item[1] for item in sorted(best_by_key.values(), key=lambda x: x[0])]


def hybrid_retrieve(
    query: str,
    *,
    k: int = 3,
    bm25: BM25Okapi,
    bm25_docs: list[Document],
    multi_vector_retriever: MultiVectorRetriever,
    parent_lookup: dict[str, Document],
) -> list[Document]:
    bm25_ranked_hits: list[tuple[int, Document]] = []
    if bm25_docs:
        scores = bm25.get_scores(_tokenize_for_bm25(query))
        sorted_indices = sorted(
            range(len(bm25_docs)),
            key=lambda idx: scores[idx],
            reverse=True,
        )
        top_indices = sorted_indices[: max(k * 3, k)]
        bm25_ranked_hits = [
            (rank, bm25_docs[idx]) for rank, idx in enumerate(top_indices, start=1)
        ]

    bm25_resolved = _resolve_bm25_hits(bm25_ranked_hits, parent_lookup)
    faiss_results = multi_vector_retriever.invoke(query)

    scores_by_key: dict[tuple[str, str, int], float] = {}
    docs_by_key: dict[tuple[str, str, int], Document] = {}

    for rank, document in enumerate(bm25_resolved, start=1):
        key = _document_key(document)
        scores_by_key[key] = scores_by_key.get(key, 0.0) + 0.4 * (1 / (rank + RRF_K))
        docs_by_key[key] = document

    for rank, document in enumerate(faiss_results, start=1):
        key = _document_key(document)
        scores_by_key[key] = scores_by_key.get(key, 0.0) + 0.6 * (1 / (rank + RRF_K))
        docs_by_key[key] = document

    ranked = sorted(scores_by_key.items(), key=lambda item: item[1], reverse=True)
    return [docs_by_key[key] for key, _ in ranked[:k]]


def _serialize_parent_store(parent_store: InMemoryStore) -> dict[str, Any]:
    return dict(parent_store.store)


def _load_parent_store(records: dict[str, Any]) -> InMemoryStore:
    parent_store = InMemoryStore()
    parent_store.store = dict(records)
    return parent_store


def save_hybrid_retriever(
    retriever: HybridRetriever,
    index_dir: str | Path = FAISS_INDEX_DIR,
    docstore_path: str | Path = DOCSTORE_FILE,
) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(retriever.multi_vector_retriever.vectorstore.index, str(FAISS_INDEX_FILE))

    payload = {
        "bm25_docs": retriever.bm25_docs,
        "parent_lookup": retriever.parent_lookup,
        "vectorstore_docstore": dict(
            retriever.multi_vector_retriever.vectorstore.docstore._dict
        ),
        "index_to_docstore_id": dict(
            retriever.multi_vector_retriever.vectorstore.index_to_docstore_id
        ),
        "parent_store": _serialize_parent_store(retriever.multi_vector_retriever.docstore),
    }
    with open(docstore_path, "wb") as f:
        pickle.dump(payload, f)


def load_hybrid_retriever(
    embeddings: Embeddings | None = None,
    index_dir: str | Path = FAISS_INDEX_DIR,
    docstore_path: str | Path = DOCSTORE_FILE,
) -> HybridRetriever:
    embeddings = embeddings or get_embedding_model()
    index = faiss.read_index(str(FAISS_INDEX_FILE))

    with open(docstore_path, "rb") as f:
        payload = pickle.load(f)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(payload["vectorstore_docstore"]),
        index_to_docstore_id=payload["index_to_docstore_id"],
    )
    parent_store = _load_parent_store(payload["parent_store"])
    multi_vector_retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=parent_store,
        id_key="doc_id",
    )
    bm25_docs: list[Document] = payload["bm25_docs"]
    bm25_tokens = [_tokenize_for_bm25(doc.page_content) for doc in bm25_docs]
    bm25 = BM25Okapi(bm25_tokens if bm25_tokens else [["placeholder"]])

    return HybridRetriever(
        bm25=bm25,
        bm25_docs=bm25_docs,
        multi_vector_retriever=multi_vector_retriever,
        parent_lookup=payload["parent_lookup"],
    )


def build_global_hybrid_retriever(
    *,
    data_dir: str | Path = DATA_DIR,
    embeddings: Embeddings | None = None,
    summarizer_llm: BaseChatModel | None = None,
) -> HybridRetriever:
    ensure_data_directories()
    samples = load_all_samples(data_dir)
    documents = [doc for sample in samples for doc in preprocess_sample(sample)]
    table_documents = [doc for doc in documents if doc.metadata["type"] == "table"]

    print_table_summarization_estimate(len(table_documents))
    proceed = input().strip().lower()
    if proceed not in {"", "y", "yes", "n", "no"}:
        proceed = "y"
    if proceed in {"n", "no"}:
        raise RuntimeError("Ingestion cancelled by user.")

    summarizer_llm = summarizer_llm or get_table_summarizer()
    summaries = checkpoint_table_summaries(
        table_documents,
        summarizer=lambda markdown: _default_table_summary(markdown, summarizer_llm),
    )
    retriever = build_hybrid_retriever(
        documents,
        embeddings=embeddings,
        table_summaries=summaries,
        use_table_summaries=True,
    )
    save_hybrid_retriever(retriever)
    return retriever


def build_eval_sample_retriever(
    sample: dict[str, Any],
    embeddings: Embeddings | None = None,
) -> HybridRetriever:
    documents = preprocess_sample(sample)
    return build_hybrid_retriever(
        documents,
        embeddings=embeddings,
        use_table_summaries=False,
    )


def build_pooled_eval_retriever(
    samples: list[dict[str, Any]],
    embeddings: Embeddings | None = None,
) -> HybridRetriever:
    documents = [doc for sample in samples for doc in preprocess_sample(sample)]
    return build_hybrid_retriever(
        documents,
        embeddings=embeddings,
        use_table_summaries=False,
    )


def load_or_build_index() -> HybridRetriever:
    ensure_data_directories()
    if should_skip_ingestion():
        return load_hybrid_retriever()
    return build_global_hybrid_retriever()
