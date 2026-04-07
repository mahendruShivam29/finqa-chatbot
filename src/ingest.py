import json
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken


DATA_DIR = Path("data")
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
TABLE_SUMMARIES_CHECKPOINT = DATA_DIR / "table_summaries_checkpoint.json"
DATASET_FILES = ("train.json", "dev.json", "test.json")
ENCODING = tiktoken.get_encoding("cl100k_base")


def ensure_data_directories() -> None:
    """Create the local data and source directory structure required by the SDD."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_json_file(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_all_samples(data_dir: str | Path = DATA_DIR) -> list[dict[str, Any]]:
    """Load train/dev/test FinQA samples from disk using Python's json library."""
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
    """Apply the SDD chunking rules: paragraph-level text chunks and one intact table."""
    sample_id = str(sample["id"])
    paragraphs = [*sample.get("pre_text", []), *sample.get("post_text", [])]

    documents = _paragraph_documents(paragraphs, sample_id)
    table_markdown = table_to_markdown(sample.get("table", []))
    documents.append(
        Document(
            page_content=table_markdown,
            metadata={"type": "table", "source": sample_id, "chunk_index": 0},
        )
    )
    return documents


def should_skip_ingestion(index_dir: str | Path = FAISS_INDEX_DIR) -> bool:
    return Path(index_dir).exists()


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


def summarize_table_with_retry(
    markdown_table: str,
    summarizer,
    max_retries: int = 3,
    initial_backoff_s: int = 2,
) -> str:
    """
    Summarize a table with retry/backoff.

    The actual summarizer implementation is wired in Step 2 when the retrieval
    stack is built. This helper enforces the Step 1 batching/retry contract.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return summarizer(markdown_table)
        except Exception as exc:  # pragma: no cover - network/API failures are runtime concerns
            last_error = exc
            if attempt == max_retries - 1:
                break
            time.sleep(initial_backoff_s * (2**attempt))
    raise RuntimeError("Table summarization failed after retries") from last_error


def checkpoint_table_summaries(
    table_documents: list[Document],
    summarizer,
    checkpoint_every: int = 100,
    checkpoint_path: str | Path = TABLE_SUMMARIES_CHECKPOINT,
) -> dict[str, str]:
    """Resumeable table summarization for the global index ingestion flow."""
    summaries = load_table_summary_checkpoint(checkpoint_path)

    for idx, document in enumerate(table_documents, start=1):
        source = str(document.metadata.get("source", ""))
        chunk_index = int(document.metadata.get("chunk_index", 0))
        key = f"{source}:table:{chunk_index}"
        if key in summaries:
            continue

        summaries[key] = summarize_table_with_retry(document.page_content, summarizer)
        if idx % checkpoint_every == 0:
            save_table_summary_checkpoint(summaries, checkpoint_path)

    save_table_summary_checkpoint(summaries, checkpoint_path)
    return summaries


def load_or_build_index():
    """
    Step 1 persistence gate.

    If the persisted FAISS index already exists, later steps will load it from
    disk. Otherwise, later steps will run the full ingestion pipeline.
    """
    ensure_data_directories()
    if should_skip_ingestion():
        return None
    return None
