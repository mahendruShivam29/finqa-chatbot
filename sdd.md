# Software Design Document (SDD): FinQA Agentic-RAG Chatbot — v2

---

## Revision Notes (v1 → v2)

| # | Issue | Fix Applied |
|---|---|---|
| 1 | **Retrieval Precision@3 formula was wrong** — dividing by `3 × num_samples` ignores variable gold set sizes | Changed to standard Precision and Recall computed per-sample, then macro-averaged |
| 2 | **Per-sample retrieval makes retrieval eval trivially easy** (~5-10 docs per index) | Added explicit limitation acknowledgment + optional "pooled hard mode" eval variant |
| 3 | **Percentage normalization fallback was too permissive** — chained ×100/÷100 could accept answers off by 10,000× | Restricted to a single normalization step; added guard against double-normalization |
| 4 | **No guard for empty code extraction in `reason_and_code`** | Added explicit empty-code guard that increments `error_count` |
| 5 | **`generate_final_answer` didn't receive `retrieved_context` or `generated_code`** | Updated node to include both in its prompt |
| 6 | **BM25 dedup for table parent resolution unclear** | Added explicit dedup logic description in hybrid retrieve |
| 7 | **ASCII architecture diagram confusing** | Simplified |

---

## 1. Executive Summary

| Field | Detail |
|---|---|
| **Objective** | Build a terminal-based Question-Answering chatbot capable of performing complex numerical reasoning over financial documents using the FinQA dataset. |
| **Core Tech Stack** | Python 3.10+, GPT-4o-mini (primary) / GPT-4o (judge only), LangChain, LangGraph, LangSmith (tracing), FAISS, BM25, PythonREPLTool |
| **Methodology** | Agentic-RAG (Retrieval-Augmented Generation with Tool Use) — Hybrid Search, Multi-Vector Retrieval, and deterministic Python code execution |
| **Interface** | Terminal / IDE only — no web UI |
| **Budget Constraint** | Total OpenAI spend must remain under **$5**. All design decisions reflect this hard ceiling. |

### Cost-Aware Model Strategy

| Role | Model | Rationale |
|---|---|---|
| **Reasoning & Code Generation** | `gpt-4o-mini` | ~$0.15/M input, ~$0.60/M output — 20× cheaper than GPT-4o with strong code generation |
| **Table Summarization (ingestion)** | `gpt-4o-mini` | Keeps ingestion cost under $0.50 for all samples |
| **LLM-as-a-Judge (10 samples)** | `gpt-4o` | Higher quality judgment, but limited to 10 calls (~$0.05) |
| **Final Answer Generation** | `gpt-4o-mini` | Simple natural-language formatting — no need for expensive model |
| **Embeddings** | `text-embedding-3-small` | $0.02/M tokens — cheapest OpenAI embedding |

**Estimated Total Cost Breakdown:**

| Component | Est. Calls | Est. Cost |
|---|---|---|
| Table summaries (ingestion, ~8K tables) | ~8,000 | ~$0.40 |
| Eval: reasoning + code (100 samples, avg 1.5 attempts) | ~150 | ~$0.30 |
| Eval: final answer generation (100 samples) | 100 | ~$0.10 |
| Eval: LLM-as-a-Judge (10 samples, GPT-4o) | 10 | ~$0.05 |
| Interactive demo usage | ~50 | ~$0.10 |
| Embeddings (all ingestion) | 1 | ~$0.02 |
| **Total** | | **~$0.97** |

This leaves a ~$4 safety margin for retries, iteration, and debugging.

> **Architecture Note on Production Serving:** In a production GPU environment, the reasoning model can be replaced with a self-hosted open-source model via vLLM (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`), served on an OpenAI-compatible endpoint. The LangGraph node code is identical — only the `base_url` and `model` parameters change. On a CPU-only laptop, Ollama provides the same abstraction. Neither is required for this implementation since we use the OpenAI API directly.

---

## 2. Key Questions & Architectural Decisions

### 2.1 Data Understanding

**Dataset Source:** The FinQA dataset is available at <https://github.com/czyssrs/FinQA> under the `dataset/` directory. The three required files are `train.json`, `dev.json`, and `test.json`. Clone or download these files into the `data/` directory.

**Characteristics:** The FinQA dataset consists of earnings reports from S&P 500 companies, each containing both unstructured text paragraphs and structured tables. Questions require multi-step numerical reasoning (addition, subtraction, division, percentages, etc.) that spans both modalities.

**Assumptions:**

- The provided financial text and tables for a given question contain all variables necessary to compute the answer.
- Tables can be serialized into Markdown format without losing row/column spatial meaning, provided they are never split mid-table.
- The ground truth numerical answer is stored in `sample["qa"]["exe_ans"]` as a float (or occasionally a string with `%`).

**What Makes Financial QA Unique vs. General QA:**

- General QA extracts or summarizes text. Financial QA requires exact mathematical computation — a hallucinated number is catastrophically wrong, not just imprecise.
- Answers require bridging tabular data and textual context to identify the correct variables and formulate a correct equation.
- The FinQA ground truth uses a domain-specific language (DSL) for programs (e.g., `subtract(5023, 4765)`), which cannot be used for direct code comparison against standard Python. Evaluation must focus on numerical answer matching, not code matching.

### 2.2 Retrieval Scope Decision — CRITICAL

**Problem:** FinQA questions are designed to be answerable from their *own* sample context (`pre_text`, `post_text`, `table`). Building a single global index across thousands of samples from unrelated companies would make retrieval extremely noisy and tank precision.

**Decision: Per-Sample Scoped Retrieval**

We adopt a **per-sample retrieval** architecture. For each incoming question during evaluation, the retriever searches only within the documents belonging to that question's sample (its `pre_text`, `post_text`, and `table`). This mirrors the FinQA task setup and isolates the retrieval challenge to *selecting the right chunks within a relevant report*, not finding the right report among thousands.

**Limitation Acknowledgment:** Per-sample indexes are small (~5-10 documents), making retrieval precision artificially high compared to a real-world setting. This is an intentional design choice that mirrors the FinQA benchmark protocol. To provide a harder retrieval signal, an optional "pooled hard mode" is included in eval (see Step 5f).

**For the interactive terminal mode**, because we cannot know which sample the user's question belongs to, we build a **global index** over all samples and retrieve from it. The terminal mode is a best-effort demo; the evaluation mode is the authoritative measurement.

**Implementation:**

- `eval.py`: For each sample, build a temporary in-memory FAISS + BM25 index over only that sample's documents. Retrieve from it. This is the measured configuration.
- `main.py`: Use the persisted global index for interactive questions. Accept that retrieval precision will be lower in this mode.

### 2.3 Method Selection

**Approaches Considered:**

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Standard RAG** | Simple to implement | LLMs are unreliable arithmetic engines; multi-step financial calculations hallucinate frequently | ❌ Rejected |
| **Fine-Tuning** | Better domain tone and familiarity | Does not eliminate arithmetic hallucinations; expensive and slow to iterate; well exceeds $5 budget | ❌ Rejected |
| **Prompt Engineering (CoT)** | Improves step-by-step reasoning | Still relies on the LLM's internal calculator, error-prone for multi-step calculations | ❌ Rejected |
| **Agentic-RAG + Code Execution** | Separates reasoning from computation; LLM writes code, Python executes deterministically; eliminates arithmetic hallucination | More complex graph; requires sandboxing in production | ✅ Chosen |

**Chosen Approach:** Agentic-RAG via LangGraph.

**Rationale:** LangGraph enables a fault-tolerant cyclic graph (`Retrieve → Reason_and_Code → Execute → Check → Answer`) with explicit retry logic and graceful failure handling. The LLM acts as a reasoning and code-writing agent; the Python REPL acts as a deterministic calculator. GPT-4o-mini is a strong code generator at a fraction of GPT-4o cost, making this approach viable within the $5 budget.

**Security Consideration:** The Python execution tool must be sandboxed in production to prevent Arbitrary Code Execution (ACE) vulnerabilities. This demo uses LangChain's `PythonREPLTool` locally. All code comments must note that in production this is replaced by E2B (`e2b_code_interpreter`) or a Dockerized execution container.

### 2.4 Evaluation Strategy

**Primary Metric — Execution Accuracy / Exact Match (EM):**

- Evaluation focuses exclusively on the final numerical answer.
- A prediction is correct if: `abs(predicted - ground_truth) <= 1e-4`.
- Ground truth is read from `sample["qa"]["exe_ans"]`.
- The predicted answer is extracted from the agent's `final_answer` string using the regex pattern `r"[-+]?\d+\.?\d*|[-+]?\.\d+"`, taking the **last** match as the primary numerical result.
- **Single-step percentage normalization:** If the first direct comparison fails, attempt **exactly one** normalization step. If the predicted string contains `%` immediately after the matched number, the value has already been divided by 100 during extraction — do NOT divide again. The fallback checks are:
  - `abs(predicted * 100 - ground_truth) <= tol`
  - `abs(predicted / 100 - ground_truth) <= tol`

  No further chaining is applied. This prevents false positives from double-normalization (e.g., accepting an answer that is off by 10,000×).

**Secondary Metric — LLM-as-a-Judge:**

- On a 10-sample subset (to manage API cost — uses GPT-4o at ~$0.05 total), use GPT-4o to judge whether the agent's retrieved context and reasoning steps are logically consistent with the ground truth reasoning.
- Prompt: *"Given this question, ground truth answer, and agent reasoning, did the agent retrieve the correct data and reason correctly? Answer Yes or No with one sentence of justification."*
- Report the Yes/No ratio as the **Reasoning Quality Score**.

**Other Metrics:**

| Metric | Method |
|---|---|
| **Retrieval Precision (macro-avg)** | Per sample: `hits / len(retrieved_docs)`. Averaged across all eval samples. |
| **Retrieval Recall (macro-avg)** | Per sample: `hits / len(gold_keys)`. Averaged across all eval samples. |
| **Average Latency** | Wall-clock time per question, averaged over all 100 eval samples. |
| **Token Cost** | Total tokens consumed during eval (tracked via LangSmith). |

### 2.5 Production & Monitoring Plan

**Monitoring with LangSmith:**

- Every LangGraph execution is traced end-to-end in LangSmith, capturing: node execution order, token counts per node, tool inputs/outputs, execution errors, and total latency.
- Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in the environment.
- Tag each run with `project="finqa-agentic-rag"`.

**Drift Detection:**

- Monitor **Tool Error Rate** in LangSmith: if the Python REPL failure rate rises above 10%, it indicates the LLM is generating malformed code, likely due to input document format drift.
- Monitor EM accuracy on a weekly holdout sample. A drop of more than 5 percentage points triggers a review.

**Production Serving (vLLM):**

- In a GPU production environment, replace the OpenAI API with vLLM: `python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct`. The `base_url` changes from `https://api.openai.com/v1` to `http://localhost:8000/v1`. No other code changes required.
- Embeddings in production would be served via Text Embeddings Inference (TEI) or Infinity, replacing the OpenAI embedding API call.

---

## 3. Implementation Instructions

> **AI INSTRUCTIONS:** Implement the following system step-by-step in Python. Do not build a web UI. The interface is terminal-only. Follow every instruction exactly — do not make architectural decisions not specified here.

### Step 1: Environment & Data Setup

#### 1a. Dependencies

Create `requirements.txt`:

```
langchain
langchain-openai
langchain-community
langchain-experimental
langgraph
langsmith
openai
pandas
faiss-cpu
rank_bm25
tiktoken
python-dotenv
```

#### 1b. Data Download

- Clone or download the FinQA dataset from <https://github.com/czyssrs/FinQA>.
- Place `train.json`, `dev.json`, and `test.json` into the `data/` directory.
- Write a data loader function in `ingest.py` that reads these JSON files using Python's `json` library and returns a list of sample dicts.

#### 1c. Chunking Strategy — CRITICAL

Implement a `preprocess_sample(sample)` function that applies the following rules strictly:

**Text paragraphs — Paragraph-Level Chunking (no splitting):**

- Extract the list from `sample["pre_text"]` and `sample["post_text"]`.
- Treat each paragraph as its own `Document` object. **Do not concatenate and re-split.** Most FinQA paragraphs are under 512 tokens. Only if a single paragraph exceeds 512 tokens, split it using `RecursiveCharacterTextSplitter` with `chunk_size=512` and `chunk_overlap=50`, and assign all resulting sub-chunks the same `chunk_index` as the original paragraph.
- Each chunk becomes a `Document` object with metadata:
  ```python
  {"type": "text", "source": sample["id"], "chunk_index": i}
  ```
  where `i` is the positional index of the original paragraph in the concatenated `pre_text + post_text` list. This directly maps to `gold_inds` keys like `"text_3"`.

**Why paragraph-level:** This ensures a 1:1 mapping between document chunks and `gold_inds` keys. Merge-then-split strategies create chunk boundary misalignment that inflates false negatives in retrieval evaluation.

**Tables: DO NOT CHUNK.**

- Extract `sample["table"]` (a list of lists).
- Convert the entire table into a single Markdown string using a helper function `table_to_markdown(table)` that formats the first row as the header and subsequent rows as data rows.
- Keep the entire Markdown table as **one single** `Document` object with metadata:
  ```python
  {"type": "table", "source": sample["id"], "chunk_index": 0}
  ```
- Never split a table across chunks.

#### 1d. Persistence Check — CRITICAL

Before running ingestion, check if `data/faiss_index/` exists. If it does, skip ingestion entirely and load from disk. Only run the full ingestion pipeline (including table summarization) if the index does not exist. This prevents re-running expensive API calls on every launch.

```python
import os
if os.path.exists("data/faiss_index"):
    # load index from disk
else:
    # run full ingestion and save to disk
```

#### 1e. Table Summarization — Batching & Checkpointing

Table summarization is needed **only for the global index** (interactive terminal mode). The eval mode builds per-sample indexes on the fly and can use the raw table markdown directly without summarization (see Step 5c).

Because table summarization via GPT-4o-mini is moderately expensive (~8K calls), implement the following safeguards:

- **Checkpointing:** After every 100 tables, write intermediate results to `data/table_summaries_checkpoint.json`. If ingestion is restarted, load the checkpoint and resume from the last completed batch.
- **Rate limit handling:** Wrap each API call in a retry loop with exponential backoff (max 3 retries, starting at 2s).
- **Cost estimation:** Before starting, print the estimated number of API calls and approximate cost. Example output:
  ```
  Table summarization: ~8,000 tables × ~300 tokens/call
  Estimated cost: ~$0.36 (gpt-4o-mini @ $0.15/M input + $0.60/M output)
  Proceed? [Y/n]
  ```
- **Model:** Use `gpt-4o-mini` for all table summarization. Do NOT use GPT-4o for this step.

---

### Step 2: Vector Store & Retrieval (in `ingest.py`)

#### 2a. Embedding Model

Use `OpenAIEmbeddings(model="text-embedding-3-small")` for all embeddings.

#### 2b. Multi-Vector Retriever for Tables

Implement the following parent-child mapping pattern for table documents:

1. For each table `Document`, call `gpt-4o-mini` with the prompt:
   > "Summarize the following financial table in plain English, highlighting key numerical values and their meaning:\n\n{markdown_table}"
2. Store the summary as a new `Document` (the "child").
3. Embed the **summary** Document into FAISS.
4. Store the **original full Markdown table** Document in an `InMemoryByteStore` (LangChain's `InMemoryStore`), keyed by a UUID.
5. Map the FAISS embedding to the UUID of the original table using LangChain's `MultiVectorRetriever`.

This means retrieval returns the **full raw Markdown table**, not just the summary.

#### 2c. Hybrid Search — Two-Stage Merge

> **Design Note:** `EnsembleRetriever` cannot directly wrap a `MultiVectorRetriever` because BM25 returns summary documents while MultiVector returns parent (raw table) documents. We use a **two-stage merge** instead.

**BM25 Indexing Strategy:** BM25 indexes **both** the raw Markdown table **and** the table summary for each table document, in addition to all text chunks. This ensures keyword matches against financial terms in tables are not lost. Each BM25 document carries metadata linking it back to its parent (the raw table or the text chunk itself).

**Implementation:**

```python
def hybrid_retrieve(query: str, k: int = 3) -> list[Document]:
    """
    Stage 1: BM25 retrieves from ALL docs (text chunks + table summaries + raw table markdown).
             For table hits (type=="table_summary" or type=="table_raw_bm25"),
             resolve to the parent raw table document using the parent_id in metadata.
    Stage 2: FAISS via MultiVectorRetriever retrieves parent docs
             (raw tables for table hits, text chunks for text hits).
    Merge:   Combine results, deduplicate by (source, type, chunk_index) tuple,
             apply reciprocal rank fusion (RRF) with weights BM25=0.4, FAISS=0.6.
    """
```

- **BM25 retriever:** Built over all text chunks, table summary documents, and raw table markdown documents.
- **FAISS MultiVectorRetriever:** Built over embedded summaries, returns parent documents (raw tables).
- **BM25 parent resolution & dedup:** When a BM25 hit has `metadata["type"]` in `{"table_summary", "table_raw_bm25"}`, resolve it to the parent raw table document via `metadata["parent_id"]`. If multiple BM25 hits resolve to the same parent (e.g., both the summary and raw markdown of the same table match), keep only the **highest-ranked** hit for RRF scoring. Dedup key is the tuple `(metadata["source"], "table", metadata["chunk_index"])`.
- **Merge via RRF:** For each document appearing in either result set, compute a fused score: `score = 0.4 * (1 / (bm25_rank + 60)) + 0.6 * (1 / (faiss_rank + 60))`. Return the top-K by fused score.

#### 2d. Persistence

After building the FAISS index, save it using `faiss.write_index()` and save the `InMemoryStore` contents and BM25 corpus using `pickle`. On subsequent launches, load both from disk to skip ingestion.

> **Note:** `pickle` is not portable across Python versions. This is acceptable for a local demo. A production system would use a proper document store (e.g., Redis, PostgreSQL with pgvector).

#### 2e. Retriever Tool (for terminal/interactive mode only)

Wrap the `hybrid_retrieve` function using `create_retriever_tool` with:

- `name="financial_document_retriever"`
- `description="Searches and retrieves relevant financial text paragraphs and tables from earnings reports. Use this tool first to find the data needed to answer a financial question."`

---

### Step 3: Tool Creation (in `tools.py`)

#### 3a. Python REPL Tool

```python
from langchain_experimental.tools import PythonREPLTool

# PRODUCTION NOTE: In a production environment, replace PythonREPLTool with
# a sandboxed execution environment such as E2B (e2b_code_interpreter) or a
# Dockerized container to prevent Arbitrary Code Execution (ACE) vulnerabilities.

python_repl = PythonREPLTool()
python_repl.description = (
    "Use this tool to execute Python code for mathematical calculations. "
    "Pass the exact Python code to run as a string. "
    "Always print() the final result so it appears in the output. "
    "Use this for all arithmetic — never compute numbers in your head."
)
```

#### 3b. Tool List

Export a list: `tools = [financial_document_retriever, python_repl]`

---

### Step 4: LangGraph Agentic Workflow (in `graph.py`)

#### 4a. State Definition

```python
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: str
    generated_code: str
    execution_result: str
    error_count: int
    final_answer: str
```

#### 4b. Node Definitions

Implement the following four nodes:

**Node 1: `retrieve`**

- Takes the last `HumanMessage` from `state["messages"]` as the query.
- Calls `financial_document_retriever` (interactive mode) or the per-sample scoped retriever (eval mode — see Step 5).
- Concatenates all retrieved document page contents into a single string.
- **Empty retrieval guard:** If zero documents are retrieved, set `retrieved_context` to `"NO RELEVANT DOCUMENTS FOUND."` and let the pipeline continue. The `reason_and_code` node will see this and the `generate_final_answer` node will produce an appropriate "unable to answer" response.
- Returns `{"retrieved_context": combined_context}`.

**Node 2: `reason_and_code`**

- Constructs a prompt combining `state["retrieved_context"]` and the original question from `state["messages"][0].content`.
- If `state["retrieved_context"] == "NO RELEVANT DOCUMENTS FOUND."`, prepend to the prompt:
  > "WARNING: No relevant documents were retrieved. Attempt to answer only if you have enough information from the question itself. Otherwise, print 'UNABLE_TO_ANSWER'."
- If `state["error_count"] > 0`, append the previous error to the prompt:
  > "Your previous code failed with: {state['execution_result']}. Fix the error and try again."
- **LLM call:** Use `gpt-4o-mini` as the primary model.
  ```python
  response = gpt4o_mini_client.invoke(prompt)
  # gpt4o_mini_client = ChatOpenAI(model="gpt-4o-mini")
  ```
- System prompt:
  > "You are a financial analyst assistant. Given financial context and a question, write a Python script that calculates the exact numerical answer. Always end your script with a print() statement showing the result. Return ONLY the Python code, no explanation."
- Extracts the code block from the response (strip markdown fences if present).
- **Empty code guard:** If the extracted code string is empty or contains only whitespace after stripping fences, do NOT pass it to the execute node. Instead, return:
  ```python
  {
      "generated_code": "",
      "execution_result": "ERROR: LLM returned empty code block. No code to execute.",
      "error_count": state["error_count"] + 1,
  }
  ```
  The conditional edge from `execute` will then route to retry (if under limit) or `generate_final_answer`.
- Returns `{"generated_code": code, "messages": [AIMessage(content=code)]}`.

**Node 3: `execute`**

- Runs `state["generated_code"]` through the `PythonREPLTool`.
- If execution succeeds: returns `{"execution_result": output}`.
- If execution raises an exception: returns `{"execution_result": str(error), "error_count": state["error_count"] + 1}`.

**Node 4: `generate_final_answer`**

- If `state["error_count"] >= 3`, the answer is:
  > "I was unable to compute the answer after 3 attempts. Last error: {state['execution_result']}"
- Otherwise, calls `gpt-4o-mini` with:
  > "Given this question: {question}\n\nRetrieved context:\n{state['retrieved_context']}\n\nGenerated code:\n{state['generated_code']}\n\nComputation result: {state['execution_result']}\n\nWrite a clear, concise natural language answer. State the numerical result explicitly."
- Returns `{"final_answer": answer_string}`.

#### 4c. Graph Edges

```
START → retrieve → reason_and_code

Conditional edge from reason_and_code:
  if generated_code is empty → route as if execute returned error (use error_count to decide retry or final answer)
  else → execute

Conditional edge from execute:
  if no error → generate_final_answer → END
  if error AND error_count < 3 → reason_and_code  (retry loop)
  if error AND error_count >= 3 → generate_final_answer → END
```

#### 4d. Graph Compilation

```python
graph = builder.compile()
# Export as compiled_graph for use in main.py and eval.py
```

---

### Step 5: Evaluation Script (in `eval.py`)

#### 5a. Data Loading

- Load `data/dev.json`.
- Take the first 100 samples.
- Accept an optional `--samples N` CLI argument (default 100) so developers can test with a small subset first:
  ```python
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--samples", type=int, default=100)
  args = parser.parse_args()
  samples = samples[:args.samples]
  ```

#### 5b. Ground Truth Parsing

For each sample, extract the ground truth:

```python
raw = sample["qa"]["exe_ans"]
if isinstance(raw, str):
    raw = raw.strip()
    if raw.endswith("%"):
        ground_truth = float(raw.rstrip("%")) / 100.0
    else:
        ground_truth = float(raw)
else:
    ground_truth = float(raw)
```

#### 5c. Per-Sample Scoped Retrieval (eval mode only)

For each sample, build a **temporary** in-memory retrieval index:

1. Run `preprocess_sample(sample)` to get text chunks and table document.
2. Build a temporary FAISS index + BM25 index over only these documents. **No table summarization is needed in eval mode** — embed and index the raw Markdown table directly. This saves ~100 GPT-4o-mini calls and keeps eval cost near-zero for embeddings.
3. The `retrieve` node in eval mode uses this scoped retriever instead of the global one.

This ensures evaluation measures reasoning quality, not cross-report retrieval noise.

**Limitation:** Per-sample indexes contain only ~5-10 documents. Retrieval precision will be artificially high because the search space is tiny. This is consistent with the FinQA benchmark protocol (questions are evaluated against their own context), but does not demonstrate the hybrid retriever's ability to discriminate across a large corpus.

#### 5d. Agent Invocation

For each sample:

```python
question = sample["qa"]["question"]
result = compiled_graph.invoke({
    "messages": [HumanMessage(content=question)],
    "error_count": 0,
})
final_answer_text = result["final_answer"]
```

#### 5e. Answer Extraction with Single-Step Percentage Normalization

```python
import re

def extract_predicted_answer(text: str) -> float | None:
    """Extract the last number from the answer text, normalizing percentages."""
    # Find all numbers, possibly followed by %
    matches = re.findall(r"([-+]?\d+\.?\d*|[-+]?\.\d+)\s*(%)?", text)
    if not matches:
        return None
    last_num, pct_sign = matches[-1]
    value = float(last_num)
    if pct_sign == "%":
        value /= 100.0
    return value

def check_answer(predicted: float, ground_truth: float, tol: float = 1e-4) -> bool:
    """
    Check with SINGLE-STEP percentage normalization only.

    Attempts:
      1. Direct match:       |pred - gt| <= tol
      2. pred was fraction:  |pred * 100 - gt| <= tol   (e.g., pred=0.15, gt=15.0)
      3. pred was whole:     |pred / 100 - gt| <= tol   (e.g., pred=15.0, gt=0.15)

    No further chaining (×100 then ÷100 or vice versa) to prevent
    false positives from double-normalization.
    """
    if abs(predicted - ground_truth) <= tol:
        return True
    if abs(predicted * 100 - ground_truth) <= tol:
        return True
    if abs(predicted / 100 - ground_truth) <= tol:
        return True
    return False
```

#### 5f. Metrics Calculation

**EM with tolerance and single-step percentage normalization:**

```python
correct = check_answer(predicted, ground_truth)
```

Accumulate and report: `EM_accuracy = correct_count / num_samples`.

**Retrieval Precision & Recall (macro-averaged):**

For each eval sample (using per-sample scoped retrieval), retrieve top 3 documents. Compare each retrieved document's `metadata["type"]` + `metadata["chunk_index"]` against the keys in `sample["qa"]["gold_inds"]`:

```python
# gold_inds keys look like "text_3", "table_0"
gold_keys = set(sample["qa"]["gold_inds"].keys())
retrieved_keys = {
    f"{doc.metadata['type']}_{doc.metadata['chunk_index']}"
    for doc in retrieved_docs
}
hits = len(gold_keys & retrieved_keys)

# Per-sample metrics
precision_i = hits / len(retrieved_keys) if retrieved_keys else 0.0
recall_i = hits / len(gold_keys) if gold_keys else 0.0
```

Report macro-averaged:
- `Retrieval_Precision = mean(precision_i for all samples)`
- `Retrieval_Recall = mean(recall_i for all samples)`

**Optional — Pooled Hard Mode (10-sample clusters):**

To stress-test the hybrid retriever, optionally pool 10 consecutive samples into a single index and run retrieval against that larger corpus. Report separately as `Pooled_Precision` and `Pooled_Recall`. This is not the primary metric but provides a more realistic retrieval difficulty signal.

```python
parser.add_argument("--pooled-eval", action="store_true", help="Run additional pooled retrieval eval")
```

**Average Latency:** Use `time.time()` before and after each `compiled_graph.invoke()`. Report mean in seconds.

**LLM-as-a-Judge (10-sample subset):**

On the first 10 samples only, send to **GPT-4o** (this is the one place we use the more expensive model):

> "Question: {question}\nGround Truth Answer: {ground_truth}\nAgent Reasoning and Code: {generated_code}\nAgent Final Answer: {final_answer_text}\n\nDid the agent retrieve the correct data and reason correctly to arrive at its answer? Reply with Yes or No followed by one sentence of justification."

Report the Yes ratio as `Reasoning_Quality_Score`.

#### 5g. Results Logging

Write each sample's result to `data/eval_results.jsonl` so failures can be inspected without re-running:

```python
import json

with open("data/eval_results.jsonl", "a") as f:
    f.write(json.dumps({
        "sample_id": sample["id"],
        "question": question,
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": correct,
        "final_answer": final_answer_text,
        "generated_code": result.get("generated_code", ""),
        "error_count": result.get("error_count", 0),
        "latency_s": latency,
        "precision": precision_i,
        "recall": recall_i,
    }) + "\n")
```

#### 5h. Terminal Output

Print the following to the terminal upon completion:

```
=== FinQA Evaluation Results ({N} samples) ===
Exact Match Accuracy (±1e-4):     XX.X%
Retrieval Precision (macro-avg):  XX.X%
Retrieval Recall (macro-avg):     XX.X%
Reasoning Quality Score (LLM-as-a-Judge, n=10): X/10
Average Latency per Question:     X.XXs
Detailed results saved to:        data/eval_results.jsonl
```

---

### Step 6: Terminal Interface & LangSmith Tracing (in `main.py`)

#### 6a. Environment Setup

Load the following from a `.env` file using `python-dotenv`:

```
OPENAI_API_KEY=your_openai_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=finqa-agentic-rag
```

#### 6b. Auto-Ingestion on First Run

On startup, check for the persisted global index. If it does not exist, run the full ingestion pipeline automatically before entering the interactive loop. Do not require a separate `python src/ingest.py` step.

```python
from ingest import load_or_build_index

retriever = load_or_build_index()  # returns the global hybrid retriever
```

#### 6c. Terminal Loop

```python
while True:
    question = input("\nEnter a financial question (or 'quit' to exit): ").strip()
    if question.lower() == "quit":
        break

    print("\n[1/4] Retrieving relevant financial documents...")
    for step in compiled_graph.stream(
        {"messages": [HumanMessage(content=question)], "error_count": 0}
    ):
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        if node_name == "retrieve":
            ctx = node_output.get("retrieved_context", "")
            print(f"[2/4] Retrieved context ({len(ctx)} chars)")
        elif node_name == "reason_and_code":
            print(f"[3/4] Generated code:\n{node_output.get('generated_code', '')}")
        elif node_name == "execute":
            print(f"      Execution result: {node_output.get('execution_result', '')}")
        elif node_name == "generate_final_answer":
            print(f"\n[4/4] Answer: {node_output.get('final_answer', '')}")
```

---

## 4. Repository Structure

```
finqa-agentic-rag/
├── data/
│   ├── train.json              # FinQA dataset (downloaded separately)
│   ├── dev.json
│   ├── test.json
│   ├── faiss_index/            # Auto-generated on first run
│   ├── docstore.pkl            # Auto-generated on first run
│   ├── table_summaries_checkpoint.json  # Ingestion checkpoint
│   └── eval_results.jsonl      # Generated by eval.py
├── src/
│   ├── ingest.py       # Preprocessing, chunking, multi-vector retriever,
│   │                   # hybrid search, persistence, load_or_build_index()
│   ├── tools.py        # PythonREPLTool (with E2B comments) and retriever tool
│   ├── graph.py        # AgentState TypedDict, all 4 nodes, edges, compiled graph
│   ├── main.py         # Terminal CLI with streaming output and LangSmith tracing
│   └── eval.py         # Evaluation: EM, Precision, Recall, latency, LLM-as-a-Judge
├── .env.example
├── requirements.txt
├── README.md
└── report.md
```

---

## 5. Deliverables

### 5.1 README.md Must Contain

- Prerequisites (Python 3.10+, OpenAI API key with ~$5 credit)
- Installation steps (`pip install -r requirements.txt`)
- `.env` setup instructions
- How to run the chatbot: `python src/main.py` (auto-ingests on first run)
- How to run evaluation: `python src/eval.py` (default 100 samples) or `python src/eval.py --samples 5` for quick test
- Note on production serving: vLLM as drop-in replacement for OpenAI API, Ollama for local CPU inference
- Architecture diagram in ASCII art showing the LangGraph node flow:

```
START ──► retrieve ──► reason_and_code ──► execute
                            ▲                 │
                            │  error < 3      │
                            └─────────────────┤
                                              │ success OR error >= 3
                                              ▼
                                    generate_final_answer ──► END
```

### 5.2 report.md Must Contain

- **Introduction:** FinQA dataset overview, what makes financial QA uniquely hard
- **Methodology:** Full Agentic-RAG architecture explanation, chunking rules (paragraph-level text, intact tables), multi-vector retrieval, hybrid search rationale (two-stage RRF merge with dual BM25 indexing and explicit parent-dedup logic), sandboxed execution necessity, per-sample vs. global retrieval design decision (with limitation acknowledgment)
- **Cost Management:** Model selection rationale (GPT-4o-mini for reasoning, GPT-4o only for judge), cost breakdown, budget adherence
- **Evaluation Results:** Table showing EM accuracy, Precision, Recall, latency, and LLM-as-a-Judge score from `eval.py`
- **Production Considerations:** LangSmith tracing setup, vLLM as the GPU-production equivalent (with Ollama as CPU alternative), TEI for embedding serving, E2B for sandboxed code execution, drift detection via tool error rate monitoring

### 5.3 Presentation Outline (include as `presentation_outline.md`)

| Slide | Content |
|---|---|
| 1 | **Title & Problem Statement** — why financial QA is uniquely hard (numerical hallucination, tabular+text fusion) |
| 2 | **Data Understanding** — FinQA structure, paragraph-level chunking strategy (why not merge-then-split), what `gold_inds` tells us, per-sample scoped retrieval decision and its limitations |
| 3 | **Architecture** — LangGraph cyclic graph diagram, 4 nodes, retry guard, empty-code guard, cost-aware model selection (GPT-4o-mini primary, GPT-4o judge-only) |
| 4 | **Retrieval Deep-Dive** — Multi-vector retriever (summary→FAISS, UUID→raw table), two-stage BM25+FAISS merge via RRF, dual BM25 indexing (raw table + summary), parent-dedup logic |
| 5 | **Demo** — Live terminal walkthrough with streaming node output |
| 6 | **Evaluation** — EM table, Precision & Recall (macro-avg), LLM-as-a-Judge reasoning score, single-step percentage normalization logic, LangSmith trace screenshot |
| 7 | **Path to Production** — vLLM on GPU (Ollama on CPU), TEI for embeddings, E2B sandboxing, LangSmith drift monitoring |
