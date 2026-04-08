from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain_experimental.tools import PythonAstREPLTool
from pydantic import PrivateAttr


class ConfigurableRetrieverAdapter(BaseRetriever):
    _retriever: Any = PrivateAttr(default=None)

    def set_retriever(self, retriever: Any) -> None:
        self._retriever = retriever

    def _get_relevant_documents(self, query: str) -> list[Document]:
        if self._retriever is None:
            return []
        return self._retriever.invoke(query, k=3)


_retriever_adapter = ConfigurableRetrieverAdapter()

financial_document_retriever = create_retriever_tool(
    _retriever_adapter,
    name="financial_document_retriever",
    description=(
        "Searches and retrieves relevant financial text paragraphs and tables "
        "from earnings reports. Use this tool first to find the data needed to "
        "answer a financial question."
    ),
)

# PRODUCTION NOTE: In a production environment, replace the local Python REPL
# a sandboxed execution environment such as E2B (e2b_code_interpreter) or a
# Dockerized container to prevent Arbitrary Code Execution (ACE) vulnerabilities.
python_repl = PythonAstREPLTool()
python_repl.description = (
    "Use this tool to execute Python code for mathematical calculations. "
    "Pass the exact Python code to run as a string. "
    "Always print() the final result so it appears in the output. "
    "Use this for all arithmetic - never compute numbers in your head."
)

tools = [financial_document_retriever, python_repl]


def configure_retriever_tool(retriever) -> None:
    _retriever_adapter.set_retriever(retriever)


def retrieve_documents(query: str, k: int = 3) -> list[Document]:
    if _retriever_adapter._retriever is None:
        return []
    return _retriever_adapter._retriever.invoke(query, k=k)
