import re
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from tools import python_repl, retrieve_documents


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: str
    generated_code: str
    execution_result: str
    error_count: int
    final_answer: str


def _extract_code(text: str) -> str:
    stripped = text.strip()
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return stripped


def build_graph(scoped_retriever=None):
    llm = ChatOpenAI(model="gpt-4o-mini")

    def retrieve(state: AgentState) -> dict:
        question = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                question = message.content
                break

        if scoped_retriever is not None:
            documents = scoped_retriever.invoke(question, k=3)
        else:
            documents = retrieve_documents(question, k=3)

        if not documents:
            return {"retrieved_context": "NO RELEVANT DOCUMENTS FOUND."}

        combined_context = "\n\n".join(doc.page_content for doc in documents)
        return {"retrieved_context": combined_context}

    def reason_and_code(state: AgentState) -> dict:
        question = state["messages"][0].content if state["messages"] else ""
        prompt_parts = []

        if state.get("retrieved_context") == "NO RELEVANT DOCUMENTS FOUND.":
            prompt_parts.append(
                "WARNING: No relevant documents were retrieved. Attempt to answer only "
                "if you have enough information from the question itself. Otherwise, "
                "print 'UNABLE_TO_ANSWER'."
            )

        prompt_parts.append(f"Question: {question}")
        prompt_parts.append(f"Retrieved context:\n{state.get('retrieved_context', '')}")

        if state.get("error_count", 0) > 0 and state.get("execution_result"):
            prompt_parts.append(
                "Your previous code failed with: "
                f"{state['execution_result']}. Fix the error and try again."
            )

        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a financial analyst assistant. Given financial context "
                        "and a question, write a Python script that calculates the exact "
                        "numerical answer. Always end your script with a print() statement "
                        "showing the result. Return ONLY the Python code, no explanation."
                    )
                ),
                HumanMessage(content="\n\n".join(prompt_parts)),
            ]
        )
        raw_content = response.content if isinstance(response.content, str) else str(response.content)
        code = _extract_code(raw_content)
        if not code.strip():
            return {
                "generated_code": "",
                "execution_result": "ERROR: LLM returned empty code block. No code to execute.",
                "error_count": state["error_count"] + 1,
            }

        return {"generated_code": code, "messages": [AIMessage(content=code)]}

    def execute(state: AgentState) -> dict:
        try:
            output = python_repl.invoke(state.get("generated_code", ""))
            output_text = str(output)
            if "Traceback" in output_text:
                return {
                    "execution_result": f"ERROR: {output_text}",
                    "error_count": state["error_count"] + 1,
                }
            return {"execution_result": output_text}
        except Exception as exc:
            return {
                "execution_result": f"ERROR: {exc}",
                "error_count": state["error_count"] + 1,
            }

    def generate_final_answer(state: AgentState) -> dict:
        if state.get("error_count", 0) >= 3:
            return {
                "final_answer": (
                    "I was unable to compute the answer after 3 attempts. "
                    f"Last error: {state.get('execution_result', '')}"
                )
            }

        question = state["messages"][0].content if state["messages"] else ""
        response = llm.invoke(
            [
                HumanMessage(
                    content=(
                        f"Given this question: {question}\n\n"
                        f"Retrieved context:\n{state.get('retrieved_context', '')}\n\n"
                        f"Generated code:\n{state.get('generated_code', '')}\n\n"
                        f"Computation result: {state.get('execution_result', '')}\n\n"
                        "Write a clear, concise natural language answer. "
                        "State the numerical result explicitly."
                    )
                )
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)
        return {"final_answer": answer}

    def route_after_reason(state: AgentState) -> str:
        if not state.get("generated_code", "").strip():
            if state.get("error_count", 0) >= 3:
                return "generate_final_answer"
            return "reason_and_code"
        return "execute"

    def route_after_execute(state: AgentState) -> str:
        if str(state.get("execution_result", "")).startswith("ERROR:"):
            if state.get("error_count", 0) < 3:
                return "reason_and_code"
            return "generate_final_answer"
        return "generate_final_answer"

    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("reason_and_code", reason_and_code)
    builder.add_node("execute", execute)
    builder.add_node("generate_final_answer", generate_final_answer)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "reason_and_code")
    builder.add_conditional_edges(
        "reason_and_code",
        route_after_reason,
        {
            "reason_and_code": "reason_and_code",
            "execute": "execute",
            "generate_final_answer": "generate_final_answer",
        },
    )
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {
            "reason_and_code": "reason_and_code",
            "generate_final_answer": "generate_final_answer",
        },
    )
    builder.add_edge("generate_final_answer", END)
    return builder.compile()
