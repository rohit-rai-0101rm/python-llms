from typing_extensions import TypedDict
from typing import Optional, Literal
from langgraph.graph import StateGraph, START, END
from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


class State(TypedDict):
    user_query: str
    llm_output: Optional[str]
    is_good: Optional[bool]


def chatbot(state: State):

    print("chatbot node",state)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=state["user_query"]
    )
    return {"llm_output": response.text}


def evaluate_response(state: State) -> Literal["chatbot_other", "endnode"]:
    print("chatbot node evaluate other",state)

    if state["is_good"]:
        return "endnode"
    return "chatbot_other"


def chatbot_other(state: State):

    print("chatbot node other",state)
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=state["user_query"]
    )
    return {"llm_output": response.text}


def endnode(state: State):
    print("chatbot node end other",state)

    return state


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("chatbot_other", chatbot_other)
graph_builder.add_node("endnode", endnode)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", evaluate_response)
graph_builder.add_edge("chatbot_other", "endnode")
graph_builder.add_edge("endnode", END)

graph = graph_builder.compile()

result = graph.invoke(
    {
        "user_query": "What is 2+2",
        "is_good": True
    }
)

print(result)