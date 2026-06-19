from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.mongodb import MongoDBSaver

load_dotenv()

llm = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai"
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

config = {
    "configurable": {
        "thread_id": "akshay"
    }
}

with MongoDBSaver.from_conn_string(
    "mongodb://localhost:27017"
) as checkpointer:

    graph = graph_builder.compile(
        checkpointer=checkpointer
    )

    print("\nType 'exit' to quit\n")

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        print("\nAI:")

        final_response = ""

        for event in graph.stream(
            {
                "messages": [user_input]
            },
            config=config,
            stream_mode="values"
        ):
            if "messages" in event:
                msg = event["messages"][-1]

                if hasattr(msg, "content"):
                    final_response = msg.content

        print(final_response)
        print()