from typing_extensions import TypedDict

from typing import Annotated
from dotenv import load_dotenv


from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model  # FIXED
load_dotenv()
llm = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

class State(TypedDict):
    messages:Annotated[list,add_messages]


def chatbot(state:State):
    response=llm.invoke(state.get("messages"))

    return {"messages":[response]}

def sampleNode(state:State):
    print("\n\nInside sample node",state)


    
    return {"messages":["Hi, Sample message appened"]}

graph_builder=StateGraph(State)

graph_builder.add_node("chatbot",chatbot)


graph_builder.add_node("sampleNode",sampleNode)


graph_builder.add_edge(START,"chatbot")

graph_builder.add_edge("chatbot","sampleNode")

graph_builder.add_edge("sampleNode",END)


graph=graph_builder.compile()

updated_state=graph.invoke(State({"messages":["Hi, my name is Rohit Rai"]}))


print("\n\nupdated_state",updated_state)