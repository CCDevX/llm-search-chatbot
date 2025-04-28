import os

from langchain_mistralai import ChatMistralAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate

os.environ["TAVILY_API_KEY"] = "key"
os.environ["MISTRAL_API_KEY"] = "key"

prompt = ChatPromptTemplate.from_messages(
    [("system", "Tu es Frederic, un developpeur fullstack."),
     ("placeholder", "{messages}"),
     ("user", "N'hesite pas à être serviable."),
     ])

def format_for_model(state):
    return prompt.invoke({"messages": state["messages"]})

memory = MemorySaver()

model = ChatMistralAI(model="mistral-large-latest")
search_tool = TavilySearchResults(
    max_results=2)

tools = [search_tool]
agent_executor = create_react_agent(model, tools, checkpointer=memory,state_modifier=format_for_model)

def print_stream(graph, inputs, config):
    for s in graph.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

config = {"configurable": {"thread_id": "thread-1"}}

inputs = {"messages": [
    {"role": "user", "content": "Salut!"},
]}

print_stream(agent_executor, inputs, config)    

inputs = {"messages": [
    {"role": "user", "content": "Peux-tu me parler de la librairie React ?"},
]}

print_stream(agent_executor, inputs, config)    


print_stream(agent_executor, inputs, config)   