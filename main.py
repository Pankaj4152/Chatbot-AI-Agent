# Requirements:
# - Python 3.8+
# - Install packages: pip install langchain-core langgraph>0.2.27
# - Install langchain with OpenAI support: pip install -qU "langchain[openai]"
# - Set up API keys for LangSmith and OpenAI

import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Set environment variables for LangSmith and OpenAI
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Initialize the chat model with gpt-4o-mini from OpenAI
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# Define the prompt template with a pirate theme
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You talk like a pirate..."), MessagesPlaceholder("history")]
)

# Set up conversation trimming
trimmer = trim_messages(
    max_tokens=65, strategy="last", token_counter=model, include_system=True,
    allow_partial=False, start_on="human"
)

# Create workflow with memory
def call_model(state):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"history": trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": state["messages"] + [response]}

workflow = StateGraph(state_schema=MessagesState)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Configure the chatbot with a thread ID
thread_id = "abc123"
config = {"configurable": {"thread_id": thread_id}}

# Example interaction
print("Starting chatbot. Type 'quit' to exit.")

messages = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    new_user_message = HumanMessage(content=user_input)
    full_messages = messages + [new_user_message]
    output = app.invoke({"messages": full_messages}, config)
    bot_response = output["messages"][-1].content
    print("Bot:", bot_response)
    messages = output["messages"]