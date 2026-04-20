from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.state import AgentState
from agent.prompts import INPUT_COLLECTION_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def input_collector_node(state: AgentState) -> AgentState:
    """
    Asks the user for any missing profile fields.
    Runs AFTER profile_structurer so it knows exactly what's missing.
    Only returns an updated messages list with the AI's question.
    """
    user_profile = state.get("user_profile", {})
    messages = state.get("messages", [])

    prompt = ChatPromptTemplate.from_messages([
        ("system", INPUT_COLLECTION_PROMPT),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm

    response = chain.invoke({
        "user_profile": str(user_profile),
        "messages": messages
    })

    return {"messages": [AIMessage(content=response.content)]}