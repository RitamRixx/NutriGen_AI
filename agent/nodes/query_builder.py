from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts import QUERY_BUILDER_PROMPT
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def query_builder_node(state: AgentState) -> AgentState:
    """
    Converts the user profile + metrics into an optimised semantic search
    query that will be used to retrieve relevant nutrition context from Chroma.
    """

    profile = state.get("user_profile", {})
    matrics = state.get("calculated_metrics", {})

    full_profile = {**profile, **matrics}
    prompt = ChatPromptTemplate.from_template(QUERY_BUILDER_PROMPT)

    chain = prompt | llm

    response = chain.invoke({"user_profile": str(full_profile)})
    query = response.content.strip().strip('"')

    print(f"[Query Builder] Generated query: {query}")
    return {"query": query}