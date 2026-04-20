# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from agent.state import AgentState
from agent.prompts import DIET_GENERATION_PROMPT
from ingestion.vector_store import load_vector_store
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7,
)

VECTOR_DB = load_vector_store(persist_directory="./vector_store")

def diet_generator_node(state: AgentState) -> AgentState:
    """
    1. Loads pre-built Chroma vector store
    2. Retrieves relevant nutrition docs using MMR (condition-aware)
    3. Feeds profile + metrics + context into the diet generation LLM
    """
    
    profile = state.get("user_profile", {})
    metrics = state.get("calculated_metrics", {})
    query = state.get("query", "balanced diet plan")
    health_condition = profile.get("health_condition","none")

    docs = []

    context = "No additional context available."

    try:
        # vector_store = load_vector_store(persist_directory="./vector_store")
        vector_store = VECTOR_DB

        search_kwargs = {"k": 5, "fetch_k": 20}
        if health_condition and health_condition.lower() not in ("none", "None"):
            pass
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )

        docs = retriever.invoke(query)
        context = "\n\n---\n\n".join([
            f"[Source: {doc.metadata.get('source_type', 'unknown')}]\n{doc.page_content}"
            for doc in docs
        ])
        print(f"[diet_generator] Retrieved {len(docs)} docs")

    except Exception as e:
        print(f"[diet_generator] Retrieval failed: {e}. Proceeding without context.")

    # --- Generation ---
    prompt = ChatPromptTemplate.from_template(DIET_GENERATION_PROMPT)
    chain = prompt | llm

    response = chain.invoke({
        "user_profile": str(profile),
        "calculated_metrics": str(metrics),
        "context": context
    })

    return {
        "raw_llm_output": response.content,
        "retrieved_docs": docs,
        "messages": [AIMessage(content="Your personalized 7-day diet plan is ready! Scroll down to view it.")]
    }

