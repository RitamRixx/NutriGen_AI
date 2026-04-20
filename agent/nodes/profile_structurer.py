from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from agent.state import AgentState
from agent.prompts import PROFILE_EXTRACTION_PROMPT
import json
import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


required_fields = ["weight_kg", "height_cm", "goal", "workout", "sleep_quality", "health_condition"]


def check_profile_completeness(profile: dict) -> bool:
    return all(profile.get(field) is not None for field in required_fields)

def profile_structurer_node(state: AgentState) -> AgentState:
    """
    Parses the full conversation history and extracts/updates the structured user profile.
    Runs at the start of every turn BEFORE input_collector so the collector
    knows exactly what fields are still missing.
    """

    messages = state.get("messages", [])
    existing_profile = state.get("user_profile", {})

    conv_text = "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}" 
                          for msg in messages])
    
    prompt = ChatPromptTemplate.from_template(PROFILE_EXTRACTION_PROMPT)
    chain = prompt | llm

    try:
        response = chain.invoke({"messages": conv_text})
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.lower().startswith("json"):
                content = content[4:]
            content = content.strip()

        extracted: dict = json.loads(content)

        # Merge: only overwrite fields that were actually extracted (not null)
        updated_profile = {**existing_profile}
        for key, value in extracted.items():
            if value is not None:
                updated_profile[key] = value

        is_complete = check_profile_completeness(updated_profile)

        return {
            "user_profile": updated_profile,
            "is_profile_complete": is_complete
        }

    except (json.JSONDecodeError, Exception) as e:
        print(f"[profile_structurer] Extraction error: {e}")
        # Don't crash the graph — return unchanged profile
        return {
            "user_profile": existing_profile,
            "is_profile_complete": check_profile_completeness(existing_profile)
        }