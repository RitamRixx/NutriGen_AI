import json
import re
from agent.state import AgentState

DISCLAIMER = (
    "This is a general AI-generated recommendation. "
    "Please consult a registered dietitian or healthcare professional "
    "before making significant dietary changes."
)


def output_formatter_node(state: AgentState) -> AgentState:
    """
    Parses raw LLM output into a validated DietPlan dict.
    Strips markdown fences, injects disclaimer, handles parse errors.
    """
    raw = state.get("raw_llm_output", "")

    try:
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            brace_match = re.search(r"\{[\s\S]*\}", raw)
            json_str = brace_match.group(0) if brace_match else raw

        diet_plan: dict = json.loads(json_str)

        if "summary" in diet_plan:
            diet_plan["summary"]["disclaimer"] = DISCLAIMER

        print("[output_formatter] Diet plan parsed successfully.")
        return {"diet_plan": diet_plan, "errors": []}

    except (json.JSONDecodeError, AttributeError) as e:
        error_msg = f"JSON parse failed: {e}"
        print(f"[output_formatter] {error_msg}")
        return {
            "diet_plan": {},
            "errors": [error_msg]
        }