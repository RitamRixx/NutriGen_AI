import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import agent_graph

st.set_page_config(
    page_title="NutriGen AI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

WELCOME_MESSAGE = (
    "👋 Hi! I'm **NutriGen**, your personal AI nutritionist.\n\n"
    "I'll ask you a few quick questions about your health and lifestyle, "
    "then generate a personalized **7-day Indian meal plan** just for you.\n\n"
    "Let's start — what's your primary goal? *(weight loss / weight gain / maintenance)*"
)


# ── Session state init ──────────────────────────────────────────────────────

def init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": WELCOME_MESSAGE}
        ]
    if "diet_plan" not in st.session_state:
        st.session_state.diet_plan = None


def get_current_state() -> dict:
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    try:
        return agent_graph.get_state(config).values
    except Exception:
        return {}


# ── Diet plan renderer ──────────────────────────────────────────────────────

def display_diet_plan(plan: dict):
    st.divider()
    st.header("🥗 Your Personalized 7-Day Meal Plan")

    summary = plan.get("summary", {})
    col1, col2 = st.columns(2)
    col1.metric("🎯 Goal", str(summary.get("goal", "—")).replace("_", " ").title())
    col2.metric("🔥 Daily Calories", f"{summary.get('daily_calories', '—')} kcal")

    if summary.get("disclaimer"):
        st.info(summary["disclaimer"])

    weekly = plan.get("weekly_plan", [])
    if not weekly:
        st.warning("Plan data unavailable. Please try again.")
        return

    tabs = st.tabs([d["day"] for d in weekly])
    for tab, day in zip(tabs, weekly):
        with tab:
            meals = day.get("meals", {})
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**🌅 Breakfast**")
                st.write(meals.get("breakfast", "—"))
                st.markdown("**☀️ Lunch**")
                st.write(meals.get("lunch", "—"))
            with c2:
                st.markdown("**🌙 Dinner**")
                st.write(meals.get("dinner", "—"))
                st.markdown("**🍎 Snacks**")
                st.write(meals.get("snacks", "—"))
            if day.get("notes"):
                st.caption(f"📝 {day['notes']}")


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("🥗 NutriGen AI")
        st.caption("Powered by Gemini + LangGraph + RAG")
        st.divider()

        state = get_current_state()
        profile = state.get("user_profile", {})
        metrics = state.get("calculated_metrics", {})

        if profile:
            st.subheader("📋 Your Profile")
            for k, v in profile.items():
                if v is not None:
                    st.write(f"**{k.replace('_', ' ').title()}:** {v}")

        if metrics:
            st.divider()
            st.subheader("📊 Calculated Metrics")
            if metrics.get("bmi"):
                st.metric("BMI", metrics["bmi"])
            if metrics.get("recommended_calories"):
                st.metric("Target Calories", f"{metrics['recommended_calories']} kcal/day")

        st.divider()
        if st.button("🔄 Start New Session", use_container_width=True):
            for key in ["thread_id", "chat_history", "diet_plan"]:
                st.session_state.pop(key, None)
            st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session()
    render_sidebar()

    st.title("🥗 NutriGen AI — Personal Nutritionist")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Render diet plan below chat if available
    if st.session_state.diet_plan:
        display_diet_plan(st.session_state.diet_plan)

    # Chat input
    if user_input := st.chat_input("Your message..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.spinner("NutriGen is thinking..."):
            try:
                agent_graph.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config
                )

                state = agent_graph.get_state(config).values
                all_messages = state.get("messages", [])

                # Pick the last AI message added this turn
                last_ai = next(
                    (m for m in reversed(all_messages) if isinstance(m, AIMessage)),
                    None
                )
                if last_ai:
                    with st.chat_message("assistant"):
                        st.markdown(last_ai.content)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": last_ai.content}
                    )

                # Capture diet plan when ready
                diet_plan = state.get("diet_plan")
                if diet_plan and diet_plan.get("weekly_plan"):
                    st.session_state.diet_plan = diet_plan

            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")

        st.rerun()


if __name__ == "__main__":
    main()