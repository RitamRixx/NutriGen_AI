from typing import TypedDict, List, Optional, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class UserProfile(TypedDict, total=False):
    age: int
    weight_kg: float
    height_cm: float
    goal: Literal["weight_loss", "weight_gain", "maintenance"]
    workout: bool
    sleep_quality: Literal["poor", "average", "good"]
    health_condition: Literal[
        "diabetes",
        "thyroid",
        "hypertension",
        "pcos",
        "lactose_intolerance",
        "Obesity",
        "none"
    ]


class CalculatedMetrics(TypedDict, total=False):
    bmi: float
    bmr: float
    tdee: float
    recommended_calories: int


class Meal(TypedDict):
    breakfast: str
    lunch: str
    dinner: str
    snacks: str


class DayPlan(TypedDict):
    day: str
    meals: Meal
    notes: str


class DietPlan(TypedDict, total=False):
    summary: dict
    weekly_plan: List[DayPlan]


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: UserProfile
    calculated_metrics: CalculatedMetrics
    query: str
    retrieved_docs: List[Document]
    diet_plan: DietPlan
    raw_llm_output: str
    errors: List[str]
    retry_count: int
    is_profile_complete: bool