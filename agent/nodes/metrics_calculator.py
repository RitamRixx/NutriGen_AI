from agent.state import AgentState

def metrics_calculator_node(state: AgentState) -> AgentState:
    profile = state.get("user_profile", {})
    age = int(profile.get("age"))
    weight = float(profile.get("weight_kg"))
    height = float(profile.get("height_cm"))
    goal = profile.get("goal", "maintenance")
    workout = profile.get("workout", False)

    # BMI

    height_m = height / 100
    bmi = round(weight / (height_m ** 2), 2)

    # brm

    bmr = round((10 * weight) + (6.25 * height) - (5 * age) - 78, 2)

    # Activity factor

    if workout:
        activity_factor = 1.55
    else:
        activity_factor = 1.2

    # TDEE Calculation

    tdee = round(bmr * activity_factor, 2)

    # Calorie target based on goal
    if goal == "weight_loss":
        recommended_calories = int(tdee - 400)
    elif goal == "weight_gain":
        recommended_calories = int(tdee + 300)
    else:
        recommended_calories = int(tdee)

    recommended_calories = max(recommended_calories, 1200)

    return {
        "calculated_metrics": {
            "bmi": bmi,
            "bmr": bmr,
            "tdee": tdee,
            "recommended_calories": recommended_calories
        }
    }


