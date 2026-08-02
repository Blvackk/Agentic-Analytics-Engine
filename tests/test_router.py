from src.agents.router import detect_route

print("\n========== ROUTER ==========\n")

questions = [
    ("Show missing values", "eda"),
    ("What is the correlation between age and income?", "statistics"),
    ("Plot salary distribution", "visualization"),
    ("Train a regression model", "machine_learning"),
    ("Hello", "unknown"),
]

for question, expected in questions:

    predicted = detect_route(question)

    print(f"Question : {question}")
    print(f"Expected : {expected}")
    print(f"Predicted: {predicted}\n")

    assert predicted == expected

print("Router Tests Passed!")