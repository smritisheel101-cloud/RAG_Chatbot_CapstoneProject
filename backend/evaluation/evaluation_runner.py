
from app.services.agent import create_rag_agent
from app.evaluation.ragas_evaluator import evaluate_rag

def run_evaluation():

    questions = [
        "Which Dell laptop is best for gaming?",
        "What are the specs of Dell XPS 13?"
    ]

    ground_truths = [
        "Alienware series is best for gaming",
        "Dell XPS 13 has Intel processor and high resolution display"
    ]

    agent = create_rag_agent()

    answers = []
    contexts = []

    for q in questions:

        response = agent.invoke(
            {"messages": [{"role": "user", "content": q}]}
        )

        answers.append(response["messages"][-1].content)

        contexts.append(["retrieved document chunk"])

    result = evaluate_rag(
        questions,
        answers,
        contexts,
        ground_truths
    )

    print(result)


if __name__ == "__main__":
    run_evaluation()