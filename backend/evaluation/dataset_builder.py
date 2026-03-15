
import pandas as pd

def build_eval_dataset(questions, answers, contexts):

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts
    }

    df = pd.DataFrame(data)

    return df