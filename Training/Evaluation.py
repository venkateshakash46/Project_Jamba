import pandas as pd
import evaluate
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMops.Configurations import TrainConfi as tc
import evaluate

class ModelEvaluator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tc.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            tc.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        # Set model to evaluation mode
        self.model.eval()

        print("Data Science assistant is ready!")

        # Load evaluation metrics
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def generate_answer(self,question, max_new_tokens=300):

        messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Data Science tutor. "
                        "Explain concepts clearly, provide correct Python "
                        "code when useful, and explain the code step by step."
                    )
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

        inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )

        answer_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = self.tokenizer.decode(
                answer_tokens,
                skip_special_tokens=True
            )

        return answer.strip()


    def evaluate_dataset(self, dataframe):

        predictions = []
        references = []

        for _, row in tqdm(dataframe.iterrows(),
                           total=len(dataframe)):

            question = row["question"]
            expected_answer = row["answer"]

            generated_answer = self.generate_answer(question)

            predictions.append(generated_answer)
            references.append(expected_answer)


        # ROUGE
        rouge_results = self.rouge.compute(
            predictions=predictions,
            references=references
        )


        # BERTScore
        bert_results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )

        bert_f1 = sum(bert_results["f1"]) / len(
            bert_results["f1"]
        )


        return {
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "bertscore_f1": bert_f1
        }

train_df = pd.read_parquet(
    tc.train_df
)

test_df = pd.read_parquet(
    tc.test_df
)
evaluator = ModelEvaluator()

train_sample = train_df.sample(
    n=min(500, len(tc.train_df)),
    random_state=42
)

test_sample = test_df.sample(
    n=min(500, len(tc.test_df)),
    random_state=42
)


train_metrics = evaluator.evaluate_dataset(
    train_sample
)

test_metrics = evaluator.evaluate_dataset(
    test_sample
)
print("TRAIN EVALUATION")
print(train_metrics)

print("\nTEST EVALUATION")
print(test_metrics)