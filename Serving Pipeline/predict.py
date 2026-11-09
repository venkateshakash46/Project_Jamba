import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMops.Configurations import TrainConfi as tc

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class jambaPredictor:

    def __init__(self):
        self.model_name = tc.model_name

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print("Loading model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model.eval()

        print(" model loaded successfully!")

    def predict(self,question, max_new_tokens=300):

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
