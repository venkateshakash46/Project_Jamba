import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

from LLMops.Configurations import TrainConfi as tc


class ChatTemplate:

    def __init__(self, train_df, test_df, validation_df):
        self.train_df = pd.read_parquet(train_df)
        self.test_df = pd.read_parquet(test_df)
        self.validation_df = pd.read_parquet(validation_df)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tc.model_name
        )

    def apply_chat_template(self, example):
        messages = [
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": example["answer"]
            }
        ]

        formatted_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )  ### by mention False we say that dont do Tokenization
        ## Interesting : the reason we are doing is because if we apply true it will do formated test and then tokenizion one at
        ## but here we kept false so we can do batch wise tokenization efficiently due to gpu

        return {
            "text": formatted_text
        }

    # Convert Pandas DataFrame ----> Hugging Face Dataset ( this is due to make a use of map() to apply chattemplate)
    ## so we can avoid the writing the code for for loop to apply each of time chat template to each example
    def convert_to_hf_dataset(self):
        train_dataset = Dataset.from_pandas(
            self.train_df,
            preserve_index=False
        )

        validation_dataset = Dataset.from_pandas(
            self.validation_df,
            preserve_index=False
        )

        test_dataset = Dataset.from_pandas(
            self.test_df,
            preserve_index=False
        )

        return (
            train_dataset,
            validation_dataset,
            test_dataset
        )

    def Tokenization(self, dataset):
        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                truncation=True,
                max_length=tc.max_length
            )

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )

    def run(self):
        train_dataset, validation_dataset, test_dataset = (
            self.convert_to_hf_dataset()
        )

        train_dataset = train_dataset.map(
            self.apply_chat_template
        )

        validation_dataset = validation_dataset.map(
            self.apply_chat_template
        )

        test_dataset = test_dataset.map(
            self.apply_chat_template
        )

        train_dataset = self.Tokenization(train_dataset)
        validation_dataset = self.Tokenization(validation_dataset)
        test_dataset = self.Tokenization(test_dataset)

        return (
            train_dataset,
            validation_dataset,
            test_dataset
        )

ct = ChatTemplate(
    tc.train_df,
    tc.test_df,
    tc.validation_df
)

train_dataset, test_dataset, validation_dataset = ct.run()

print("Training")
print(train_dataset[:2])

print("Test")
print(test_dataset[:2])

print("Validation")
print(validation_dataset[:2])


# Save tokenized datasets
train_dataset.to_pandas().to_parquet(
    "train_tokenized.parquet",
    index=False
)

test_dataset.to_pandas().to_parquet(
    "test_tokenized.parquet",
    index=False
)

validation_dataset.to_pandas().to_parquet(
    "validation_tokenized.parquet",
    index=False
)