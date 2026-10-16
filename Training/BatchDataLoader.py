from datasets import load_dataset
from torch.utils.data import DataLoader
from LLMops.PIpeLines.TrainingPipeLine.DataCollator import DataCollator
from LLMops.Configurations import TrainConfi as tc


class BatchDataLoader:

    def __init__(self, datafile, collator):
        self.dataset = load_dataset(
            "parquet",
            data_files=datafile,
            split="train"
        )
        self.collator = collator

    def get_dataloader(self):

        return DataLoader(
            self.dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            collate_fn=self.collator
        )

#ChatTemplate → tokenizes the data. DataLoader → selects/groups examples into batches.
# DataCollator → pads and prepares the batch (including labels for causal LM)
dc = DataCollator().get_collator()

dl = BatchDataLoader(
    r'D:\Arise\python\Mlops\LLMops\PIpeLines\TrainingPipeLine\train_tokenized.parquet',
    dc
)

train_dataloader = dl.get_dataloader()

for batch in train_dataloader:
    print(batch["input_ids"].shape)
    print(batch["attention_mask"].shape)
    print(batch["labels"].shape)
    break