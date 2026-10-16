from datasets import load_dataset
from LLMops.Configurations import DPConfi as dp

class Ingestion:

    def __init__(self, data_path,output_path):
        self.data_path = data_path
        self.output_path = output_path

    def LoadData(self):

        dataset = load_dataset(
            self.data_path
        )

        print("### Dataset Loaded")
        print("### Details Of The Dataset")
        print(dataset)
        print()

        print(f"### Samples from {self.data_path}")

        for i in range(5):
            print(dataset["train"][i])

        df = dataset["train"].to_pandas()

        df.to_parquet(self.output_path,index=False)
        print(f"### Dataset saved to: {self.output_path}")

        return df


ing = Ingestion(dp.ds_name,dp.rdata)

df = ing.LoadData()

print(df)
