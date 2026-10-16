import pandas as pd
from LLMops.PIpeLines.DataPipeline.DataValidation import DataValidation
from LLMops.Configurations import DPConfi as dp

class DataCleaning:
    def __init__(self,datapath):
        self.df = pd.read_parquet(datapath)
        self.validation = DataValidation(datapath)

    def duplicate_remover(self):
        row_remove= self.validation.duplicate_rows_to_remove()
        self.df = self.df.drop(index=row_remove)
        print('Duplicates are Removed')

    def NullRemover(self):

        self.df = self.df.dropna(
            subset=["question", "answer"]
        )

        ### Remove empty/whitespace-only values ###
        self.df["question"] = self.df["question"].str.strip()
        self.df["answer"] = self.df["answer"].str.strip()

        self.df = self.df[
            (self.df["question"] != "") &
            (self.df["answer"] != "")
            ]

        print('Null and empty values are removed')

    def Normalization(self):
        self.df["question"] = (
            self.df["question"]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        self.df["answer"] = (
            self.df["answer"]
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        print('Normalization is done')

    def run(self):
        self.duplicate_remover()
        self.NullRemover()
        self.Normalization()

        self.df.to_excel(
            r'D:\Arise\python\Mlops\LLMops\DataHouse\FinalDataSet.xlsx',
            index=False
        )
        self.df.to_parquet(r'D:\Arise\python\Mlops\LLMops\DataHouse\FinalDataSet.parquet',
            index=False)

        return self.df

dc = DataCleaning(
    dp.rdata
)

clean_df = dc.run()

print(clean_df.shape)
print(clean_df.head())
