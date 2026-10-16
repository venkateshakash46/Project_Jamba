from sklearn.model_selection import train_test_split
import pandas as pd
from LLMops.Configurations import DPConfi as dp

class DataSplitter:
    def __init__(self, FinalDataSet):
        self.df =pd.read_parquet(FinalDataSet)


    def data_split(self):
        train_df, temp_df = train_test_split(
            self.df,
            test_size=0.2,
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42
        )

        train_df.to_parquet(r"D:\Arise\python\Mlops\LLMops\DataHouse\train.parquet", index=False)
        val_df.to_parquet(r"D:\Arise\python\Mlops\LLMops\DataHouse\validation.parquet", index=False)
        test_df.to_parquet(r"D:\Arise\python\Mlops\LLMops\DataHouse\test.parquet", index=False)

        return train_df, val_df, test_df


ds=DataSplitter(dp.finaldata)
ds.data_split()
