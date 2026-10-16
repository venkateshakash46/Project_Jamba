from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer

import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset
''' input_ids:
[101, 50, 60,  0,  0]

attention_mask: so this is created by datacollator
[  1,  1,  1,  0,  0]'''

from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling
    )
import TrainConfi as tc


class DataCollator:


    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
                tc.model_name
            )

        # Use EOS token as padding token
        # only if tokenizer doesn't already have one.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            ## this is used to mention the default padding when we have differnet length for us it is '2'

        self.collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
            ## since we are working with causual model predict token from previous not
            ## masked-language-model training such as BERT
            )

    def get_collator(self):
        return self.collator

