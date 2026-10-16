import torch

from datasets import load_dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from LLMops.Configurations import TrainConfi as tc
from LLMops.PIpeLines.TrainingPipeLine.DataCollator import DataCollator


class FineTune:

    def fine_tune(self):

        ## Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            tc.model_name,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token



        ##Load Jamba model
        print("\nLoading Jamba model...")
        model = AutoModelForCausalLM.from_pretrained(
            tc.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        ## Create LoRA configuration
        print("\nCreating LoRA configuration...")
        lora_config = LoraConfig(
            r=tc.lora_r,
            lora_alpha=tc.lora_alpha,
            lora_dropout=tc.lora_dropout,
            target_modules=tc.lora_target_modules,
            bias=tc.lora_bias,
            task_type=tc.lora_task_type,
        )

        # Preparing and Attaching LoRA
        print("\nAttaching LoRA adapters...")
        peft_model = get_peft_model(
            model,
            lora_config,
        )


        ## Check trainable parameters
        print("\nTrainable parameters:")
        peft_model.print_trainable_parameters()

        ## Load training dataset
        print("\nLoading training dataset...")
        train_dataset = load_dataset(
            "parquet",
            data_files=tc.Ttrain_data,
            split="train",
        )

        ## Load validation dataset

        print("Loading validation dataset...")

        val_dataset = load_dataset(
            "parquet",

            data_files=tc.val_data,

            split="train",
        )


        ## Data Collator
        print("\nCreating DataCollator...")
        dc = DataCollator()
        datacollator = dc.get_collator()

        ## Training Arguments
        training_args = TrainingArguments(
            output_dir=tc.output_dir,
            per_device_train_batch_size=tc.batch_size,
            per_device_eval_batch_size=tc.batch_size,
            gradient_accumulation_steps=  tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            num_train_epochs=tc.num_train_epochs,
            weight_decay=tc.weight_decay,
            fp16=True,
            logging_steps=10,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            report_to="none",
            remove_unused_columns=False,
        )

        ## Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=datacollator,
        )

        ## Start Training
        print("\n")
        print("=" * 60)
        print("STARTING JAMBA LoRA FINE-TUNING")
        print("=" * 60)
        print("\n")
        trainer.train()

        # Saving  LoRA Adapter and Tokenizer configurations
        print("\nSaving LoRA adapter...")

        peft_model.save_pretrained(
            tc.output_dir
        )

        tokenizer.save_pretrained(
            tc.output_dir
        )

        print("\n")
        print("=" * 60)
        print("FINE-TUNING COMPLETED")
        print("=" * 60)
        print("\nLoRA adapter saved to:")
        print(tc.output_dir)