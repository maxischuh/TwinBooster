import json
import os
import math
import random
import re
import numpy as np
import sklearn
import transformers
from pynvml import *
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
import torch
from transformers import Trainer, AutoModelForMaskedLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def normalize_str(text: list, tokenizer) -> list:
    text = [tokenizer.backend_tokenizer.normalizer.normalize_str(t) for t in text]
    return text


class TextProcessor:
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.trainer = None
        self.seed = 42
        self.max_length = None
        self.run()

    def run(self):
        self.set_seed(self.seed)

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        sklearn.utils.check_random_state(self.seed)
        transformers.set_seed(self.seed)
        # pd.core.common._random_state(self.seed)

    def get_seed(self):
        return self.seed

    def tokenize_function(self, examples):
        return self.tokenizer(
            normalize_str(examples["text"], self.tokenizer),
            padding=True,
            truncation=True,
            max_length=self.max_length,  # to be optimized
            return_tensors="pt",
            return_overflowing_tokens=True,
        )

    def fine_tune(
        self,
        dataset_path: str,
        output_dir: str,
        ampere: bool = True,
        num_train_epochs: float = 3.0,
        learning_rate: float = 3e-5,  # optimal value
        weight_decay: float = 0.01,
        batch_size: int = 32,  # optimal value
        max_length: int = 128,  # optimal value
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-6,
        warmup_steps: int = 500,
    ):

        dataset = np.load(dataset_path, allow_pickle=True)
        output_dir = output_dir.split("/")[-1]
        train, test = train_test_split(dataset, test_size=0.2, random_state=42)
        # test, val = train_test_split(test, test_size=0.5, random_state=42)
        dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(pd.DataFrame(train, columns=["text"])),
                "test": Dataset.from_pandas(pd.DataFrame(train, columns=["text"])),
            }
        )
        # "val": Dataset.from_pandas(val)})
        self.max_length = max_length

        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"],
            num_proc=16,  # To avoid RAM overflow
        )
        tokenized_dataset = tokenized_dataset.remove_columns(
            "overflow_to_sample_mapping"
        )
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm_probability=0.15
        )
        torch.backends.cuda.matmul.allow_tf32 = ampere

        training_args = TrainingArguments(
            output_dir=output_dir.split("/")[-1],
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            remove_unused_columns=False,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            fp16=ampere,
            tf32=ampere,
            num_train_epochs=num_train_epochs,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
        )

        model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)

        print("Training...")
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        self.trainer.train()

        eval_results = self.trainer.evaluate()
        print(f"Perplexity after: {math.exp(eval_results['eval_loss']):.2f}")
        print("Saving...")
        self.trainer.save_model(output_dir)
        # Save metrics
        with open(f"{output_dir}/metrics.json", "w") as f:
            json.dump(self.trainer.evaluate(), f)


if __name__ == "__main__":
    model_checkpoint = "microsoft/deberta-v3-base"

    text_processor = TextProcessor(model_checkpoint)
    text_processor.fine_tune(
        dataset_path="../llm_dataset/shuffled_text.npy",
        output_dir=f"{model_checkpoint}-finetuned-augmented",
    )
