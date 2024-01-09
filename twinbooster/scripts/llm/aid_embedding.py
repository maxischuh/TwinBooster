from transformers import AutoModel, AutoTokenizer
import torch
from tqdm.auto import tqdm
import pandas as pd
import os
from datasets import Dataset, DatasetDict
import numpy as np
import gc
import torch
from tqdm import tqdm
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embeddings(inputs):
    inputs = {
        "input_ids": torch.LongTensor(inputs["input_ids"]).to(device),
        "attention_mask": torch.LongTensor(inputs["attention_mask"]).to(device),
        "token_type_ids": torch.LongTensor(inputs["token_type_ids"]).to(device),
    }
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.detach()
    embeddings = torch.mean(embeddings, dim=1)
    del inputs, outputs
    gc.collect()  # force garbage collection to free up memory
    return embeddings.cpu().numpy(force=True)  # move embeddings back to cpu


def read_and_concatenate_csvs_from_folder(path: str) -> pd.DataFrame:
    """
    Read and concatenate CSV files with the same prefix in a folder.

    :param path: The directory where the CSV files are located.
    :return: A concatenated pandas DataFrame containing data from all the CSV files.
    """
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_files.sort()

    if not csv_files:
        raise ValueError(
            "No CSV files found in the specified folder with the given prefix."
        )

    df = pd.read_csv(f"{path}/{csv_files[0]}", low_memory=False)

    for file in tqdm(csv_files[1:]):
        single_file = pd.read_csv(f"{path}/{file}", low_memory=False)
        df = pd.concat([df, single_file], ignore_index=True)

    return df


def replace_patterns(input_string):
    # Replace "§§§" with ". " if the previous symbol was not a "."
    result = re.sub(r"(?<!\.)§§§", ". ", input_string)
    # If there is a "." then just add a " "
    result = re.sub(r"\.§§§", " ", result)
    # Replace "§§" with ". " if the previous symbol was not a "."
    result = re.sub(r"(?<!\.)§§", ". ", result)
    # If there is a "." then just add a " "
    result = re.sub(r"\.§§", " ", result)
    # Check that never two " " occur
    result = re.sub(r"  +", " ", result)
    # Check that never two "." or two ":", ";", "!", "?" or other sentence end symbols occur
    result = re.sub(r"([.:;!?])\1+", r"\1", result)
    # Ensure the string does not end with a space
    result = result.rstrip()
    return result


model_name = "mschuh/PubChemDeBERTa-augmented"
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

data = read_and_concatenate_csvs_from_folder("../llm_dataset/text")

data["text"] = data["text"].apply(replace_patterns).to_list()
# data["text"] = [tokenizer.backend_tokenizer.normalizer.normalize_str(t) for t in data["text"]]
data["text"] = data["text"].apply(tokenizer.backend_tokenizer.normalizer.normalize_str)
data = Dataset.from_pandas(data)

inputs = data.map(
    lambda examples: tokenizer(
        examples["text"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    ),
    batched=True,
    remove_columns=["aid", "text", "Unnamed: 0"],
)
overflow_mapping = inputs["overflow_to_sample_mapping"]
inputs = inputs.remove_columns("overflow_to_sample_mapping")

all_embeddings = np.empty((0, 768))
batch_size = 100  # adjust this according to your memory capacity
num_batches = len(inputs) // batch_size + (len(inputs) % batch_size != 0)

for i in tqdm(range(num_batches)):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, len(inputs))
    embeddings = get_embeddings(inputs[start_index:end_index])
    # np.savez(f"../dataset/fine-tuned-emb/embeddings_{i}.npz", embeddings=embeddings)
    all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)

np.savez("../llm_dataset/fine_tuned_embeddings.npz", embeddings=all_embeddings)
np.savez(
    "../llm_dataset/fine_tuned_overflow_mapping.npz", overflow_mapping=overflow_mapping
)

data = read_and_concatenate_csvs_from_folder("../llm_dataset/text")
# other_embeddings = np.load("../llm_dataset/fine_tuned_embeddings.npz")["embeddings"]
# overflow_mapping = np.load("../llm_dataset/fine_tuned_overflow_mapping.npz")["overflow_mapping"]
aids = data["aid"]

last_value = overflow_mapping[0]
adjustment_factor = 0

# Create an adjusted overflow mapping
adjusted_overflow_mapping = np.zeros_like(overflow_mapping)
for i, value in enumerate(overflow_mapping):
    if value < last_value:
        # A cycle has completed, increase the adjustment factor
        adjustment_factor += 1000
    adjusted_overflow_mapping[i] = value + adjustment_factor
    last_value = value

means = {}
for i in tqdm(np.unique(adjusted_overflow_mapping)):
    tmp = all_embeddings[np.where(adjusted_overflow_mapping == i)]
    means[aids[i]] = np.mean(tmp, axis=0).astype(np.single)

df = pd.DataFrame(means)
df.to_pickle("../llm_dataset/fine_tuned_aid_embeddings.pkl")
