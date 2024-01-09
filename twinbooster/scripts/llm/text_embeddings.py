import gc
import re
import random
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from transformers import AutoTokenizer, AutoModel


class TextEmbedding:
    def __init__(self, model_checkpoint: str = "mschuh/PubChemDeBERTa-augmented"):
        self.n_augmentations = None
        self.tokens = None
        self.embedding = None
        self.text = None
        self.model_checkpoint = model_checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.seed = 42
        self.max_length = 128
        self.run()

    def run(self):
        self.set_seed(self.seed)

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        sklearn.utils.check_random_state(self.seed)
        transformers.set_seed(self.seed)
        transformers.enable_full_determinism(self.seed)

    @staticmethod
    def shuffle_sentences_within_paragraphs(text: str):
        # Split the text into paragraphs using "§" as a delimiter
        paragraphs = text.split("§")
        # Initialize a list to hold shuffled paragraphs
        shuffled_paragraphs = []
        # Iterate over each paragraph
        for paragraph in paragraphs:
            # Split the paragraph into sentences using ". " as a delimiter
            sentences = paragraph.split(". ")
            # Shuffle the sentences within the paragraph
            random.shuffle(sentences)
            # Join the shuffled sentences back into the paragraph
            shuffled_paragraph = ". ".join(sentences)
            # Append the shuffled paragraph to the list
            shuffled_paragraphs.append(shuffled_paragraph)
        # Join the shuffled paragraphs back into the text
        random.shuffle(shuffled_paragraphs)
        shuffled_text = "§".join(shuffled_paragraphs)
        return shuffled_text

    @staticmethod
    def replace_patterns(input_string: str):
        # Replace multiple "§" with a single "§"
        result = re.sub(r"§+", "§", input_string)
        # Replace "§" with ". " if the previous symbol was not a "."
        result = re.sub(r"(?<!\.)§", ". ", result)
        # If there is a "." followed by "§", just add a space
        result = re.sub(r"\.§", " ", result)
        # Check that never two consecutive spaces occur, replace with a single space
        result = re.sub(r"  +", " ", result)
        # Check that never two consecutive identical sentence end symbols occur
        result = re.sub(r"([.:;!?])\1+", r"\1", result)
        # Ensure the string does not end with a space
        result = result.rstrip()
        return result

    def embedding_generator(self, text: str):
        def model_io(inputs: dict):
            inputs = {
                "input_ids": torch.LongTensor(inputs["input_ids"]),
                "attention_mask": torch.LongTensor(inputs["attention_mask"]),
                "token_type_ids": torch.LongTensor(inputs["token_type_ids"]),
            }
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.detach()
            embeddings = torch.mean(embeddings, dim=1)
            del inputs, outputs
            gc.collect()
            return embeddings.numpy(force=True)

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=True,
        )
        embedding = model_io(tokens)
        # Correcting dimensionality
        if embedding.shape[0] > 1:
            embedding = np.mean(embedding, axis=0)
        else:
            embedding = embedding.reshape(-1)

        del text, tokens
        gc.collect()
        return np.array(embedding)

    def set_text(self, text: str, n_augmentations: int or None = 5):
        """
        Set the embedding for a given text.
        :param text: Input text to be embedded. "." are used as sentence delimiters.
        "§" is used as a "paragraph" delimiter. These are important for the augmentation.
        :param n_augmentations: The number of augmentations to be used. If None, the "original" embedding is generated.
        """
        if type(text) == list:
            raise TypeError("Text must be a string, not a list.")
        self.text = self.tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        self.n_augmentations = n_augmentations
        self.embedding = self.embedding_generator(self.replace_patterns(self.text))

        if self.n_augmentations is None:
            self.text = self.replace_patterns(self.text)
        else:
            self.text = [self.text]

            for i in range(1, self.n_augmentations + 1):
                random.seed(self.seed + i)
                self.text.append(
                    self.shuffle_sentences_within_paragraphs(
                        self.replace_patterns(self.text[0])
                    )
                )
            self.text = [self.replace_patterns(text) for text in self.text]
            self.embedding = np.array(
                [self.embedding_generator(text) for text in self.text]
            ).T

    def get_embedding(self, output="numpy"):
        if output == "pandas":
            cols = (
                [f"original"]
                + [f"shuffled_{i}" for i in range(1, self.n_augmentations + 1)]
                if self.n_augmentations is not None
                else ["original"]
            )
            return pd.DataFrame(self.embedding, columns=cols)
        elif output == "numpy":
            return self.embedding
        else:
            raise ValueError("Output must be either 'pandas' or 'numpy'.")

    def get_text(self):
        return self.text
