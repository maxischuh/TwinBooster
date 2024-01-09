import os
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
from rdkit import Chem, DataStructs
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys

from twinbooster.scripts.utils.utils_parallel import *
from twinbooster.scripts.utils.utils_chem import *


class Preprocessor:
    def __init__(
        self,
        path: str,
        radius: int = 2,
        n_bits: int = 1024,
        balance_ratio: int = "pos",  # None, "pos" or int
        num_workers: int = 1,
        text_embedding: str = "deberta",
    ):

        # store attributes
        self.path = path
        self.radius = radius
        self.n_bits = n_bits
        self.balance_ratio = balance_ratio
        self.num_workers = num_workers
        self.text_embedding = text_embedding

        # load raw data
        print(f"[Dataloader]: Loading data from {path}")
        # Check if files exists
        if os.path.isfile(path + "/data.pkl"):
            mol_db = pd.read_pickle(path + "/data.pkl")
        elif os.path.isfile(path + "/data.parquet"):
            mol_db = pd.read_parquet(path + "/data.parquet")
        elif os.path.isfile(path + "/data.csv"):
            mol_db = pd.read_csv(path + "/data.csv", low_memory=False)
        else:
            raise ValueError("No data file found in the specified path")

        print(f"[Dataloader]: mol_db info: {mol_db.info()}")

        # with open(path + "/embeddings.pkl", "rb") as input_file:
        #     text_db = pkl.load(input_file)
        if text_embedding == "deberta":
            print("[Dataloader]: Using DeBERTa embeddings")
            text_db = pd.read_pickle(path + "/embeddings.pkl")
        elif text_embedding == "lsa":
            print("[Dataloader]: Using LSA embeddings")
            text_db = pd.read_pickle(path + "/lsa_embeddings.pkl")
        else:
            raise ValueError("Please specify a valid text embedding")

        # selecting relevant compounds and converting AIDs to embeddings
        print(
            f"[Dataloader]: Sampling compounds (ratio = {balance_ratio}) and getting embeddings..."
        )
        processed_db = self.process_data(mol_db)
        self.embeddings = self.process_text(processed_db, text_db)

        # computing fingerprints
        print(f"[Dataloader]: Generating fingerprints using {num_workers} threads...")
        smiles = list(processed_db["SMILES"])
        if num_workers > 1:
            mols = parallel(get_mols, num_workers, smiles)
            fps = parallel(get_fp, num_workers, mols, radius, n_bits)
        else:
            mols = get_mols(smiles)
            fps = get_fp(mols, radius, n_bits)

        # storing in array
        print(f"[Dataloader]: Storing FPs...")
        self.fp = store_fp(fps, n_bits)

    def process_data(self, db: pd.DataFrame) -> pd.DataFrame:
        """
        WARNING: This function may be deprecated in the future!

        Samples molecules from the table containing smiles-activity pairs

        Loops over each column of the input dataframe, selects all active compounds
        and an equal number of inactives and stacks the SMILES, retaining the AID of
        the assay and the activity labels

        Args:
            db:     (N A) dataframe containing the full matrix of SMILES-Bioactivity
                    pairs in each assay, where N is the number of compounds and A the
                    number of AIDs

        Returns:
            Dataframe (N,3) containing trios of smiles, aid and label
        """
        # get strings with AIDs from column names
        cols = list(db.columns)[1:]

        # get smiles in list
        smiles = list(db["smiles"])

        # create containers for the final preprocessed dataset
        smi = []
        aid = []
        label = []

        # loop over all AIDs
        for col in cols:
            # select actives and a multiple of n_actives of inactives from AID
            db_actives = db.loc[db[col] == 1]
            if self.balance_ratio is None:
                db_inactives = db.loc[db[col] == 0]
            elif (
                self.balance_ratio == "pos"
            ):  # only use positive samples, set db_inactives to empty df with en empty index
                db_inactives = pd.DataFrame(columns=db.columns)
                db_inactives.index = pd.Index([])
            else:
                db_inactives = db.loc[db[col] == 0].sample(
                    len(db_actives) * self.balance_ratio, replace=True
                )

            # get respective smiles and labels from sampled dataframes
            index = list(db_actives.index) + list(db_inactives.index)
            label_temp = list(db_actives[col]) + list(db_inactives[col])
            smi_temp = [smiles[x] for x in index]

            # add smiles, labels and AID to the containers
            smi += smi_temp
            aid += [col] * len(smi_temp)
            label += label_temp

        # store in a dataframe
        output = pd.DataFrame({"SMILES": smi, "AID": aid, "Label": label})

        return output

    def process_text(self, processed_db: pd.DataFrame, text_db: pd.DataFrame):
        """Collects embeddings from AIDs

        Function that creates the appropriate embeddings for each molecule-AID pair.
        If the molecule is inactive in a given assay, the embedding is flipped. E is embedding size.

        Args:
            processed_db:   (N, 3) dataframe containing trios of smiles, aid and label
            embedding_db:   (E, X) dataframe containing the embeddings for each aid

        Returns:
            Numpy array of size (N, E) containing the correct embeddings for each
            molecule-AID pair.
        """

        # get relevant lists out of dataframe
        aid = list(processed_db["AID"])
        label = list(processed_db["Label"])

        # preallocate matrix of correct size
        embedding = np.zeros((len(aid), text_db.shape[0]))  # TODO: changed to LSA

        # store embedding or flip it depending on the label
        for i in range(len(aid)):
            if label[i] == 1:
                embedding[i, :] = np.array(text_db[int(aid[i])])
            else:
                embedding[i, :] = np.array(text_db[int(aid[i])]) * -1

        return embedding

    def __return_generator(self, device, batch_size: int = 512, shuffle: bool = True):
        """
        Utility to return Pytorch-ready generator for training
        """
        dataset = CustomDataset(device, self.fp, self.embeddings)
        generator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return generator

    def return_generator(
        self,
        device,
        batch_size: int = 512,
        shuffle: bool = True,
        validation_split: float = None,
    ) -> (DataLoader, DataLoader):
        """
        Utility to return Pytorch-ready generator for training
        """
        dataset = CustomDataset(device, self.fp, self.embeddings)
        if validation_split is not None:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(validation_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler
            )
            validation_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=valid_sampler
            )

            return train_loader, validation_loader

        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            return train_loader, None


class CustomDataset(Dataset):
    def __init__(self, device, fp, embeddings):
        self.fp = fp
        self.embeddings = embeddings
        self.device = device

    def __len__(self):
        """
        Method necessary for Pytorch training
        """
        return len(self.fp)

    def __getitem__(self, idx):
        """
        Method necessary for Pytorch training
        """
        fp_sample = torch.tensor(self.fp[idx], dtype=torch.float32)
        embedding_sample = torch.tensor(self.embeddings[idx], dtype=torch.float32)

        fp_sample = fp_sample.to(self.device)
        embedding_sample = embedding_sample.to(self.device)

        return fp_sample, embedding_sample
