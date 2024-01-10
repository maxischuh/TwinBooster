import numpy as np
import sys
import joblib
from typing import List

from twinbooster.scripts.barlow_twins.barlow_twins import BarlowTwins
from twinbooster.scripts.utils.utils_chem import get_fp, get_mols, store_fp
from twinbooster.scripts.llm.text_embeddings import TextEmbedding


class Model:
    def __init__(self, path):
        """
        Initializes a Model object.

        Args:
            path (str): The path to the pretrained model.

        Attributes:
            path (str): The path to the pretrained model.
            model (BarlowTwins): The pretrained model.
            llm (TextEmbedding): The pretrained DeBERTa encoder.
            radius (int): The radius parameter for fingerprint generation.
            n_bits (int): The number of bits for fingerprint generation.
        """
        self.path = path
        self.model = BarlowTwins(verbose=False)
        self.model.load_model(path)
        self.llm = TextEmbedding()
        self.radius = self.model.param_dict["radius"]
        self.n_bits = self.model.param_dict["n_bits"]

    def mol_to_ecfp(self, mols):
        """
        Computes the Extended-Connectivity Fingerprint (ECFP) for a given list of molecules.

        Args:
            mols (list): A list of molecules.

        Returns:
            numpy.ndarray: The computed fingerprints.
        """
        fp = get_fp(mols, self.radius, self.n_bits)
        output = store_fp(fp, self.n_bits)
        return output

    def pair_to_embedding(self, mols, text: str, l2_norm: bool = True):
        """
        Computes the embedding for a pair of molecules and a text.

        Args:
            mols (list): A list of molecules.
            text (str): The input text.
            l2_norm (bool, optional): Whether to apply L2 normalization to the embedding. Defaults to True.

        Returns:
            numpy.ndarray: The computed embedding.
        """
        ecfp = self.mol_to_ecfp(mols)
        text_emb = self.llm.embedding_generator(text)
        text_emb = np.tile(text_emb, (len(mols), 1))
        embedding = self.model.zero_shot(ecfp, text_emb, l2_norm)
        return embedding


class TwinBooster:
    """
    TwinBooster class for predicting using Barlow Twins and LightGBM models.

    Args:
        model_path (str): Path to the Barlow Twins model.
        lgbm_path (str): Path to the LightGBM model.
        thresholds (tuple): Tuple of two thresholds for classifying predictions.

    Attributes:
        model_path (str): Path to the Barlow Twins model.
        thresholds (tuple): Tuple of two thresholds for classifying predictions.
        barlowtwins (Model): Barlow Twins model.
        lgbm (LightGBM model): LightGBM model.

    """

    def __init__(
            self, 
            model_path: str = "scripts/barlow_twins/best_model",
            lgbm_path: str = "scripts/lgbm/best_model/bt_zero_shot_model_24102023_2058_15122023_1758.joblib",
            thresholds: tuple = (0.5918970516203166, 0.30255694514339015)
        ):
        self.model_path = model_path
        self.lgbm_path = lgbm_path
        self.thresholds = thresholds

        # load models
        self.barlowtwins = Model(self.model_path)
        self.lgbm = joblib.load(self.lgbm_path)

    def predict(self, smiles: List[str], text: str, get_confidence: bool = False):
        """
        Predicts the activity of molecules based on their SMILES representation and text input.

        Args:
            smiles (List[str]): List of SMILES representations of molecules.
            text (str): Text input for prediction.
            get_confidence (bool, optional): Whether to compute confidence. Defaults to False.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Predicted probabilities or tuple of predicted probabilities and confidence.

        """
        # compute embedding
        mols = get_mols(smiles)
        embedding = self.barlowtwins.pair_to_embedding(mols, text)

        # predict
        pred = self.lgbm.predict_proba(embedding)[:, 1]

        if get_confidence:
            # compute confidence
            active_threshold, inactive_threshold = self.thresholds
            confidence = [1 if (x >= active_threshold) or (x <= inactive_threshold) else 0 for x in pred]

            return pred, confidence

        else:
            return pred