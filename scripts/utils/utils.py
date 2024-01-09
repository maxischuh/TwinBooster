import sys
import argparse
import os
import pandas as pd
import pickle
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import lightgbm as lgbm
from datetime import datetime
from matplotlib import pyplot as plt
import joblib
from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical
from smac import Scenario
from smac import MultiFidelityFacade as MFFacade
from smac.intensifier.hyperband import Hyperband
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm, trange


def init_text_emb(emb_type: str = "deberta"):
    global text_emb
    if emb_type == "deberta":
        text_emb = pd.read_pickle(
            "../pretraining/embeddings.pkl"
        )
    elif emb_type == "lsa":
        text_emb = pd.read_pickle(
            "../pretraining/lsa_embeddings.pkl"
        )


def process_task(dataset, task, split):
    assay, aid = dataset.load_task(task_index=task, split=split, return_aid=True)
    if assay is None or aid is None or aid not in text_emb.columns:
        return None, None, None, None
    smiles = assay["SMILES"]
    labels = assay["Property"]
    text = text_emb[aid]
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    ecfp = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols]

    assert len(ecfp) == len(labels)
    return np.array(ecfp), np.array(text), np.array(labels), aid


def get_metrics(y, preds):
    try:
        roc = roc_auc_score(y, preds)
        pr = average_precision_score(y, preds)
    except:
        roc = None
        pr = None
    return roc, pr


def generate_data(bt_model, dataset, split: str, stash: str, regenerate: bool = False):
    assert split in ["train", "test", "valid", "train_extra"]

    try:
        if regenerate:
            raise Exception()
        features = np.load(
            f"/mnt/mgs/oc2/code/twinbooster/scripts/lgbm/bt_zero_{split}_{stash}_features.npy"
        )
        labels = np.load(
            f"/mnt/mgs/oc2/code/twinbooster/scripts/lgbm/bt_zero_{split}_{stash}_labels.npy"
        )
        print(f"Loaded precomputed {split} features")

    except:
        features = np.empty((0, 2 * bt_model.param_dict["embedding_dim"]))
        labels = np.empty((0,))

        first_split = split.split("_")[0] if "_" in split else split

        for task in trange(dataset.max_task_index[first_split]):
            fp, txt, y, aid = process_task(dataset, task, split=split)
            if fp is None or txt is None:
                continue

            txt_reshaped = np.tile(txt, (fp.shape[0], 1))
            zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
            features = np.concatenate((features, zs_features), axis=0)
            labels = np.concatenate((labels, y), axis=0)

        if "_extra" in split:
            for task in trange(dataset.max_task_index[split]):
                fp, txt, y, aid = process_task(dataset, task, split=split)
                if fp is None or txt is None:
                    continue

                txt_reshaped = np.tile(txt, (fp.shape[0], 1))
                zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
                features = np.concatenate((features, zs_features), axis=0)
                labels = np.concatenate((labels, y), axis=0)

        np.save(
            f"/mnt/mgs/oc2/code/twinbooster/scripts/lgbm/bt_zero_{split}_{stash}_features.npy",
            features,
        )
        np.save(
            f"/mnt/mgs/oc2/code/twinbooster/scripts/lgbm/bt_zero_{split}_{stash}_labels.npy",
            labels,
        )
        print(f"Saved precomputed {split} features")

    return features, labels


def generate_ablation_data(
    dataset,
    split: str,
    comment: str,
    regenerate: bool = False,
    feature_size: int = 1792,
):
    assert split in ["train", "test", "valid", "train_extra"]

    try:
        if regenerate:
            raise Exception()
        features = np.load(
            f"../scripts/lgbm/bt_zero_{split}_{comment}_features.npy"
        )
        labels = np.load(
            f"../scripts/lgbm/bt_zero_{split}_{comment}_labels.npy"
        )
        print(f"Loaded precomputed {split} ablation features and labels")

    except:
        features = np.empty((0, feature_size))
        labels = np.empty((0,))

        first_split = split.split("_")[0] if "_" in split else split

        for task in trange(dataset.max_task_index[first_split]):
            fp, txt, y, aid = process_task(dataset, task, split=split)
            if fp is None or txt is None:
                continue

            txt_reshaped = np.tile(txt, (fp.shape[0], 1))
            # zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
            concat_features = np.concatenate((fp, txt_reshaped), axis=1)
            features = np.concatenate((features, concat_features), axis=0)
            labels = np.concatenate((labels, y), axis=0)

        if "_extra" in split:
            for task in trange(dataset.max_task_index[split]):
                fp, txt, y, aid = process_task(dataset, task, split=split)
                if fp is None or txt is None:
                    continue

                txt_reshaped = np.tile(txt, (fp.shape[0], 1))
                # zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
                concat_features = np.concatenate((fp, txt_reshaped), axis=1)
                features = np.concatenate((features, concat_features), axis=0)
                labels = np.concatenate((labels, y), axis=0)

        np.save(
            f"../scripts/lgbm/bt_zero_{split}_{comment}_features",
            features,
        )
        np.save(
            f"../scripts/lgbm/bt_zero_{split}_{comment}_labels.npy",
            labels,
        )
        print(f"Saved precomputed {split} ablation features and labels")

    return features, labels


def mondrian_conformal_prediction(classifier: LightGBMClassifier, X, y, significance_level=0.8):
    """
    Performs inductive conformal prediction using 5-fold cross-validation with a pre-trained classifier.

    :param classifier: Pre-trained classifier (e.g., Random Forest).
    :param X: Features of the training set.
    :param y: Labels of the training set.
    :param significance_level: Significance level for determining thresholds.

    :return: Tuple of mean thresholds for active and inactive classes.
    """
    kf = KFold(n_splits=5)
    active_thresholds = []
    inactive_thresholds = []

    for _, cal_index in kf.split(X):
        X_cal = X[cal_index]
        y_cal = y[cal_index]

        # Predict on the calibration fold
        cal_predictions = classifier.predict_proba(X_cal)[:, 1]

        # Sort the predictions and compute thresholds
        active = np.sort(cal_predictions[y_cal == 1])
        inactive = np.sort(cal_predictions[y_cal == 0])

        if len(active) > 0:
            active_thresholds.append(
                active[int((1 - significance_level) * len(active))]
            )
        if len(inactive) > 0:
            inactive_thresholds.append(
                inactive[int(significance_level * len(inactive))]
            )

    # Compute the mean thresholds across all folds
    mean_active_threshold = np.mean(active_thresholds) if active_thresholds else None
    mean_inactive_threshold = (
        np.mean(inactive_thresholds) if inactive_thresholds else None
    )

    return mean_active_threshold, mean_inactive_threshold
