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

from twinbooster import Model, FSMOLDataset
from twinbooster.scripts.utils.utils import *


def optimize(
    features_pretrain, labels_pretrain, features_valid, labels_valid, n_trials: int = 25
):

    space = ConfigurationSpace(
        name="LGBM-hspace",
        seed=42,
        space={
            "num_leaves": Integer("num_leaves", (64, 256), q=64),
            "learning_rate": Float("learning_rate", (1e-8, 1.0), log=True),
            # "n_estimators": Integer("n_estimators", (200, 2000), q=200),
            "min_child_samples": Integer("min_child_samples", (5, 100)),
            "subsample": Float("subsample", (0.4, 1.0)),
            "subsample_freq": Integer("subsample_freq", (0, 7)),
            "reg_lambda": Float("reg_lambda", (1e-8, 10.0)),
        },
    )
    train_calls_counter = 0

    def train(config: Configuration, seed: int, budget: int):
        obj_gbm = lgbm.LGBMClassifier(
            n_estimators=int(np.ceil(budget)),
            random_state=seed,
            n_jobs=64,
            **config,
        )
        obj_gbm.fit(
            features_pretrain, labels_pretrain
        )  # , eval_set=[(features_valid, labels_valid)])
        obj_preds = obj_gbm.predict_proba(features_valid)[:, 1]
        roc, pr = get_metrics(
            labels_valid, obj_preds
        )  # assuming you have a function to get metrics

        result = 2 - pr - roc
        nonlocal train_calls_counter  # Use nonlocal to modify the counter declared outside the function
        train_calls_counter += 1
        print(f"({train_calls_counter}/{n_trials}) ROC: {roc}, PR: {pr}, min: {result}")
        return result

    scenario = Scenario(
        configspace=space,
        name=f"LGBM-train-{n_trials}-new-ablation",
        # walltime_limit=120,  # Limit to two minutes
        n_trials=n_trials,  # Evaluated max 500 trials
        # n_workers=4,  # Use eight workers
        min_budget=100,  # Minimum budget
        max_budget=2000,  # Maximum budget
        seed=42,
    )

    intensifier = Hyperband(scenario=scenario)

    smac = MFFacade(
        scenario=scenario,
        target_function=train,
        intensifier=intensifier,
    )
    incumbent = smac.optimize()

    print(f"Optimized parameters: {incumbent}")

    return incumbent


def benchmark(
    dataset,
    best_params: dict,
    features_pretrain,
    labels_pretrain,
    feature_size: int,
    replicates: int,
):
    n_test = dataset.max_task_index["test"]
    lgbm_results = []

    best_params["n_jobs"] = 64

    for replicate in trange(replicates, desc="Replicates"):
        best_params["random_state"] = replicate
        gb_model = lgbm.LGBMClassifier(**best_params)
        gb_model.fit(features_pretrain, labels_pretrain)

        if replicate == 0:
            features = np.empty((0, feature_size))
            labels = np.empty((0,))

        for task in trange(n_test, desc="Tasks", leave=False):
            fp, txt, y, aid = process_task(dataset, task, split="test")
            if fp is None or txt is None:
                continue

            txt_reshaped = np.tile(txt, (fp.shape[0], 1))
            # zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
            concat_features = np.concatenate((fp, txt_reshaped), axis=1)
            if replicate == 0:
                features = np.concatenate((features, concat_features), axis=0)
                labels = np.concatenate((labels, y), axis=0)

            baseline = np.sum(y) / len(y)

            # Get lgbm predictions
            lgbm_preds = gb_model.predict_proba(concat_features)[:, 1]
            lgbm_roc, lgbm_pr = get_metrics(y, lgbm_preds)

            lgbm_results.append(
                {
                    "aid": aid,
                    "replicate": replicate,
                    "roc": lgbm_roc,
                    "pr": lgbm_pr,
                    "dpr": lgbm_pr - baseline,
                    "baseline": baseline,
                }
            )

    return lgbm_results, gb_model, features, labels


def main():
    OPTIMIZE = False
    BENCHMARK = True
    text_emb_type = "lsa"

    init_text_emb(text_emb_type)  # deberta or lsa
    comment = f"abl_ecfp+{text_emb_type}"
    feature_size = 1024 + text_emb.shape[0]  # ECFP + DeBERTa or ECFP + LSA

    if OPTIMIZE:
        pretrain_features, pretrain_labels = generate_ablation_data(
            dataset=fsmol,
            split="train",
            feature_size=feature_size,
            comment=comment,
            regenerate=False,
        )
        valid_features, valid_labels = generate_ablation_data(
            dataset=fsmol,
            split="valid",
            feature_size=feature_size,
            comment=comment,
            regenerate=False,
        )
        (
            pretrain_features,
            plus_features,
            pretrain_labels,
            plus_labels,
        ) = train_test_split(
            pretrain_features, pretrain_labels, test_size=0.2, random_state=42
        )
        valid_features = np.concatenate((valid_features, plus_features), axis=0)
        valid_labels = np.concatenate((valid_labels, plus_labels), axis=0)

        # Shuffle the features and labels but keep them aligned
        shuffle_idx = np.arange(pretrain_features.shape[0])
        np.random.shuffle(shuffle_idx)
        pretrain_features = pretrain_features[shuffle_idx]
        pretrain_labels = pretrain_labels[shuffle_idx]
        shuffle_idx = np.arange(valid_features.shape[0])
        np.random.shuffle(shuffle_idx)
        valid_features = valid_features[shuffle_idx]
        valid_labels = valid_labels[shuffle_idx]

        study = optimize(
            pretrain_features,
            pretrain_labels,
            valid_features,
            valid_labels,
            n_trials=200,
        )
        pickle.dump(
            study,
            open(
                f"../scripts/lgbm/results/mf/lgbm_{comment}_study.pkl",
                "wb",
            ),
        )

    elif BENCHMARK:
        pretrain_features, pretrain_labels = generate_ablation_data(
            dataset=fsmol,
            split="train_extra",
            feature_size=feature_size,
            comment=comment,
            regenerate=False,
        )
        valid_features, valid_labels = generate_ablation_data(
            dataset=fsmol,
            split="valid",
            feature_size=feature_size,
            comment=comment,
            regenerate=False,
        )

        # Shuffle the features and labels but keep them aligned
        shuffle_idx = np.arange(pretrain_features.shape[0])
        np.random.shuffle(shuffle_idx)
        pretrain_features = pretrain_features[shuffle_idx]
        pretrain_labels = pretrain_labels[shuffle_idx]
        shuffle_idx = np.arange(valid_features.shape[0])
        np.random.shuffle(shuffle_idx)
        valid_features = valid_features[shuffle_idx]
        valid_labels = valid_labels[shuffle_idx]

        best_params = dict(
            pd.read_pickle(
                f"../scripts/lgbm/results/mf/lgbm_{comment}_study.pkl"
            )
        )
        best_params["n_estimators"] = 2000
        print(f"Loaded best params: {best_params}")
        lgbm_results, model, features, labels = benchmark(
            fsmol,
            best_params,
            pretrain_features,
            pretrain_labels,
            feature_size=feature_size,
            replicates=10,
        )

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")
        # Create directory if it doesn't exist
        if not os.path.exists(
            f"../scripts/lgbm/results/ablations/{dt_string}"
        ):
            os.makedirs(
                f"../scripts/lgbm/results/ablations/{dt_string}"
            )

        # Grouping and averaging the results
        lgbm_results = pd.DataFrame(lgbm_results)
        lgbm_results.to_csv(
            f"../scripts/lgbm/results/ablations/{dt_string}/bt_zero_shot_results_{comment}_{dt_string}.csv",
            index=False,
        )

        print(f"dpr: {lgbm_results['dpr'].mean()}({lgbm_results['dpr'].std()})")
        print(f"roc: {lgbm_results['roc'].mean()}({lgbm_results['roc'].std()})")

        plt.plot(0, lgbm_results["dpr"].mean(), "go", label="Ablation TB (ours)")

        # Plotting the FSMol results
        fsmol_results = pd.read_csv(
            "/mnt/mgs/nas/code/FS-Mol/values_df.csv", index_col=0
        )
        fsmol_results_std = pd.read_csv(
            "/mnt/mgs/nas/code/FS-Mol/stds_df.csv", index_col=0
        )
        for col in fsmol_results.columns:
            plt.errorbar(
                fsmol_results.index,
                fsmol_results[col],
                yerr=fsmol_results_std[col],
                label=col,
                fmt="o",
                linestyle="dashed",
                alpha=0.75,
            )

        plt.legend()
        plt.title("Zero-shot $\Delta$PRAUC")
        plt.xlabel("Anchor Size")
        plt.xticks([0] + list(fsmol_results.index))
        plt.grid()

        plt.ylabel("$\Delta$PRAUC")
        plt.savefig(
            f"../scripts/lgbm/results/ablations/{dt_string}/bt_zero_shot_dpr_{comment}_{dt_string}.pdf",
            bbox_inches="tight",
        )

        # Save model
        joblib.dump(
            model,
            f"../scripts/lgbm/results/ablations/{dt_string}/bt_zero_shot_model_{comment}_{dt_string}.joblib",
        )


if __name__ == "__main__":
    main()
