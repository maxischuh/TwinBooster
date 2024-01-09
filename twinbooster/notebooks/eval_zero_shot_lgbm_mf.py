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


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process model optimization and benchmarking.")
    parser.add_argument("--optimize", action='store_true', help="Flag to run optimization.")
    parser.add_argument("--benchmark", action='store_true', help="Flag to run benchmark.")
    parser.add_argument("--replicates", type=int, default=1, help="Number of replicates for benchmarking.")
    parser.add_argument("--trials", type=int, default=25, help="Number of optimization trials.")
    parser.add_argument("--significance_level", type=float, default=0.8, help="Significance level for conformal prediction.")
    parser.add_argument("--results_path", type=str, default="./results", help="Path to save the results.")
    parser.add_argument("--model_input", type=str, required=True, "../scripts/barlow_twins/best_model" help="Path to the input model.")
    return parser.parse_args()


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
        name=f"LGBM-train-{n_trials}-new",
        output_directory=Path(f"/mnt/mgs/oc2/code/twinbooster/scripts/lgbm/results/mf"),
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
    bt_model,
    dataset,
    best_params: dict,
    features_pretrain,
    labels_pretrain,
    conformal_prediction: bool = False,
    replicates: int = 1,
):
    n_test = dataset.max_task_index["test"]
    lgbm_results = []

    best_params["n_jobs"] = 64

    for replicate in trange(replicates, desc="Replicates"):
        best_params["random_state"] = replicate
        gb_model = lgbm.LGBMClassifier(**best_params)
        gb_model.fit(features_pretrain, labels_pretrain)

        if conformal_prediction:
            active_threshold, inactive_threshold = mondrian_conformal_prediction(
                gb_model, features_pretrain, labels_pretrain
            )

        if replicate == 0:
            features = np.empty((0, 2 * bt_model.param_dict["embedding_dim"]))
            labels = np.empty((0,))

        for task in trange(n_test, desc="Tasks", leave=False):
            fp, txt, y, aid = process_task(dataset, task, split="test")
            if fp is None or txt is None:
                continue

            txt_reshaped = np.tile(txt, (fp.shape[0], 1))
            zs_features = bt_model.zero_shot(fp, txt_reshaped, l2_norm=True)
            if replicate == 0:
                features = np.concatenate((features, zs_features), axis=0)
                labels = np.concatenate((labels, y), axis=0)

            baseline = np.sum(y) / len(y)

            lgbm_preds = gb_model.predict_proba(zs_features)[:, 1]
            lgbm_roc, lgbm_pr = get_metrics(y, lgbm_preds)

            if conformal_prediction:
                significance = [
                    1 if (x >= active_threshold) or (x <= inactive_threshold) else 0
                    for x in lgbm_preds
                ]
                cp = pd.DataFrame(
                    {"preds": lgbm_preds, "labels": y, "significance": significance}
                )
                cp_ratio = np.sum(cp.significance) / len(cp)
                cp = cp[cp["significance"] == 1]
                cp_roc, cp_pr = get_metrics(cp.labels, cp.preds)
                cp_baseline = np.sum(cp.labels) / len(cp)

            lgbm_results.append(
                {
                    "aid": aid,
                    "replicate": replicate,
                    "roc": lgbm_roc,
                    "pr": lgbm_pr,
                    "dpr": lgbm_pr - baseline,
                    "baseline": baseline,
                    "cp_roc": cp_roc if conformal_prediction else None,
                    "cp_pr": cp_pr if conformal_prediction else None,
                    "cp_dpr": cp_pr - cp_baseline if conformal_prediction else None,
                    "cp_baseline": cp_baseline if conformal_prediction else None,
                    "cp_ratio": cp_ratio if conformal_prediction else None,
                    "active_threshold": active_threshold
                    if conformal_prediction
                    else None,
                    "inactive_threshold": inactive_threshold
                    if conformal_prediction
                    else None,
                }
            )

    return lgbm_results, gb_model, features, labels


def main():
    args = parse_arguments()

    # Initialize the model, embeddings, and dataset
    bt_model = Model(args.model_input)
    fsmol = FSMOLDataset()
    init_text_emb()

    # Path for saving results
    os.makedirs(args.results_path, exist_ok=True)

    if args.optimize:
        # Generate data for training and validation
        pretrain_features, pretrain_labels = generate_data(
            bt_model.model, fsmol, split="train", stash=args.model_input, regenerate=False
        )
        valid_features, valid_labels = generate_data(
            bt_model.model, fsmol, split="valid", stash=args.model_input, regenerate=False
        )

        # Optimization process
        best_params = optimize(
            pretrain_features,
            pretrain_labels,
            valid_features,
            valid_labels,
            n_trials=args.trials,
        )

        # Save the optimization study
        with open(os.path.join(args.results_path, f"lgbm_{args.model_input}_study.pkl"), "wb") as f:
            pickle.dump(best_params, f)

    if args.benchmark:
        # Load best parameters if not in optimize mode
        if not args.optimize:
            with open(os.path.join(args.results_path, f"lgbm_{args.model_input}_study.pkl"), "rb") as f:
                best_params = pickle.load(f)

        # Generate data for benchmarking
        pretrain_features, pretrain_labels = generate_data(
            bt_model.model, fsmol, split="train_extra", stash=args.model_input, regenerate=False
        )

        # Benchmark process
        lgbm_results, model, features, labels = benchmark(
            bt_model.model,
            fsmol,
            best_params,
            pretrain_features,
            pretrain_labels,
            replicates=args.replicates,
            conformal_prediction=args.significance_level is not None,
        )

        # Save benchmark results
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")
        results_file_path = os.path.join(args.results_path, f"bt_zero_shot_results_{args.model_input}_{dt_string}.csv")
        lgbm_results.to_csv(results_file_path, index=False)

        # Save the model
        joblib.dump(
            model,
            os.path.join(args.results_path, f"bt_zero_shot_model_{args.model_input}_{dt_string}.joblib")
        )


if __name__ == "__main__":
    main()
