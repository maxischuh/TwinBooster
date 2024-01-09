import numpy as np
import optuna

from twinbooster.scripts.barlow_twins.pretraining_pipeline import run_pretraining


def objective(trial):
    # balance_ratio = trial.suggest_categorical("balance_ratio", [1, 2, 3, 5, 10])
    balance_ratio = None
    enc_n_neurons = trial.suggest_categorical("enc_n_neurons", [1024, 2048, 4096, 8192])
    enc_n_layers = trial.suggest_int("enc_n_layers", 1, 4)
    proj_n_neurons = trial.suggest_categorical(
        "proj_n_neurons", [1024, 2048, 4096, 8192]
    )
    proj_n_layers = trial.suggest_int("proj_n_layers", 1, 4)
    embedding_dim = trial.suggest_categorical("embedding_dim", [512, 1024, 2048, 4096])
    # act_function = trial.suggest_categorical("act_function", ["relu", "swish", "leaky_relu", "elu", "selu"])
    act_function = "swish"
    loss_weight = trial.suggest_float("loss_weight", 5e-3, 1e-1, log=True)
    # batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048, 4096])
    batch_size = 1024
    # epochs = trial.suggest_int("epochs", 20, 100, step=20)
    epochs = 20
    # optimizer = trial.suggest_categorical("optimizer", ["adam", "nadam", "adamax", "adamw"])
    optimizer = "adamw"
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # beta_1 = trial.suggest_float("beta_1", 0.8, 0.999)
    # beta_2 = trial.suggest_float("beta_2", 0.8, 0.999)
    beta_1, beta_2 = 0.9, 0.999
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
    step_size = trial.suggest_int("step_size", 2, 20, step=2)
    gamma = trial.suggest_float("gamma", 0.1, 0.9, step=0.1)

    model = run_pretraining(
        message="Hyperparameter tuning",
        path="../../pretraining/preprocessor.pkl",
        load_preprocessor=True,
        radius=2,
        n_bits=1024,
        balance_ratio=balance_ratio,
        num_workers=64,
        enc_n_neurons=enc_n_neurons,
        enc_n_layers=enc_n_layers,
        proj_n_neurons=proj_n_neurons,
        proj_n_layers=proj_n_layers,
        embedding_dim=embedding_dim,
        act_function=act_function,
        loss_weight=loss_weight,
        batch_size=batch_size,
        text_emb_size=768,
        epochs=epochs,
        optimizer=optimizer,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        weight_decay=weight_decay,
        step_size=step_size,
        gamma=gamma,
        hyperparameter_tuning=True,
        # val_split=0.1,        # TODO: Why? --> Way to slow
    )

    loss = model.history["train_loss"][-1]
    # quotient = loss / model.history["train_loss"][0]
    loss /= proj_n_neurons

    return loss  # * quotient


study = optuna.create_study(direction="minimize")
study.enqueue_trial(
    {
        "balance_ratio": None,
        "enc_n_neurons": 1024 * 4,
        "enc_n_layers": 2,
        "proj_n_neurons": 1024 * 2,
        "proj_n_layers": 2,
        "embedding_dim": 1024,
        "act_function": "swish",
        "loss_weight": 0.005,
        "batch_size": 512,
        "epochs": 40,
        "optimizer": "adamw",
        "learning_rate": 0.0001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0.005,
        "step_size": 10,
        "gamma": 0.1,
    }
)

trails = 50
study.optimize(objective, n_trials=trails, n_jobs=4, show_progress_bar=True)
study.trials_dataframe().to_csv(f"trials_{trails}.csv")
np.save("best_params.npy", study.best_params)
