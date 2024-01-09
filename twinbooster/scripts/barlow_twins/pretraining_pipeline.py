import argparse
import torch
import time
import os

from twinbooster.scripts.preprocessor import Preprocessor, CustomDataset
from twinbooster.scripts.barlow_twins.barlow_twins import BarlowTwins


parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Pipeline arguments
# -----------------------------------------------------------------------------#

parser.add_argument(
    "--message",
    type=str,
    default="yes",
    help="Whether to ask user for a description of the run",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Which device to use for training",
)

# Preprocessing arguments
# -----------------------------------------------------------------------------#
parser.add_argument(
    "--model_type",
    type=str,
    default="barlow_twins",
    help="Which model to use",
    choices=["barlow_twins"],
)

parser.add_argument(
    "--path", type=str, help="Folder containing raw data and embeddings"
)

parser.add_argument(
    "--load_preprocessor",
    action="store_true",
    default=False,
    help="Defines whether the path points to a preprocessor",
)

parser.add_argument(
    "--radius", type=int, default=2, help="Radius to use for fingerprint calculation"
)

parser.add_argument(
    "--n_bits",
    type=int,
    default=1024,
    help="Number of bits to use for fingerprint calculation",
)

# parser.add_argument("--balance_ratio", type=int, default=None,
#                     help="Ratio between actives and inactives to keep while preprocessing")

parser.add_argument(
    "--num_workers",
    type=int,
    default=64,
    help="How many threads to use for batch generation",
)

parser.add_argument("--val_split", type=float, default=None)

# Barlow Twins arguments
# -----------------------------------------------------------------------------#

parser.add_argument(
    "--enc_n_neurons", type=int, default=4096, help="Number of neurons in the encoders"
)

parser.add_argument(
    "--enc_n_layers", type=int, default=2, help="Number of layers in the encoders"
)

parser.add_argument(
    "--proj_n_neurons",
    type=int,
    default=4096,
    help="Number of neurons in the projectors",
)

parser.add_argument(
    "--proj_n_layers", type=int, default=2, help="Number of layers in the projectors"
)

parser.add_argument(
    "--embedding_dim",
    type=int,
    default=1024,
    help="Dimensionality in the informational bottleneck",
)

parser.add_argument(
    "--act_function", type=str, default="relu", help="Activation function to use"
)

parser.add_argument(
    "--text_emb_size", type=int, default=768, help="Input text embedding size"
)

parser.add_argument(
    "--loss_weight",
    type=float,
    default=0.005,
    help="Off-diagonal loss contribution weight",
)

parser.add_argument(
    "--batch_size", type=int, default=2048, help="Batch size to use while training"
)

parser.add_argument(
    "--epochs", type=int, default=25, help="Number of epochs to use while training"
)

parser.add_argument(
    "--optimizer", type=str, default="adamw", help="Optimizer to use while training"
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate to use while training",
)

parser.add_argument(
    "--beta_1",
    type=float,
    default=0.9,
    help="First beta hyperparameter for Adam-like optimizers",
)

parser.add_argument(
    "--beta_2",
    type=float,
    default=0.999,
    help="Second beta hyperparameter for Adam-like optimizers",
)

parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-3,
    help="L2 regularization on network weights",
)

parser.add_argument(
    "--step_size",
    type=int,
    default=5,
    help="Number of epochs before decreasing learning rate",
)

parser.add_argument(
    "--gamma", type=float, default=0.1, help="Learning rate decay coefficient"
)

parser.add_argument(
    "--text_embedding",
    type=str,
    default="deberta",
    help="Which text embedding to use",
    choices=["deberta", "lsa"],
)

args = parser.parse_args()


def run_pretraining(
    message,
    path,
    load_preprocessor,
    radius,
    n_bits,
    balance_ratio,
    num_workers,
    enc_n_neurons,
    enc_n_layers,
    proj_n_neurons,
    proj_n_layers,
    embedding_dim,
    act_function,
    text_emb_size,
    loss_weight,
    batch_size,
    epochs,
    optimizer,
    learning_rate,
    beta_1,
    beta_2,
    weight_decay,
    step_size,
    gamma,
    hyperparameter_tuning=False,
    val_split=None,
    text_embedding="deberta",
    model_type="barlow_twins",
):
    # check if GPU is available
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    device = args.device

    print("---------------------------------------------------------------")
    print("Starting pretraining pipeline")
    print(f"Training will happen on {device}")
    if message == "yes":
        msg = input("Please provide a brief description of this run: ")
    else:
        msg = "No user description provided for this run"
    print("---------------------------------------------------------------")

    # start measuring time for data loading
    start = time.time()

    if not load_preprocessor:
        # load data and create generator
        data = Preprocessor(
            path, radius, n_bits, balance_ratio, text_embedding=text_embedding
        )
        train, val = data.return_generator(device, batch_size, num_workers, val_split)
        print("[Dataloader]: Saving preprocessor...")

        if text_embedding == "deberta":
            pp_path = path + "/preprocessor.pkl"
        elif text_embedding == "lsa":
            pp_path = path + "/lsa_preprocessor.pkl"
        else:
            raise ValueError("Please specify a valid text embedding")

        with open(pp_path, "wb") as file:  # TODO: changed to LSA
            pickle.dump(data, file)

    else:
        # load previously saved preprocessor
        print("[Dataloader]: Loading preprocessor...")
        with open(path, "rb") as file:
            data = pickle.load(file)
        train, val = data.return_generator(device, batch_size, num_workers, val_split)

    # print dataset size
    print(f"[Dataloader]: Loaded {len(train) * batch_size} compound-assay pairs")

    # measure time in minutes
    t_preprocessing = int((time.time() - start) / 60)

    print("---------------------------------------------------------------")
    print(f"Data loaded, process took {t_preprocessing} minutes")
    print("Moving on to network pretraining")
    print("---------------------------------------------------------------")

    # start measuring time for network pretraining
    start = time.time()

    # create model instance with correct parameters
    if model_type == "barlow_twins":
        model = BarlowTwins(
            n_bits=n_bits,
            enc_n_neurons=enc_n_neurons,
            enc_n_layers=enc_n_layers,
            proj_n_neurons=proj_n_neurons,
            proj_n_layers=proj_n_layers,
            embedding_dim=embedding_dim,
            act_function=act_function,
            text_emb_size=text_emb_size,
            loss_weight=loss_weight,
            batch_size=batch_size,
            optimizer=optimizer,
            learning_rate=learning_rate,
            betas=(beta_1, beta_2),
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma,
        )
    else:
        raise ValueError("Please specify a valid model type")

    # add scaler and fingerprint params to model
    model.consume_preprocessor(data)

    # train model
    model.move_to_device(device)
    model.train(
        train_data=train,
        val_data=val,
        num_epochs=epochs,
    )

    if hyperparameter_tuning:
        return model

    # save model in stash
    model.save_model("./stash")

    # measure time in minutes
    t_model = int((time.time() - start) / 60)

    # make log dictionary
    log = {
        arg_name: arg_value
        for arg_name, arg_value in locals().items()
        if arg_name != "self"
    }
    log["t_preprocessing"] = t_preprocessing
    log["t_model"] = t_model
    log["device"] = device

    # get newest directory name in ./stash, since it is where the model was saved
    newdir = max(
        (
            os.path.join("./stash", d)
            for d in os.listdir("./stash")
            if os.path.isdir(os.path.join("./stash", d))
        ),
        key=os.path.getctime,
        default=None,
    )
    newdir = os.path.basename(newdir)

    # save log in newest dir
    log_path = "./stash/" + newdir + "/log.txt"
    with open(log_path, "a") as file:
        file.write("----------------\n")
        file.write(f"Run description: {msg}\n")
        file.write("----------------\n")
        for arg, value in log.items():
            file.write(f"{arg}: {value}\n")

    print("---------------------------------------------------------------")
    print(f"Pretraining finished, process took {t_model} minutes")
    print(f"Saved log of run")
    print("Pretraining pipeline finished")
    print("---------------------------------------------------------------")


###############################################################################

if __name__ == "__main__":
    run_pretraining(
        message=args.message,
        path=args.path,
        load_preprocessor=args.load_preprocessor,
        radius=args.radius,
        n_bits=args.n_bits,
        balance_ratio="pos",
        num_workers=args.num_workers,
        enc_n_neurons=args.enc_n_neurons,
        enc_n_layers=args.enc_n_layers,
        proj_n_neurons=args.proj_n_neurons,
        proj_n_layers=args.proj_n_layers,
        embedding_dim=args.embedding_dim,
        act_function=args.act_function,
        text_emb_size=args.text_emb_size,
        loss_weight=args.loss_weight,
        batch_size=args.batch_size,
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.learning_rate,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        weight_decay=args.weight_decay,
        step_size=args.step_size,
        gamma=args.gamma,
        val_split=args.val_split,
        text_embedding=args.text_embedding,
        model_type=args.model_type,
    )
