# hyperparam_MLP_top10.py

import csv
import torch
from torch.utils.data import DataLoader

from data import CustomDatasetOne
from utile import BOARD_SIZE
from MLP import (
    MLP, MLP1, MLP2, MLP3, MLP4, MLP5, MLP6, MLP7, MLP8, MLP9,
    MLP10, MLP11, MLP12, MLP13, MLP14, MLP15, MLP16, MLP17, MLP18, MLP19, MLP20,
    MLP21, MLP22, MLP23, MLP24, MLP25, MLP26, MLP27, MLP28, MLP29, MLP30
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

LEN_SAMPLES = 1
EARLY_STOPPING = 15
CSV_FILE = "results_MLP_top10_hyperparams.csv"


def get_dataloaders(batch_size=1000, len_samples=1, filelist_train="train.txt"):
    # --- Train ---
    train_conf = {
        "filelist": filelist_train,
        "len_samples": len_samples,
        "path_dataset": "dataset/",
        "batch_size": batch_size,
    }
    print(f"Training Dataset ({filelist_train}) ... ")
    ds_train = CustomDatasetOne(train_conf, load_data_once4all=True)
    train_loader = DataLoader(ds_train, batch_size=train_conf["batch_size"])

    # --- Dev ---
    dev_conf = {
        "filelist": "dev.txt",
        "len_samples": len_samples,
        "path_dataset": "dataset/",
        "batch_size": batch_size,
    }
    print("Development Dataset ... ")
    ds_dev = CustomDatasetOne(dev_conf, load_data_once4all=True)
    dev_loader = DataLoader(ds_dev, batch_size=dev_conf["batch_size"])

    return train_loader, dev_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(opt_name, params, lr):
    if opt_name == "Adam":
        return torch.optim.Adam(params, lr=lr)
    elif opt_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def main():
    # ================== TOP 10 MLP ==================
    # D’après ton classement :
    # 1. MLP21
    # 2. MLP18
    # 3. MLP11
    # 4. MLP19
    # 5. MLP15
    # Complète avec les 5 suivants.
    top_mlp_classes = [
        ("MLP21", MLP21),
        ("MLP18", MLP18),
        ("MLP11", MLP11),
        ("MLP19", MLP19),
        ("MLP15", MLP15),
        # À COMPLÉTER :
        # ("MLPxx", MLPxx),
        # ...
    ]

    experiments = []

    for name, cls in top_mlp_classes:
        # --- 1) Sweep de learning rate (Adam, 100 epochs) ---
        for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
            experiments.append({
                "exp_name": f"{name}_lr_{lr}",
                "base_name": name,
                "model_class": cls,
                "optimizer": "Adam",
                "lr": lr,
                "batch_size": 1000,
                "max_epoch": 100,
                "filelist_train": "train.txt",
                "note": f"top10_lr={lr}",
            })

        # --- 2) Optimiseurs différents (100 epochs) ---
        for opt in ["SGD", "RMSprop"]:
            experiments.append({
                "exp_name": f"{name}_opt_{opt}",
                "base_name": name,
                "model_class": cls,
                "optimizer": opt,
                "lr": 1e-3,
                "batch_size": 1000,
                "max_epoch": 100,
                "filelist_train": "train.txt",
                "note": f"top10_optimizer={opt}",
            })

        # --- 3) Impact batch_size / epochs ---
        experiments.append({
            "exp_name": f"{name}_bs500_ep60",
            "base_name": name,
            "model_class": cls,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": 500,
            "max_epoch": 60,
            "filelist_train": "train.txt",
            "note": "top10_batch=500,epochs=60",
        })
        experiments.append({
            "exp_name": f"{name}_bs1000_ep100",
            "base_name": name,
            "model_class": cls,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": 1000,
            "max_epoch": 100,
            "filelist_train": "train.txt",
            "note": "top10_batch=1000,epochs=100",
        })
        experiments.append({
            "exp_name": f"{name}_bs2000_ep100",
            "base_name": name,
            "model_class": cls,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": 2000,
            "max_epoch": 100,
            "filelist_train": "train.txt",
            "note": "top10_batch=2000,epochs=100",
        })

    # ============ ENTRAÎNEMENT ============

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "exp_name", "base_model",
            "num_params", "hidden_dim", "dropout",
            "optimizer", "lr", "batch_size", "max_epoch",
            "filelist_train",
            "best_epoch", "best_dev_accuracy",
            "note",
        ]
        writer.writerow(header)

        for exp in experiments:
            print("\n" + "=" * 70)
            print(f"Expérience MLP (top10) : {exp['exp_name']}")
            print(exp)
            print("=" * 70)

            train_loader, dev_loader = get_dataloaders(
                batch_size=exp["batch_size"],
                len_samples=LEN_SAMPLES,
                filelist_train=exp["filelist_train"],
            )

            conf = {
                "board_size": BOARD_SIZE,
                "path_save": f"./models_MLP/{exp['exp_name']}",
                "earlyStopping": EARLY_STOPPING,
                "len_inpout_seq": LEN_SAMPLES,
            }

            model = exp["model_class"](conf).to(device)
            opt = create_optimizer(exp["optimizer"], model.parameters(), exp["lr"])

            n_params = count_parameters(model)
            hidden_dim = getattr(model.lin1, "out_features", None)
            dropout_p = getattr(model.dropout, "p", None)

            best_epoch, best_dev = model.train_all(
                train_loader,
                dev_loader,
                exp["max_epoch"],
                device,
                opt,
            )

            print(f">>> {exp['exp_name']} : best_epoch={best_epoch}, "
                  f"best_dev={100 * best_dev:.3f}%")

            writer.writerow([
                exp["exp_name"],
                exp["base_name"],
                n_params,
                hidden_dim,
                dropout_p,
                exp["optimizer"],
                exp["lr"],
                exp["batch_size"],
                exp["max_epoch"],
                exp["filelist_train"],
                best_epoch,
                best_dev,
                exp["note"],
            ])

    print("\nRésumé des expériences MLP top10 enregistré dans", CSV_FILE)


if __name__ == "__main__":
    main()
