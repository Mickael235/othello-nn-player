import csv
import torch
from torch.utils.data import DataLoader

from data import CustomDatasetOne
from utile import BOARD_SIZE
from LSTMs import (
    LSTMs, LSTMs1, LSTMs2, LSTMs3, LSTMs4, LSTMs5,
    LSTMs6, LSTMs7, LSTMs8, LSTMs9, LSTMs10,
    LSTMs11, LSTMs12, LSTMs13, LSTMs14, LSTMs15,
    LSTMs16, LSTMs17, LSTMs18, LSTMs19, LSTMs20,
)

# ==== DEVICE =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


def get_dataloaders(batch_size=512, len_samples=5, filelist_train="train.txt"):
    """
    Crée les DataLoader train/dev pour LSTM.
    len_samples > 1 pour exploiter l'historique (séquence).
    """
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
    # Longueur de séquence pour LSTM (à adapter si besoin)
    len_samples = 5
    early_stopping = 20

    # === Définition des expériences ===
    experiments = []

    # 1) 20 variantes d'architecture (LSTMs1..LSTMs20) + LSTMs de base
    arch_classes = [
        ("LSTMs", LSTMs),
        ("LSTMs1", LSTMs1), ("LSTMs2", LSTMs2), ("LSTMs3", LSTMs3), ("LSTMs4", LSTMs4),
        ("LSTMs5", LSTMs5), ("LSTMs6", LSTMs6), ("LSTMs7", LSTMs7), ("LSTMs8", LSTMs8),
        ("LSTMs9", LSTMs9), ("LSTMs10", LSTMs10), ("LSTMs11", LSTMs11), ("LSTMs12", LSTMs12),
        ("LSTMs13", LSTMs13), ("LSTMs14", LSTMs14), ("LSTMs15", LSTMs15), ("LSTMs16", LSTMs16),
        ("LSTMs17", LSTMs17), ("LSTMs18", LSTMs18), ("LSTMs19", LSTMs19), ("LSTMs20", LSTMs20),
    ]
    for name, cls in arch_classes:
        experiments.append({
            "exp_name": name,
            "model_class": cls,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": 256,
            "max_epoch": 50,
            "filelist_train": "train.txt",
            "note": "architecture_variation",
        })

    # 2) Expériences de learning rate sur la LSTMs de base
    for lr in [1e-4, 1e-3, 1e-2, 1e-1]:
        experiments.append({
            "exp_name": f"LSTMs_lr_{lr}",
            "model_class": LSTMs,
            "optimizer": "Adam",
            "lr": lr,
            "batch_size": 512,
            "max_epoch": 60,
            "filelist_train": "train.txt",
            "note": f"lr={lr}",
        })

    # 3) Expériences batch_size / epochs
    for bs in [128, 256, 1024]:
        experiments.append({
            "exp_name": f"LSTMs_bs_{bs}",
            "model_class": LSTMs,
            "optimizer": "Adam",
            "lr": 1e-3,
            "batch_size": bs,
            "max_epoch": 60,
            "filelist_train": "train.txt",
            "note": f"batch_size={bs},epochs=60",
        })

    # 4) Expériences avec d'autres optimizers
    for opt_name in ["SGD", "RMSprop"]:
        experiments.append({
            "exp_name": f"LSTMs_opt_{opt_name}",
            "model_class": LSTMs,
            "optimizer": opt_name,
            "lr": 1e-3,
            "batch_size": 256,
            "max_epoch": 60,
            "filelist_train": "train.txt",
            "note": f"optimizer={opt_name}",
        })

    # (optionnel) 5) Expérience "all data" si tu as un fichier train_all.txt
    # experiments.append({
    #     "exp_name": "LSTMs_all_data",
    #     "model_class": LSTMs,
    #     "optimizer": "Adam",
    #     "lr": 1e-3,
    #     "batch_size": 256,
    #     "max_epoch": 60,
    #     "filelist_train": "train_all.txt",
    #     "note": "all_data",
    # })

    # === Entraînement + résumé ===
    csv_file = "results_LSTM_models1.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "exp_name", "model_class",
            "num_params", "hidden_dim", "dropout",
            "optimizer", "lr", "batch_size", "max_epoch",
            "filelist_train",
            "best_epoch", "best_dev_accuracy",
            "note",
        ]
        writer.writerow(header)

        for exp in experiments:
            print("\n" + "=" * 70)
            print(f"Expérience LSTM : {exp['exp_name']}")
            print(exp)
            print("=" * 70)

            train_loader, dev_loader = get_dataloaders(
                batch_size=exp["batch_size"],
                len_samples=len_samples,
                filelist_train=exp["filelist_train"],
            )

            conf = {
                "board_size": BOARD_SIZE,
                "path_save": f"./models_LSTM/{exp['exp_name']}",
                "earlyStopping": early_stopping,
                "len_inpout_seq": len_samples,
                "LSTM_conf": {"hidden_dim": 128},  # valeur par défaut, surchargée par les variantes
            }

            model = exp["model_class"](conf).to(device)
            opt = create_optimizer(exp["optimizer"], model.parameters(), exp["lr"])

            n_params = count_parameters(model)
            hidden_dim = getattr(model, "hidden_dim", None)
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
                exp["model_class"].__name__,
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

    print("\nRésumé des expériences LSTM enregistré dans", csv_file)


if __name__ == "__main__":
    main()
