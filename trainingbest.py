# trainingbest.py
#
# Entraînement d'un seul modèle LSTM "best" sur 100 epochs max
# avec affichage propre des barres de progression (compatible logs).
#
# Lancer :
#   python trainingbest.py
#   # ou pour sauver les logs sous PowerShell :
#   # python trainingbest.py 2>&1 | Tee-Object -FilePath logs_LSTMsBest.txt

import os
import csv
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from data import CustomDatasetOne
from utile import BOARD_SIZE


# ======================
#  DEVICE & LOSS
# ======================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", DEVICE)


def loss_fnc(predictions, targets):
    """
    Même logique que ton code de base :
    CrossEntropyLoss sur des cibles one-hot (on garde le même comportement).
    """
    return nn.CrossEntropyLoss()(input=predictions, target=targets)


# ======================
#  CONFIG TQDM (affichage)
# ======================

TQDM_ARGS = dict(
    dynamic_ncols=False,   # largeur fixe → évite le recalcul pénible dans les logs
    mininterval=1.0,       # n’actualise pas la barre plus d’1 fois par seconde
    ascii=True,            # caractères simples (#####) → compatible partout
    leave=False,           # ne laisse pas la barre une fois finie
    bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
)


# ======================
#  DATALOADERS
# ======================

def get_dataloaders(batch_size=1000, len_samples=5, filelist_train="train.txt"):
    """
    Crée les DataLoader train/dev pour le LSTM best.
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

    print(f"Number of samples (train) : {len(ds_train)}")
    print(f"Number of samples (dev)   : {len(ds_dev)}")

    return train_loader, dev_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================
#  MODELE LSTMsBest
# ======================

class LSTMsBest(nn.Module):
    """
    Modèle LSTM "best" optimisé :
    - hidden_dim configurable (par défaut 256)
    - 2 couches LSTM
    - bidirectionnel
    - dropout LSTM = 0.3
    - tête fully-connected : (hidden*2 -> hidden -> ReLU -> Dropout(0.3) -> 64)
    """

    def __init__(self, conf):
        super(LSTMsBest, self).__init__()

        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_LSTMsBest/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        # hidden_dim venant de la conf (sinon 256)
        self.hidden_dim = conf.get("LSTM_conf", {}).get("hidden_dim", 256)
        self.num_layers = 2

        # LSTM plus profond + bidirectionnel (optimisé)
        self.lstm = nn.LSTM(
            input_size=self.board_size * self.board_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=0.3,
            batch_first=True,
            bidirectional=True,
        )

        # Dropout utilisé dans la tête FC
        self.dropout = nn.Dropout(p=0.3)

        # hidden * 2 car bidirectionnel
        self.hidden2output = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(self.hidden_dim, self.board_size * self.board_size),
        )

    def forward(self, seq):
        """
        seq : (batch, seq_len, 8, 8)
        Retour : probas sur les 64 cases (softmax).
        """
        # On reste en torch (pas de np.squeeze)
        if seq.dim() == 3:  # (B, 8, 8) au cas où
            seq = seq.unsqueeze(1)

        # (B, T, 8, 8) -> (B, T, 64)
        seq = torch.flatten(seq, start_dim=2)

        lstm_out, _ = self.lstm(seq)     # (B, T, hidden*2)
        last = lstm_out[:, -1, :]        # (B, hidden*2)

        logits = self.hidden2output(last)  # (B, 64)
        probs = F.softmax(logits, dim=1)

        return probs.squeeze()

    # ====== training & eval ======

    def train_all(self, train, dev, num_epoch, device, optimizer):
        os.makedirs(self.path_save, exist_ok=True)

        best_dev = 0.0
        best_epoch = 0
        notchange = 0

        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()

        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            nb_batch = 0
            loss_batch = 0.0

            # -------- TRAIN LOOP --------
            self.train()
            train_bar = tqdm(
                train,
                desc=f"Train {epoch}/{num_epoch}",
                **TQDM_ARGS
            )
            for batch, labels, _ in train_bar:
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
                train_bar.set_postfix(loss=f"{loss_batch/max(nb_batch,1):.4f}")

            print(
                f"epoch : {epoch}/{num_epoch} - loss = {loss_batch / max(nb_batch, 1):.6f}"
            )
            last_training = time.time() - start_time

            # -------- EVAL TRAIN --------
            self.eval()
            train_clas_rep = self.evalulate(
                train, device, desc=f"Eval train {epoch}/{num_epoch}"
            )
            acc_train = train_clas_rep["weighted avg"]["recall"]

            # -------- EVAL DEV --------
            dev_clas_rep = self.evalulate(
                dev, device, desc=f"Eval dev {epoch}/{num_epoch}"
            )
            acc_dev = dev_clas_rep["weighted avg"]["recall"]

            last_prediction = time.time() - last_training - start_time

            print(
                f"Accuracy Train:{round(100 * acc_train, 2)}%, "
                f"Dev:{round(100 * acc_dev, 2)}% ; "
                f"Time:{round(time.time() - init_time)}s "
                f"(last_train:{round(last_training)}s, "
                f"last_pred:{round(last_prediction)}s)"
            )

            # -------- EARLY STOPPING --------
            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                torch.save(self, os.path.join(self.path_save, f"model_{epoch}.pt"))
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    print("Early stopping triggered.")
                    break

            print(
                "*" * 15,
                f"The best score on DEV {best_epoch} : {round(100 * best_dev, 3)}%",
            )

        # Recalcul propre du meilleur modèle
        best_model_path = os.path.join(self.path_save, f"model_{best_epoch}.pt")
        best_model = torch.load(best_model_path, map_location=device, weights_only=False)
        best_model.eval()
        _clas_rep = best_model.evalulate(dev, device, desc="Final DEV eval")
        best_dev_final = _clas_rep["weighted avg"]["recall"]

        print(f"Recalculing the best DEV: WAcc : {100 * best_dev_final}%")

        self.best_epoch = best_epoch
        self.best_dev = best_dev_final

        return best_epoch, best_dev_final

    def evalulate(self, test_loader, device, desc="Eval"):
        all_predicts = []
        all_targets = []

        eval_bar = tqdm(test_loader, desc=desc, **TQDM_ARGS)
        for data, target_array, _ in eval_bar:
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])

        perf_rep = classification_report(
            all_targets,
            all_predicts,
            zero_division=1,
            digits=4,
            output_dict=True,
        )
        return perf_rep


# ======================
#  MAIN TRAINING
# ======================

def main():
    len_samples = 5
    early_stopping = 20
    batch_size = 1000
    max_epoch = 100
    lr = 1e-2
    hidden_dim = 256

    # Data
    train_loader, dev_loader = get_dataloaders(
        batch_size=batch_size,
        len_samples=len_samples,
        filelist_train="train.txt",
    )

    # Config modèle
    conf = {
        "board_size": BOARD_SIZE,
        "path_save": "./models_LSTM/LSTMsBest",
        "earlyStopping": early_stopping,
        "len_inpout_seq": len_samples,
        "LSTM_conf": {"hidden_dim": hidden_dim},
    }

    model = LSTMsBest(conf).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = count_parameters(model)
    print(f"\nLSTMsBest : {n_params} paramètres trainables\n")

    best_epoch, best_dev = model.train_all(
        train_loader,
        dev_loader,
        num_epoch=max_epoch,
        device=DEVICE,
        optimizer=opt,
    )

    print("\n===== TRAINING TERMINÉ =====")
    print(f"Best epoch : {best_epoch}")
    print(f"Best Dev WAcc : {100 * best_dev:.3f}%")

    # Log dans un petit CSV
    csv_file = "results_LSTMsBest.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exp_name",
                "num_params",
                "hidden_dim",
                "dropout",
                "optimizer",
                "lr",
                "batch_size",
                "max_epoch",
                "best_epoch",
                "best_dev_accuracy",
            ]
        )
        writer.writerow(
            [
                "LSTMsBest",
                n_params,
                model.hidden_dim,
                model.dropout.p,
                "Adam",
                lr,
                batch_size,
                max_epoch,
                best_epoch,
                best_dev,
            ]
        )
    print("Résultats sauvegardés dans", csv_file)


if __name__ == "__main__":
    main()
