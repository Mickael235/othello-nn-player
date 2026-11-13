# trainingMLPBest.py
#
# Entraînement du "meilleur" MLP sur 80 epochs max (avec early stopping)
# Hyperparamètres d'après le résumé :
#   - hidden_dim = 512 → 512 → 256 → output
#   - optimizer = SGD (momentum 0.9)
#   - lr = 0.001
#   - batch_size = 32
#   - epochs ~75-80
#
# Lancer :
#   python trainingMLPBest.py

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm
import numpy as np
import os
import time

from data import CustomDatasetOne
from utile import BOARD_SIZE

# ==== DEVICE =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


# ==== LOSS FUNCTION =====
def loss_fnc(predictions, targets):
    # CrossEntropyLoss attend des logits bruts et des labels entiers
    return nn.CrossEntropyLoss()(predictions, targets.argmax(dim=-1))


# ==== CLASS MLP_Best =====
class MLP_Best(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_MLPBest/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        # Architecture optimisée
        self.lin1 = nn.Linear(self.board_size * self.board_size, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.lin2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.lin3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.lin4 = nn.Linear(256, self.board_size * self.board_size)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, seq):
        seq = np.squeeze(seq)
        if len(seq.shape) > 2:
            seq = torch.flatten(seq, start_dim=1)
        else:
            seq = torch.flatten(seq, start_dim=0)

        x = F.relu(self.bn1(self.lin1(seq)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.lin3(x)))
        x = self.dropout(x)
        outp = self.lin4(x)  # logits bruts
        return outp

    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")

        best_dev = 0.0
        best_epoch = 0
        notchange = 0

        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()

        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss_batch = 0
            nb_batch = 0

            self.train()
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()

            print(f"epoch : {epoch}/{num_epoch} - loss = {loss_batch/nb_batch:.4f}")

            self.eval()
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]

            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]

            print(f"Accuracy Train:{100*acc_train:.2f}%, Dev:{100*acc_dev:.2f}%")

            if acc_dev > best_dev:
                notchange = 0
                torch.save(self, self.path_save + f"/model_{epoch}.pt")
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

        print("*" * 15, f"The best score on DEV {best_epoch} : {100*best_dev:.3f}%")
        return best_epoch, best_dev

    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            all_predicts.extend(predicted)
            all_targets.extend(target)
        perf_rep = classification_report(
            all_targets, all_predicts, zero_division=1, digits=4, output_dict=True
        )
        return perf_rep


# ==== DATA LOADER =====
def get_dataloaders(batch_size=32, len_samples=1, filelist_train="train.txt"):
    train_conf = {
        "filelist": filelist_train,
        "len_samples": len_samples,
        "path_dataset": "dataset/",
        "batch_size": batch_size,
    }
    ds_train = CustomDatasetOne(train_conf, load_data_once4all=True)
    train_loader = DataLoader(ds_train, batch_size=train_conf["batch_size"])

    dev_conf = {
        "filelist": "dev.txt",
        "len_samples": len_samples,
        "path_dataset": "dataset/",
        "batch_size": batch_size,
    }
    ds_dev = CustomDatasetOne(dev_conf, load_data_once4all=True)
    dev_loader = DataLoader(ds_dev, batch_size=dev_conf["batch_size"])

    return train_loader, dev_loader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==== MAIN =====
def main():
    len_samples = 1
    early_stopping = 20
    batch_size = 32
    max_epoch = 80
    lr = 1e-3

    exp_name = "MLPBest"

    train_loader, dev_loader = get_dataloaders(batch_size=batch_size, len_samples=len_samples)

    conf = {
        "board_size": BOARD_SIZE,
        "path_save": f"./models_MLP/{exp_name}",
        "earlyStopping": early_stopping,
        "len_inpout_seq": len_samples,
    }

    model = MLP_Best(conf).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    n_params = count_parameters(model)
    print(f"{exp_name} : {n_params} paramètres trainables")

    best_epoch, best_dev = model.train_all(train_loader, dev_loader, max_epoch, device, optimizer)

    print("\n===== TRAINING TERMINÉ =====")
    print(f"Best epoch : {best_epoch}")
    print(f"Best Dev WAcc : {100*best_dev:.3f}%")

    csv_file = "results_MLPBest.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_name", "model_class", "num_params", "optimizer", "lr",
            "batch_size", "max_epoch", "best_epoch", "best_dev_accuracy"
        ])
        writer.writerow([
            exp_name, model.__class__.__name__, n_params, "SGD", lr,
            batch_size, max_epoch, best_epoch, best_dev
        ])

    print("Résultats sauvegardés dans", csv_file)


if __name__ == "__main__":
    main()
