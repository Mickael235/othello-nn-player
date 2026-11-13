import csv
import torch
from torch.utils.data import DataLoader
from docx import Document 
from data import CustomDatasetOne
from utile import BOARD_SIZE
from MLP import (
    MLP, MLP1, MLP2, MLP3, MLP4, MLP5, MLP6, MLP7, MLP8, MLP9,
    MLP10, MLP11, MLP12, MLP13, MLP14, MLP15, MLP16, MLP17, MLP18, MLP19, MLP20
)



# ==== DEVICE =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)


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
    len_samples = 1
    early_stopping = 15

    # === Définition des expériences ===
    experiments = []

    # 1) 20 variantes d'architecture (MLP1..MLP20) avec les mêmes hyperparams d'entraînement
    arch_classes = [
        ("MLP", MLP),
        ("MLP1", MLP1), ("MLP2", MLP2), ("MLP3", MLP3), ("MLP4", MLP4), ("MLP5", MLP5),
        ("MLP6", MLP6), ("MLP7", MLP7), ("MLP8", MLP8), ("MLP9", MLP9), ("MLP10", MLP10),
        ("MLP11", MLP11), ("MLP12", MLP12), ("MLP13", MLP13), ("MLP14", MLP14), ("MLP15", MLP15),
        ("MLP16", MLP16), ("MLP17", MLP17), ("MLP18", MLP18), ("MLP19", MLP19), ("MLP20", MLP20),
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


    # === Entraînement + résumé ===
    csv_file = "results_MLP_models0.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        print("Header écrit dans", csv_file)
        writer.writerow([
            "exp_name", "model_class",
            "num_params", "hidden_dim", "dropout",
            "optimizer", "lr", "batch_size", "max_epoch",
            "filelist_train",
            "best_epoch", "best_dev_accuracy",
            "note"
        ])

        for exp in experiments:
            print("\n" + "=" * 70)
            print(f"Expérience : {exp['exp_name']}")
            print(exp)
            print("=" * 70)

            train_loader, dev_loader = get_dataloaders(
                batch_size=exp["batch_size"],
                len_samples=len_samples,
                filelist_train=exp["filelist_train"]
            )

            conf = {
                "board_size": BOARD_SIZE,
                "path_save": f"./models_MLP/{exp['exp_name']}",
                "earlyStopping": early_stopping,
                "len_inpout_seq": len_samples,
            }

            model = exp["model_class"](conf).to(device)
            opt = create_optimizer(exp["optimizer"], model.parameters(), exp["lr"])

            n_params = count_parameters(model)
            # Récupérer hidden_dim et dropout depuis le modèle (si dispo)
            hidden_dim = getattr(model.lin1, "out_features", None)
            dropout_p = getattr(model.dropout, "p", None)

            best_epoch, best_dev = model.train_all(
                train_loader,
                dev_loader,
                exp["max_epoch"],
                device,
                opt
            )

            print(f">>> {exp['exp_name']} : best_epoch={best_epoch}, best_dev={100*best_dev:.3f}%")
            print("   → Ecriture CSV pour", exp["exp_name"])
            row = [
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
            exp["note"]
        ]

        writer.writerow(row)
        results_rows.append(row)   # on garde aussi pour le tableau DOCX

    print("\nRésumé des expériences enregistré dans", csv_file)

    # === Création du tableau DOCX (type Google Docs) ===
    doc = Document()
    doc.add_heading("Résumé des modèles MLP", level=1)

    table = doc.add_table(rows=1, cols=len(header))
    hdr_cells = table.rows[0].cells
    for i, col_name in enumerate(header):
        hdr_cells[i].text = col_name

    for row in results_rows:
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)

    docx_file = "results_MLP_models1.docx"
    doc.save(docx_file)
    print("Tableau DOCX enregistré dans", docx_file,
          "(importable dans Google Docs).")

            

    print("\nRésumé des expériences enregistré dans", csv_file)


if __name__ == "__main__":
    main()
