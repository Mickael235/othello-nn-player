import re
import csv

def parse_log(log_text):

    results = []

    # 1) Trouver tous les blocs commençant par une ligne ===== suivie de "Expérience : MLPx"
    pattern = re.compile(
        r"=+\s*\n"                    # ligne de ======
        r"Expérience\s*:\s*(MLP\d+)\s*\n"  # Expérience : MLP9
        r"(.*?)"                      # bloc qui suit
        r"(?==+\s*\n|$)",             # jusqu'à la prochaine ligne ===== ou fin du fichier
        re.DOTALL
    )

    matches = pattern.findall(log_text)

    for exp_name, bloc in matches:

        # 2) Extraire le dict Python dans ce bloc
        dict_match = re.search(r"\{[^}]+\}", bloc, re.DOTALL)
        if not dict_match:
            # pas de dict → on ignore (rare)
            continue

        dict_text = dict_match.group(0)

        # Nettoyer <class 'MLP.MLP9'> → 'MLP9'
        dict_text_clean = re.sub(
            r"<class\s+'[^\.]+\.(\w+)'>",
            r"'\1'",
            dict_text
        )

        # Conversion dictionnaire rudimentaire
        d = {}
        for item in dict_text_clean.strip("{}").split(","):
            if ":" in item:
                key, val = item.split(":", 1)
                key = key.strip().strip("'\"")
                val = val.strip().strip("'\"")
                d[key] = val

        # 3) Extraire best_epoch et best_dev dans le bloc
        best_epoch = re.search(r"best_epoch\s*=\s*(\d+)", bloc)
        best_dev = re.search(r"best_dev\s*=\s*([\d\.]+)", bloc)

        entry = {
            "exp_name": exp_name,
            "model_class": d.get("model_class"),
            "num_params": d.get("num_params"),
            "hidden_dim": d.get("hidden_dim"),
            "dropout": d.get("dropout"),
            "optimizer": d.get("optimizer"),
            "lr": d.get("lr"),
            "batch_size": d.get("batch_size"),
            "max_epoch": d.get("max_epoch"),
            "filelist_train": d.get("filelist_train"),
            "note": d.get("note"),
            "best_epoch": best_epoch.group(1) if best_epoch else "",
            "best_dev_accuracy": best_dev.group(1) if best_dev else "",
        }

        results.append(entry)

    return results
