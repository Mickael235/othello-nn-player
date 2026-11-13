import os
import itertools
import subprocess
import sys

# Utilise le même Python que pour lancer ce script
PYTHON_EXE = sys.executable

# Dossier racine des modèles (adapter si besoin)
MODELS_ROOT = "models_MLP"

# Dossier d'une classe spécifique, par ex. "MLP_MLP", "MLP1_MLP1", "MLP2_MLP2", ...
CLASS_DIR = "MLP_MLP"   # <-- à changer pour tester une autre classe

BOARD_SIZE = 8  # pas vraiment utilisé ici mais garde pour cohérence


def load_model_paths(models_root, class_dir):
    """
    Retourne une liste des chemins complets vers tous les fichiers .pt
    d'un dossier de classe, par ex. models_MLP/MLP_MLP/model_1.pt
    """
    dir_path = os.path.join(models_root, class_dir)
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Dossier introuvable : {dir_path}")

    model_paths = []
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".pt"):
            model_paths.append(os.path.join(dir_path, fname))

    if len(model_paths) < 2:
        raise ValueError(f"Pas assez de modèles dans {dir_path} pour un tournoi (>=2 nécessaires).")

    return model_paths


def run_game(model1_path, model2_path):
    """
    Lance game.py pour model1 vs model2.
    game.py joue déjà 2 parties (g=0 et g=1).
    On parse la sortie pour compter victoires / défaites / nulles.

    Retourne : (wins1, wins2, draws) pour les deux parties.
    """
    cmd = [PYTHON_EXE, "game.py", model1_path, model2_path]
    print(f"\n>>> Lancement : {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    stdout = result.stdout
    stderr = result.stderr

    if result.returncode != 0:
        print("ERREUR dans game.py :")
        print(stderr)
        raise RuntimeError("game.py a échoué")

    # Debug : tu peux regarder ce que game.py a imprimé
    # print(stdout)

    wins1 = 0
    wins2 = 0
    draws = 0

    # On compte les lignes de victoire / égalité
    for line in stdout.splitlines():
        line = line.strip()
        if "is winner" in line:
            # La ligne contient un chemin de modèle
            if model1_path in line:
                wins1 += 1
            elif model2_path in line:
                wins2 += 1
        elif line.startswith("Draw"):
            draws += 1

    print(f"Résultats {os.path.basename(model1_path)} vs {os.path.basename(model2_path)} : "
          f"{wins1} victoires pour M1, {wins2} victoires pour M2, {draws} nulles")

    return wins1, wins2, draws


def tournament_for_class(models_root, class_dir):
    """
    Tournoi tous-contre-tous dans une même classe (un dossier).
    Retourne un dict: scores[model_path] = {...}
    """
    model_paths = load_model_paths(models_root, class_dir)
    print(f"\nDossier de la classe : {os.path.join(models_root, class_dir)}")
    print("Modèles trouvés :")
    for p in model_paths:
        print("  -", p)

    scores = {
        p: {"wins": 0, "losses": 0, "draws": 0, "score": 0.0}
        for p in model_paths
    }

    # Toutes les paires (i < j)
    for m1, m2 in itertools.combinations(model_paths, 2):
        w1, w2, d = run_game(m1, m2)

        # Accumule les scores
        scores[m1]["wins"] += w1
        scores[m1]["losses"] += w2
        scores[m1]["draws"] += d

        scores[m2]["wins"] += w2
        scores[m2]["losses"] += w1
        scores[m2]["draws"] += d

    # Score = victoire + 0.5 * nul
    for p, s in scores.items():
        s["score"] = s["wins"] + 0.5 * s["draws"]

    return scores


def print_ranking(scores, class_dir):
    """
    Affiche le classement des modèles d'une classe et retourne le meilleur.
    """
    print("\n===== CLASSEMENT pour la classe", class_dir, "=====")
    ordered = sorted(scores.items(), key=lambda kv: kv[1]["score"], reverse=True)

    for rank, (path, s) in enumerate(ordered, start=1):
        name = os.path.basename(path)
        print(f"{rank}. {name}  "
              f"Score={s['score']}  "
              f"({s['wins']}W, {s['losses']}L, {s['draws']}D)")

    best_path, best_scores = ordered[0]
    print("\n>>> Meilleur modèle de la classe", class_dir, ":", os.path.basename(best_path))
    print("Chemin complet :", best_path)
    return best_path, best_scores


def main():
    scores = tournament_for_class(MODELS_ROOT, CLASS_DIR)
    best_path, best_scores = print_ranking(scores, CLASS_DIR)


if __name__ == "__main__":
    main()
