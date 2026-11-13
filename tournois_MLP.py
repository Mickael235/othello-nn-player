# tournois_MLP.py

import os
import csv
import copy
import numpy as np
import torch

from utile import (
    initialze_board,
    get_legal_moves,
    has_tile_to_flip,
    BOARD_SIZE,
)

# ==========================
#   CONFIG
# ==========================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS_DIR = "./models_MLP"
RESULTS_CSV = "results_MLP_models.csv"
NB_TOP = 5          # nombre de meilleurs modèles à afficher
GAMES_PER_PAIR = 2  # 2 parties : aller/retour


# ==========================
#   FONCTIONS AUXILIAIRES
# ==========================

def input_seq_generator(board_stats_seq, length_seq):
    """Génère la séquence d'entrée pour le MLP (même logique que ton game.py)."""
    board_stat_init = initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        input_seq = [board_stat_init]
        # Padding avec plateau initial
        for _ in range(length_seq - len(board_stats_seq) - 1):
            input_seq.append(board_stat_init)
        # Ajout des états existants
        for b in board_stats_seq:
            input_seq.append(b)

    return input_seq


def find_best_move(move_prob, legal_moves):
    """Choisit le coup légal avec la probabilité la plus élevée."""
    best_move = legal_moves[0]
    max_score = move_prob[best_move[0], best_move[1]]

    for (r, c) in legal_moves:
        if move_prob[r, c] > max_score:
            max_score = move_prob[r, c]
            best_move = (r, c)
    return best_move


def apply_flip(best_move, board_stat, NgBlackPsWhith):
    """Retourne les pions (logique identique à game.py)."""
    MOVE_DIRS = [
        (-1, -1), (-1, 0), (-1, +1),
        (0, -1),           (0, +1),
        (+1, -1), (+1, 0), (+1, +1),
    ]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction, board_stat, NgBlackPsWhith):
            i = 1
            while True:
                row = best_move[0] + direction[0] * i
                col = best_move[1] + direction[1] * i
                if board_stat[row][col] == board_stat[best_move[0], best_move[1]]:
                    break
                else:
                    board_stat[row][col] = board_stat[best_move[0], best_move[1]]
                    i += 1
    return board_stat


def play_one_game(model_black, model_white, device):
    """
    Joue une partie entre 2 modèles MLP.
    Retourne:
      +1 si noir gagne,
      -1 si blanc gagne,
       0 si nul.
    """
    board_stat = initialze_board()
    board_stats_seq = []
    pass_in_row = 0

    # On suppose que len_inpout_seq existe dans le modèle (comme dans ton MLP)
    seq_len_black = model_black.len_inpout_seq
    seq_len_white = model_white.len_inpout_seq

    while True:
        # ----- Tour des noirs -----
        NgBlackPsWhith = -1
        board_stats_seq.append(copy.copy(board_stat))

        # Préparation entrée modèle noir (plateau vu du point de vue noir : * -1)
        input_seq_boards = input_seq_generator(board_stats_seq, seq_len_black)
        model_input = np.array([input_seq_boards]) * -1  # noir
        with torch.no_grad():
            probs = model_black(torch.tensor(model_input).float().to(device))
        move_prob = probs.cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)

        legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)

        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves)
            board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
            board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            pass_in_row = 0
        else:
            pass_in_row += 1
            if pass_in_row >= 2:
                break  # fin de partie

        # ----- Tour des blancs -----
        NgBlackPsWhith = +1
        board_stats_seq.append(copy.copy(board_stat))

        input_seq_boards = input_seq_generator(board_stats_seq, seq_len_white)
        model_input = np.array([input_seq_boards])  # blanc ne multiplie pas par -1
        with torch.no_grad():
            probs = model_white(torch.tensor(model_input).float().to(device))
        move_prob = probs.cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)

        legal_moves = get_legal_moves(board_stat, NgBlackPsWhith)

        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves)
            board_stat[best_move[0], best_move[1]] = NgBlackPsWhith
            board_stat = apply_flip(best_move, board_stat, NgBlackPsWhith)
            pass_in_row = 0
        else:
            pass_in_row += 1
            if pass_in_row >= 2:
                break  # fin de partie

        # condition de fin : plateau plein
        if np.all(board_stat != 0):
            break

    score = np.sum(board_stat)
    if score < 0:
        return +1   # noir gagne
    elif score > 0:
        return -1   # blanc gagne
    else:
        return 0    # nul


def load_models_from_csv(csv_path, models_dir, device):
    """
    Lit results_MLP_models.csv et charge le meilleur modèle pour chaque exp_name.
    Suppose que le chemin est : models_dir/exp_name_exp_name/model_best_epoch.pt
    """
    models = {}
    infos = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["exp_name"]
            best_epoch = int(float(row["best_epoch"]))  # parfois stocké comme float
            best_dev = float(row["best_dev_accuracy"])
            folder = f"{name}_{name}"
            model_path = os.path.join(models_dir, folder, f"model_{best_epoch}.pt")
            if not os.path.exists(model_path):
                print(f"[WARN] Modèle introuvable pour {name}: {model_path}")
                continue

            print(f"Chargement du modèle {name} (epoch {best_epoch}) depuis {model_path}")
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            models[name] = model
            infos.append({"name": name, "best_epoch": best_epoch, "best_dev": best_dev})

    return models, infos


# ==========================
#   TOURNOI
# ==========================

def main():
    print("Device :", DEVICE)

    # 1) Charger tous les modèles à partir du CSV
    models, infos = load_models_from_csv(RESULTS_CSV, MODELS_DIR, DEVICE)
    names = [info["name"] for info in infos]

    if len(names) == 0:
        print("Aucun modèle chargé. Vérifie le CSV et les dossiers ./models_MLP/...")
        return

    # Stats du tournoi
    stats = {
        name: {"wins": 0, "draws": 0, "losses": 0, "points": 0}
        for name in names
    }

    # 2) Round-robin : chaque modèle joue contre tous les autres
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name_i, name_j = names[i], names[j]
            model_i, model_j = models[name_i], models[name_j]

            print(f"\n=== Match {name_i} vs {name_j} ===")

            # Aller : i = noir, j = blanc
            for g in range(GAMES_PER_PAIR // 2):
                res = play_one_game(model_i, model_j, DEVICE)
                if res == 1:   # noir (i) gagne
                    stats[name_i]["wins"] += 1
                    stats[name_j]["losses"] += 1
                elif res == -1:  # blanc (j) gagne
                    stats[name_j]["wins"] += 1
                    stats[name_i]["losses"] += 1
                else:
                    stats[name_i]["draws"] += 1
                    stats[name_j]["draws"] += 1

            # Retour : j = noir, i = blanc
            for g in range(GAMES_PER_PAIR // 2):
                res = play_one_game(model_j, model_i, DEVICE)
                if res == 1:   # noir (j) gagne
                    stats[name_j]["wins"] += 1
                    stats[name_i]["losses"] += 1
                elif res == -1:  # blanc (i) gagne
                    stats[name_i]["wins"] += 1
                    stats[name_j]["losses"] += 1
                else:
                    stats[name_i]["draws"] += 1
                    stats[name_j]["draws"] += 1

        # 3) Calcul du score façon LSTM : 1 point par victoire, 0.5 par nul
    for name, s in stats.items():
        s["games"] = s["wins"] + s["draws"] + s["losses"]
        s["score"] = s["wins"] + 0.5 * s["draws"]

    # 4) Classement par score décroissant
    ranking = sorted(
        stats.items(),
        key=lambda kv: kv[1]["score"],
        reverse=True
    )

    # ===== Affichage style LSTM =====
    print(f"\n===== CLASSEMENT FINAL (top {NB_TOP}) =====")
    for rank, (name, s) in enumerate(ranking[:NB_TOP], start=1):
        print(
            f"{rank}. {name} - score={s['score']:.1f} "
            f"(W={s['wins']}, D={s['draws']}, L={s['losses']}, games={s['games']})"
        )

    print("\nTous les résultats :")
    for rank, (name, s) in enumerate(ranking, start=1):
        print(
            f"{rank:02d}. {name:<6} | score={s['score']:5.1f} | "
            f"W={s['wins']:2d} D={s['draws']:3d} L={s['losses']:2d} | games={s['games']}"
        )



if __name__ == "__main__":
    main()
