# tournois_LSTM_vs_MLP.py

import os
import copy
import numpy as np
import torch

from utile import (
    initialze_board,
    get_legal_moves,
    has_tile_to_flip,
    BOARD_SIZE,
)

# IMPORTANT : on importe les modules qui contiennent les classes
# pour que torch.load puisse désérialiser correctement.
import LTM30     # contient les classes LSTMs, LSTMs1..LSTMs30
import MLP as MLP_module  # contient les classes MLP, MLP1..MLP30

# ==========================
#   CONFIG
# ==========================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LSTM_MODELS_DIR = "./models_LSTM"
MLP_MODELS_DIR = "./models_MLP"

# Meilleurs modèles LSTM (à adapter si besoin avec les vrais best_epoch)
TOP_LSTMS = {
    "LSTMs2":  50,
    "LSTMs11": 48,
    "LSTMs28": 50,
    "LSTMs14": 50,
    "LSTMs30": 50,
}

# Meilleurs modèles MLP (à adapter si besoin avec les vrais best_epoch)
TOP_MLPS = {
    "MLP21": 47,
    "MLP18": 50,
    "MLP11": 50,
    "MLP19": 50,
    "MLP15": 50,
}

# nombre de parties par paire LSTM/MLP (aller + retour)
GAMES_PER_PAIR = 10    # par ex : 5 avec LSTM noir, 5 avec MLP noir

# ==========================
#   FONCTIONS AUXILIAIRES
# ==========================

def input_seq_generator(board_stats_seq, length_seq):
    """Génère la séquence d'entrée (même logique que ton game.py)."""
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
    Joue une partie entre 2 modèles (LSTM ou MLP, peu importe).
    Retourne:
      +1 si noir gagne,
      -1 si blanc gagne,
       0 si nul.
    """
    board_stat = initialze_board()
    board_stats_seq = []
    pass_in_row = 0

    # On suppose que len_inpout_seq existe dans les deux modèles
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


def load_single_model(family, name, best_epoch, device):
    """
    Charge un modèle donné par son nom et son epoch.
    family : "LSTM" ou "MLP"
    name   : ex "LSTMs2" ou "MLP21"
    """
    if family == "LSTM":
        models_dir = LSTM_MODELS_DIR
    else:
        models_dir = MLP_MODELS_DIR

    folder = f"{name}_{name}"
    model_path = os.path.join(models_dir, folder, f"model_{best_epoch}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    print(f"Chargement {family} {name} (epoch {best_epoch}) depuis {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# ==========================
#   TOURNOI LSTM vs MLP
# ==========================

def main():
    print("Device :", DEVICE)

    # Chargement des meilleurs LSTMs
    lstm_models = {}
    for name, epoch in TOP_LSTMS.items():
        lstm_models[name] = load_single_model("LSTM", name, epoch, DEVICE)

    # Chargement des meilleurs MLP
    mlp_models = {}
    for name, epoch in TOP_MLPS.items():
        mlp_models[name] = load_single_model("MLP", name, epoch, DEVICE)

    # Stats individuelles sur ce tournoi croisé
    stats = {}
    for name in lstm_models.keys():
        stats[name] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}
    for name in mlp_models.keys():
        stats[name] = {"wins": 0, "draws": 0, "losses": 0, "games": 0}

    # Stats globales LSTM vs MLP
    global_stats = {
        "LSTM": {"wins": 0, "draws": 0, "losses": 0},
        "MLP":  {"wins": 0, "draws": 0, "losses": 0},
    }

    # Pour chaque paire (LSTM_i, MLP_j)
    for lstm_name, lstm_model in lstm_models.items():
        for mlp_name, mlp_model in mlp_models.items():
            print(f"\n=== Matchs {lstm_name} (LSTM) vs {mlp_name} (MLP) ===")

            # moitié des parties avec LSTM en noir
            for _ in range(GAMES_PER_PAIR // 2):
                res = play_one_game(lstm_model, mlp_model, DEVICE)
                if res == 1:   # noir (LSTM) gagne
                    stats[lstm_name]["wins"] += 1
                    stats[mlp_name]["losses"] += 1
                    global_stats["LSTM"]["wins"] += 1
                    global_stats["MLP"]["losses"] += 1
                elif res == -1:  # blanc (MLP) gagne
                    stats[mlp_name]["wins"] += 1
                    stats[lstm_name]["losses"] += 1
                    global_stats["MLP"]["wins"] += 1
                    global_stats["LSTM"]["losses"] += 1
                else:
                    stats[lstm_name]["draws"] += 1
                    stats[mlp_name]["draws"] += 1
                    global_stats["LSTM"]["draws"] += 1
                    global_stats["MLP"]["draws"] += 1
                stats[lstm_name]["games"] += 1
                stats[mlp_name]["games"] += 1

            # moitié des parties avec MLP en noir
            for _ in range(GAMES_PER_PAIR // 2):
                res = play_one_game(mlp_model, lstm_model, DEVICE)
                if res == 1:   # noir (MLP) gagne
                    stats[mlp_name]["wins"] += 1
                    stats[lstm_name]["losses"] += 1
                    global_stats["MLP"]["wins"] += 1
                    global_stats["LSTM"]["losses"] += 1
                elif res == -1:  # blanc (LSTM) gagne
                    stats[lstm_name]["wins"] += 1
                    stats[mlp_name]["losses"] += 1
                    global_stats["LSTM"]["wins"] += 1
                    global_stats["MLP"]["losses"] += 1
                else:
                    stats[lstm_name]["draws"] += 1
                    stats[mlp_name]["draws"] += 1
                    global_stats["LSTM"]["draws"] += 1
                    global_stats["MLP"]["draws"] += 1
                stats[lstm_name]["games"] += 1
                stats[mlp_name]["games"] += 1

    # calcul du score façon tournoi LSTM/MLP (1 pt victoire, 0.5 pt nul)
    for name, s in stats.items():
        s["score"] = s["wins"] + 0.5 * s["draws"]

    # Classement global sur ce tournoi croisé
    ranking = sorted(stats.items(), key=lambda kv: kv[1]["score"], reverse=True)

    print("\n===== CLASSEMENT GLOBAL LSTM vs MLP =====")
    for rank, (name, s) in enumerate(ranking, start=1):
        print(
            f"{rank:02d}. {name:<8} | score={s['score']:5.1f} | "
            f"W={s['wins']:2d} D={s['draws']:2d} L={s['losses']:2d} | games={s['games']}"
        )

    print("\n===== BILAN PAR FAMILLE =====")
    for fam in ["LSTM", "MLP"]:
        g = global_stats[fam]
        total_games = g["wins"] + g["draws"] + g["losses"]
        score = g["wins"] + 0.5 * g["draws"]
        print(
            f"{fam}: score={score:.1f} "
            f"(W={g['wins']}, D={g['draws']}, L={g['losses']}, games={total_games})"
        )


if __name__ == "__main__":
    main()
