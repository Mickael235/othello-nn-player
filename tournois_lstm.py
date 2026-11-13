import os
import csv
import copy
import numpy as np
import torch
import torch.nn.functional as F

from utile import initialze_board, get_legal_moves, has_tile_to_flip, BOARD_SIZE

# =============================
# 1. Fonctions utilitaires
# =============================

def input_seq_generator(board_stats_seq, length_seq):
    """Construit la séquence d'entrées pour le modèle LSTM."""
    board_stat_init = initialze_board()

    if len(board_stats_seq) >= length_seq:
        input_seq = board_stats_seq[-length_seq:]
    else:
        input_seq = [board_stat_init]
        # padding avant le début réel
        for _ in range(length_seq - len(board_stats_seq) - 1):
            input_seq.append(board_stat_init)
        # puis on ajoute les vrais états
        for b in board_stats_seq:
            input_seq.append(b)

    return input_seq


def find_best_move(move_prob, legal_moves):
    """Choisit le meilleur coup parmi les coups légaux en fonction des proba du modèle."""
    best_move = legal_moves[0]
    max_score = move_prob[best_move[0], best_move[1]]
    for mv in legal_moves[1:]:
        s = move_prob[mv[0], mv[1]]
        if s > max_score:
            max_score = s
            best_move = mv
    return best_move


def apply_flip(best_move, board_stat, player_color):
    """Retourne les pions selon les règles d'Othello à partir d'un coup joué."""
    MOVE_DIRS = [(-1, -1), (-1, 0), (-1, +1),
                 (0, -1),           (0, +1),
                 (+1, -1), (+1, 0), (+1, +1)]

    for direction in MOVE_DIRS:
        if has_tile_to_flip(best_move, direction, board_stat, player_color):
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


def load_model(model_path, device):
    """Charge un modèle LSTM sauvegardé (checkpoint .pt)."""
    if torch.cuda.is_available():
        model = torch.load(model_path, weights_only=False).to(device)
    else:
        model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


# =============================
# 2. Une partie entre 2 modèles
# =============================

def play_one_game(path_black, path_white, device):
    """
    Joue une partie complète :
    - modèle A en noir (player -1)
    - modèle B en blanc (player +1)
    Retourne:
        1  si noir (A) gagne
       -1  si blanc (B) gagne
        0  si nul
    """

    model_black = load_model(path_black, device)
    model_white = load_model(path_white, device)

    len_seq_black = model_black.len_inpout_seq
    len_seq_white = model_white.len_inpout_seq

    board_stat = initialze_board()
    board_stats_seq = []
    pass2player = False

    while not np.all(board_stat) and not pass2player:
        # ---------- Tour Noir ----------
        player = -1
        board_stats_seq.append(copy.copy(board_stat))

        # séquence d'entrée pour le modèle noir
        input_seq_boards = input_seq_generator(board_stats_seq, len_seq_black)
        # noir voit le plateau avec ses pions = +1 (on multiplie par -1)
        model_input = np.array([input_seq_boards]) * -1
        move_prob = model_black(torch.tensor(model_input).float().to(device))
        move_prob = move_prob.detach().cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)

        legal_moves = get_legal_moves(board_stat, player)

        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves)
            board_stat[best_move[0], best_move[1]] = player
            board_stat = apply_flip(best_move, board_stat, player)
        else:
            # noir passe
            if len(board_stats_seq) > 1 and np.array_equal(board_stats_seq[-1], board_stats_seq[-2]):
                pass2player = True

        if np.all(board_stat) or pass2player:
            break

        # ---------- Tour Blanc ----------
        player = +1
        board_stats_seq.append(copy.copy(board_stat))

        input_seq_boards = input_seq_generator(board_stats_seq, len_seq_white)
        model_input = np.array([input_seq_boards])  # blanc voit le plateau tel quel
        move_prob = model_white(torch.tensor(model_input).float().to(device))
        move_prob = move_prob.detach().cpu().numpy().reshape(BOARD_SIZE, BOARD_SIZE)

        legal_moves = get_legal_moves(board_stat, player)
        if len(legal_moves) > 0:
            best_move = find_best_move(move_prob, legal_moves)
            board_stat[best_move[0], best_move[1]] = player
            board_stat = apply_flip(best_move, board_stat, player)
        else:
            # blanc passe
            if len(board_stats_seq) > 1 and np.array_equal(board_stats_seq[-1], board_stats_seq[-2]):
                pass2player = True

    score = np.sum(board_stat)
    if score < 0:
        return 1   # noir gagne
    elif score > 0:
        return -1  # blanc gagne
    else:
        return 0   # nul


# =============================
# 3. Préparation de la liste des modèles
# =============================

def load_best_epochs_from_csv(csv_path="results_LSTM_models1.csv"):
    """
    Lit le CSV des résultats pour récupérer, pour chaque exp_name,
    le meilleur epoch associé.
    """
    best_epochs = {}
    if not os.path.exists(csv_path):
        print(f"⚠️ Fichier CSV {csv_path} introuvable, "
              "impossible de l'utiliser pour les epochs.")
        return best_epochs

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["exp_name"]
            try:
                epoch = int(row["best_epoch"])
                best_epochs[name] = epoch
            except Exception:
                continue
    return best_epochs


def build_model_path(exp_name, best_epoch):
    """
    Construit le chemin vers le checkpoint d'un modèle LSTM donné :
    ./models_LSTM/<exp_name>_LSTM/model_<best_epoch>.pt
    """
    folder = f"./models_LSTM/{exp_name}_{exp_name}/"
    file = f"model_{best_epoch}.pt"
    path = os.path.join(folder, file)
    return path


# =============================
# 4. Tournoi round-robin
# =============================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    # Noms des 30 classes LSTM
    lstm_names = [
        "LSTMs",
        "LSTMs1", "LSTMs2", "LSTMs3", "LSTMs4", "LSTMs5",
        "LSTMs6", "LSTMs7", "LSTMs8", "LSTMs9", "LSTMs10",
        "LSTMs11", "LSTMs12", "LSTMs13", "LSTMs14", "LSTMs15",
        "LSTMs16", "LSTMs17", "LSTMs18", "LSTMs19", "LSTMs20",
        "LSTMs21", "LSTMs22", "LSTMs23", "LSTMs24", "LSTMs25",
        "LSTMs26", "LSTMs27", "LSTMs28", "LSTMs29", "LSTMs30",
    ]

    best_epochs = load_best_epochs_from_csv("results_LSTM_models2.csv")

    models = []
    for name in lstm_names:
        if name not in best_epochs:
            print(f"⚠️ Pas d'epoch trouvé pour {name} dans le CSV, je le saute.")
            continue
        epoch = best_epochs[name]
        path = build_model_path(name, epoch)
        if not os.path.exists(path):
            print(f"⚠️ Checkpoint introuvable pour {name} : {path} (je le saute)")
            continue
        models.append({"name": name, "path": path})

    print(f"\nNombre de modèles considérés pour le tournoi : {len(models)}")

    # Stats : victoires / défaites / nuls
    stats = {
        m["name"]: {"wins": 0, "losses": 0, "draws": 0}
        for m in models
    }

    n = len(models)
    # Round-robin : tous contre tous
    for i in range(n):
        for j in range(i + 1, n):
            A = models[i]
            B = models[j]
            print(f"\n=== Duel {A['name']} vs {B['name']} ===")

            # Partie 1 : A noir, B blanc
            res1 = play_one_game(A["path"], B["path"], device)
            if res1 == 1:
                stats[A["name"]]["wins"] += 1
                stats[B["name"]]["losses"] += 1
            elif res1 == -1:
                stats[A["name"]]["losses"] += 1
                stats[B["name"]]["wins"] += 1
            else:
                stats[A["name"]]["draws"] += 1
                stats[B["name"]]["draws"] += 1

            # Partie 2 : B noir, A blanc (retour)
            res2 = play_one_game(B["path"], A["path"], device)
            if res2 == 1:
                stats[B["name"]]["wins"] += 1
                stats[A["name"]]["losses"] += 1
            elif res2 == -1:
                stats[B["name"]]["losses"] += 1
                stats[A["name"]]["wins"] += 1
            else:
                stats[A["name"]]["draws"] += 1
                stats[B["name"]]["draws"] += 1

    # Calcul score (ex : 1 point victoire, 0.5 nul)
    ranking = []
    for name, st in stats.items():
        total_games = st["wins"] + st["losses"] + st["draws"]
        score = st["wins"] + 0.5 * st["draws"]
        ranking.append((name, score, st, total_games))

    ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n===== CLASSEMENT FINAL (top 5) =====")
    for i, (name, score, st, total) in enumerate(ranking[:5], start=1):
        print(f"{i}. {name} - score={score} "
              f"(W={st['wins']}, D={st['draws']}, L={st['losses']}, games={total})")

    print("\nTous les résultats :")
    for i, (name, score, st, total) in enumerate(ranking, start=1):
        print(f"{i:02d}. {name:8s} | score={score:5.1f} | "
              f"W={st['wins']:3d} D={st['draws']:3d} L={st['losses']:3d} | games={total}")


if __name__ == "__main__":
    main()
