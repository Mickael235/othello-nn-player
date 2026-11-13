import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time

from MLP import loss_fnc  # on réutilise la même loss que pour le MLP (si besoin tu peux la copier ici)


class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_LSTM/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

        # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size * self.board_size, self.hidden_dim, batch_first=True)
        
        # 2nd option: using output sequence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size * self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of board states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq = np.squeeze(seq)
        if len(seq.shape) > 3:
            seq = torch.flatten(seq, start_dim=2)
        else:
            seq = torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        if len(seq.shape) > 2:  # batch
            outp = self.hidden2output(lstm_out[:, -1, :])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            outp = self.hidden2output(lstm_out[-1, :])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.makedirs(self.path_save, exist_ok=True)
        best_dev = 0.0
        notchange = 0
        train_acc_list = []
        dev_acc_list = []
        torch.autograd.set_detect_anomaly(True)
        init_time = time.time()
        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            loss = 0.0
            nb_batch = 0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' +
                  str(loss_batch / nb_batch))
            last_training = time.time() - start_time

            self.eval()
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100 * acc_train, 2)}%, Dev:{round(100 * acc_dev, 2)}% ;",
                  f"Time:{round(time.time() - init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
                
            self.train()
            
        print("*" * 15, f"The best score on DEV {best_epoch} :{round(100 * best_dev, 3)}%")


        # On recharge le meilleur modèle juste pour recalculer proprement le score DEV
        best_model = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt',
                                weights_only=False)
        best_model.eval()
        _clas_rep = best_model.evalulate(dev, device)
        best_dev_final = _clas_rep['weighted avg']['recall']

        print(f"Recalculing the best DEV: WAcc : {100 * best_dev_final}%")

        # On garde en mémoire pour éventuel usage futur
        self.best_epoch = best_epoch
        self.best_dev = best_dev_final

        # On retourne les deux
        return best_epoch, best_dev_final

    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        
        for data, target_array, lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().clone().detach().numpy()
            target = target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep = classification_report(all_targets,
                                         all_predicts,
                                         zero_division=1,
                                         digits=4,
                                         output_dict=True)
        return perf_rep


class LSTMs1(LSTMs):
    """
    Variante LSTMs1 :
    - hidden_dim = 32
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs1, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs1/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 32
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs2(LSTMs):
    """
    Variante LSTMs2 :
    - hidden_dim = 64
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs2, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs2/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 64
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs3(LSTMs):
    """
    Variante LSTMs3 :
    - hidden_dim = 96
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs3, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs3/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 96
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs4(LSTMs):
    """
    Variante LSTMs4 :
    - hidden_dim = 160
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs4, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs4/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 160
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs5(LSTMs):
    """
    Variante LSTMs5 :
    - hidden_dim = 196
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs5, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs5/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 196
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs6(LSTMs):
    """
    Variante LSTMs6 :
    - hidden_dim = 256
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs6, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs6/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 256
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs7(LSTMs):
    """
    Variante LSTMs7 :
    - hidden_dim = 320
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs7, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs7/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 320
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs8(LSTMs):
    """
    Variante LSTMs8 :
    - hidden_dim = 384
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs8, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs8/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 384
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs9(LSTMs):
    """
    Variante LSTMs9 :
    - hidden_dim = 512
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs9, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs9/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 512
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs10(LSTMs):
    """
    Variante LSTMs10 :
    - hidden_dim = 1024
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs10, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs10/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 1024
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class LSTMs11(LSTMs):
    """
    Variante LSTMs11 :
    - hidden_dim = 128
    - dropout = 0.0
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs11, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs11/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.0)


class LSTMs12(LSTMs):
    """
    Variante LSTMs12 :
    - hidden_dim = 128
    - dropout = 0.05
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs12, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs12/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.05)


class LSTMs13(LSTMs):
    """
    Variante LSTMs13 :
    - hidden_dim = 128
    - dropout = 0.15
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs13, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs13/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.15)


class LSTMs14(LSTMs):
    """
    Variante LSTMs14 :
    - hidden_dim = 128
    - dropout = 0.2
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs14, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs14/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.2)


class LSTMs15(LSTMs):
    """
    Variante LSTMs15 :
    - hidden_dim = 128
    - dropout = 0.25
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs15, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs15/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.25)


class LSTMs16(LSTMs):
    """
    Variante LSTMs16 :
    - hidden_dim = 128
    - dropout = 0.3
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs16, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs16/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.3)


class LSTMs17(LSTMs):
    """
    Variante LSTMs17 :
    - hidden_dim = 128
    - dropout = 0.4
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs17, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs17/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.4)


class LSTMs18(LSTMs):
    """
    Variante LSTMs18 :
    - hidden_dim = 128
    - dropout = 0.5
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs18, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs18/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.5)


class LSTMs19(LSTMs):
    """
    Variante LSTMs19 :
    - hidden_dim = 128
    - dropout = 0.7
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs19, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs19/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.7)


class LSTMs20(LSTMs):
    """
    Variante LSTMs20 :
    - hidden_dim = 128
    - dropout = 0.9
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs20, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs20/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = 128
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.9)
