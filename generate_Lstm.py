# generate_lstm_variants.py

"""
Script pour générer un fichier LSTMs.py contenant :
- la classe LSTMs de base
- 20 variantes LSTMs1 ... LSTMs20
Chaque variante ne change qu'un hyperparamètre conceptuel :
    - soit hidden_dim (taille de l'état caché LSTM)
    - soit le dropout (sur la couche fully-connected finale)
"""

BASE_LSTM_CODE = '''import torch
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
            os.mkdir(f"{self.path_save}")
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

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100 * _clas_rep['weighted avg']['recall']}%")

        return best_epoch
    
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
'''

VARIANT_TEMPLATE = '''

class LSTMs{idx}(LSTMs):
    """
    Variante LSTMs{idx} :
    - hidden_dim = {hidden_dim}
    - dropout = {dropout}
    (un seul hyperparamètre conceptuel est différent de la LSTMs de base)
    """
    def __init__(self, conf):
        super(LSTMs{idx}, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_LSTMs{idx}/"

        # Redéfinition de l'architecture LSTM
        self.hidden_dim = {hidden_dim}
        self.lstm = nn.LSTM(self.board_size * self.board_size,
                            self.hidden_dim,
                            batch_first=True)

        self.hidden2output = nn.Linear(self.hidden_dim,
                                       self.board_size * self.board_size)
        self.dropout = nn.Dropout(p={dropout})
'''

def main():
    # Hyperparamètres de base
    base_hidden = 128
    base_dropout = 0.1

    # 10 variantes qui ne changent QUE hidden_dim
    hidden_variants = [32, 64, 96, 160, 196, 256, 320, 384, 512, 1024]

    # 10 variantes qui ne changent QUE le dropout (hidden_dim constant)
    dropout_variants = [0.0, 0.05, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]

    content = BASE_LSTM_CODE
    idx = 1

    # Variantes avec hidden_dim différents
    for h in hidden_variants:
        content += VARIANT_TEMPLATE.format(
            idx=idx,
            hidden_dim=h,
            dropout=base_dropout
        )
        idx += 1

    # Variantes avec dropout différents
    for d in dropout_variants:
        content += VARIANT_TEMPLATE.format(
            idx=idx,
            hidden_dim=base_hidden,
            dropout=d
        )
        idx += 1

    # Écriture du fichier LSTMs.py
    with open("LSTMs.py", "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Fichier LSTMs.py généré avec la classe LSTMs + 20 variantes LSTMs1..LSTMs20.")


if __name__ == "__main__":
    main()
