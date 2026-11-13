import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report

from tqdm import tqdm
import numpy as np
import os
import time


def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions,target=targets)


class MLP(nn.Module):
    def __init__(self, conf):
        """
        Multi-Layer Perceptron (MLP) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        
        super(MLP, self).__init__()  
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_MLP/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]

        # Define the layers of the MLP
        self.lin1 = nn.Linear(self.board_size*self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size*self.board_size)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, seq):
        """
        Forward pass of the MLP.

        Parameters:
        - seq (torch.Tensor): A state of board as Input.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>2:
            seq=torch.flatten(seq, start_dim=1)
        else:
            seq=torch.flatten(seq, start_dim=0)
        x = F.gelu(self.lin1(seq))
        x = self.dropout(x)
        x = F.gelu(self.lin2(x))
        x = self.dropout(x)
        outp = self.lin3(x)
        return F.softmax(outp, dim=-1)
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0 # to manage earlystopping
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}sec, last_pred:{round(last_prediction)}sec)")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

               # On recharge le meilleur modèle juste pour recalculer proprement le score DEV
        best_model = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        best_model.eval()
        _clas_rep = best_model.evalulate(dev, device)
        best_dev_final = _clas_rep['weighted avg']['recall']

        print(f"Recalculing the best DEV: WAcc : {100*best_dev_final}%")

        # On garde en mémoire pour pouvoir y accéder plus tard si besoin
        self.best_epoch = best_epoch
        self.best_dev = best_dev_final

        # On retourne les deux informations
        return best_epoch, best_dev_final

    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target,_ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().detach().numpy()
            target=target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep


class MLP1(MLP):
    """
    Variante MLP1 :
    - hidden_dim = 32
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP1, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP1/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 32)
        self.lin2 = nn.Linear(32, 32)
        self.lin3 = nn.Linear(32, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP2(MLP):
    """
    Variante MLP2 :
    - hidden_dim = 64
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP2, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP2/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 64)
        self.lin2 = nn.Linear(64, 64)
        self.lin3 = nn.Linear(64, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP3(MLP):
    """
    Variante MLP3 :
    - hidden_dim = 96
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP3, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP3/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 96)
        self.lin2 = nn.Linear(96, 96)
        self.lin3 = nn.Linear(96, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP4(MLP):
    """
    Variante MLP4 :
    - hidden_dim = 160
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP4, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP4/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP5(MLP):
    """
    Variante MLP5 :
    - hidden_dim = 196
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP5, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP5/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 196)
        self.lin2 = nn.Linear(196, 196)
        self.lin3 = nn.Linear(196, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP6(MLP):
    """
    Variante MLP6 :
    - hidden_dim = 256
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP6, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP6/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 256)
        self.lin2 = nn.Linear(256, 256)
        self.lin3 = nn.Linear(256, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP7(MLP):
    """
    Variante MLP7 :
    - hidden_dim = 320
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP7, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP7/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 320)
        self.lin2 = nn.Linear(320, 320)
        self.lin3 = nn.Linear(320, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP8(MLP):
    """
    Variante MLP8 :
    - hidden_dim = 384
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP8, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP8/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 384)
        self.lin2 = nn.Linear(384, 384)
        self.lin3 = nn.Linear(384, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP9(MLP):
    """
    Variante MLP9 :
    - hidden_dim = 512
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP9, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP9/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP10(MLP):
    """
    Variante MLP10 :
    - hidden_dim = 1024
    - dropout = 0.1
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP10, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP10/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 1024)
        self.lin2 = nn.Linear(1024, 1024)
        self.lin3 = nn.Linear(1024, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.1)


class MLP11(MLP):
    """
    Variante MLP11 :
    - hidden_dim = 128
    - dropout = 0.0
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP11, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP11/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.0)


class MLP12(MLP):
    """
    Variante MLP12 :
    - hidden_dim = 128
    - dropout = 0.05
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP12, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP12/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.05)


class MLP13(MLP):
    """
    Variante MLP13 :
    - hidden_dim = 128
    - dropout = 0.15
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP13, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP13/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.15)


class MLP14(MLP):
    """
    Variante MLP14 :
    - hidden_dim = 128
    - dropout = 0.2
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP14, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP14/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.2)


class MLP15(MLP):
    """
    Variante MLP15 :
    - hidden_dim = 128
    - dropout = 0.25
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP15, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP15/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.25)
        
        
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

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100 * _clas_rep['weighted avg']['recall']}%")

        return best_epoch, best_dev

    
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
        self.hidden_dim = 256
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
        self.hidden_dim = 256
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
        self.hidden_dim = 256
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

def loss_fnc(predictions, targets):
    """
    predictions : (B, 64)
    targets     : soit (B,) indices, soit (B,64) one-hot
    """
    if targets.dim() == 2:
        # (B,64) -> (B,)
        targets = targets.argmax(dim=-1)
    return nn.CrossEntropyLoss()(predictions, targets.long())


# ==================================================================
#  CNN de base (celui que tu as déjà, corrigé sur __init__)
# ==================================================================
class CNN_Base(nn.Module):
    def __init__(self, conf):
        super(CNN_Base, self).__init__()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN_Base/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        # --- ARCHITECTURE CNN ---
        # Couche 1 : motifs simples
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)

        # Couche 2 : formes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        # Couche 3 : plus profond
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        # Couches linéaires
        self.lin1 = nn.Linear(64 * 8 * 8, 128)
        self.lin2 = nn.Linear(128, self.board_size * self.board_size)

        self.dropout = nn.Dropout(p=0.4)

    # ---------- forward commun ----------
    def _forward_features(self, x):
        # x : (B, 8, 8) ou (B,1,8,8)
        if x.dim() == 2:
            x = x.view(-1, 1, 8, 8)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        outp = self.lin2(x)
        return F.softmax(outp, dim=-1)

    # ---------- train_all & evalulate identiques pour toutes les variantes ----------
    def train_all(self, train, dev, num_epoch, device, optimizer):
        os.makedirs(self.path_save, exist_ok=True)
        best_dev = 0.0
        best_epoch = 0
        notchange = 0

        init_time = time.time()

        for epoch in range(1, num_epoch + 1):
            start_time = time.time()
            nb_batch = 0
            loss_batch = 0.0

            self.train()
            for batch, labels, _ in tqdm(train, desc=f"Epoch {epoch}/{num_epoch}"):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.to(device).float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()

            print(f"epoch : {epoch}/{num_epoch} - loss = {loss_batch / max(nb_batch,1):.4f}")
            last_training = time.time() - start_time

            self.eval()
            train_rep = self.evalulate(train, device)
            acc_train = train_rep["weighted avg"]["recall"]

            dev_rep = self.evalulate(dev, device)
            acc_dev = dev_rep["weighted avg"]["recall"]

            last_prediction = time.time() - last_training - start_time
            print(f"Accuracy Train:{100*acc_train:.2f}%, Dev:{100*acc_dev:.2f}% ; "
                  f"Time:{round(time.time()-init_time)} "
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                best_dev = acc_dev
                best_epoch = epoch
                torch.save(self.state_dict(), os.path.join(self.path_save, f"model_{epoch}.pt"))
                print(f"***** New best DEV {best_epoch} : {100*best_dev:.3f}%")
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break

        # Rechargement du meilleur modèle
        best_path = os.path.join(self.path_save, f"model_{best_epoch}.pt")
        self.load_state_dict(torch.load(best_path, map_location=device))
        self.eval()
        rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*rep['weighted avg']['recall']:.3f}%")
        return best_epoch, rep["weighted avg"]["recall"]

    def evalulate(self, test_loader, device):
        all_predicts, all_targets = [], []
        for data, target, _ in tqdm(test_loader, desc="Eval"):
            out = self(data.float().to(device))
            pred = out.argmax(dim=-1).cpu().numpy()
            tgt = target.argmax(dim=-1).numpy()
            all_predicts.extend(pred.tolist())
            all_targets.extend(tgt.tolist())

        perf_rep = classification_report(
            all_targets, all_predicts,
            zero_division=1, digits=4, output_dict=True
        )
        return perf_rep

    
    

class LSTMs(nn.Module):
    def __init__(self, conf):
        """
        Long Short-Term Memory (LSTM) model for the Othello game.

        Parameters:
        - conf (dict): Configuration dictionary containing model parameters.
        """
        super(LSTMs, self).__init__()
        
        self.board_size=conf["board_size"]
        self.path_save=conf["path_save"]+"_LSTM/"
        self.earlyStopping=conf["earlyStopping"]
        self.len_inpout_seq=conf["len_inpout_seq"]
        self.hidden_dim = conf["LSTM_conf"]["hidden_dim"]

         # Define the layers of the LSTM model
        self.lstm = nn.LSTM(self.board_size*self.board_size, self.hidden_dim,batch_first=True)
        
        #1st option: using hidden states
        # self.hidden2output = nn.Linear(self.hidden_dim*2, self.board_size*self.board_size)
        
        #2nd option: using output seauence
        self.hidden2output = nn.Linear(self.hidden_dim, self.board_size*self.board_size)
        
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, seq):
        """
        Forward pass of the LSTM model.

        Parameters:
        - seq (torch.Tensor): A series of borad states (history) as Input sequence.

        Returns:
        - torch.Tensor: Output probabilities after applying softmax.
        """
        seq=np.squeeze(seq)
        if len(seq.shape)>3:
            seq=torch.flatten(seq, start_dim=2)
        else:
            seq=torch.flatten(seq, start_dim=1)

        lstm_out, (hn, cn) = self.lstm(seq)
        
        #1st option: using hidden states as below
        # outp = self.hidden2output(torch.cat((hn,cn),-1))
        
        #2nd option: using output sequence as below 
        #(lstm_out[:,-1,:] pass only last vector of output sequence)
        if len(seq.shape)>2: # to manage the batch of sample
            # Training phase where input is batch of seq
            outp = self.hidden2output(lstm_out[:,-1,:])
            outp = F.softmax(outp, dim=1).squeeze()
        else:
            # Prediction phase where input is a single seq
            outp = self.hidden2output(lstm_out[-1,:])
            outp = F.softmax(outp).squeeze()
        
        return outp
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        dev_epoch = 0
        notchange=0
        train_acc_list=[]
        dev_acc_list=[]
        torch.autograd.set_detect_anomaly(True)
        init_time=time.time()
        for epoch in range(1, num_epoch+1):
            start_time=time.time()
            loss = 0.0
            nb_batch =  0
            loss_batch = 0
            for batch, labels, _ in tqdm(train):
                outputs =self(batch.float().to(device))
                loss = loss_fnc(outputs,labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = '+\
                  str(loss_batch/nb_batch))
            last_training=time.time()-start_time

            self.eval()
            
            train_clas_rep=self.evalulate(train, device)
            acc_train=train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep=self.evalulate(dev, device)
            acc_dev=dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction=time.time()-last_training-start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange=0
                
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
            else:
                notchange+=1
                if notchange>self.earlyStopping:
                    break
                
            self.train()
            
            print("*"*15,f"The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")

        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt')
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")

        
        return best_epoch
    
    
    def evalulate(self,test_loader, device):
        
        all_predicts=[]
        all_targets=[]
        
        for data, target_array,lengths in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted=output.argmax(dim=-1).cpu().clone().detach().numpy()
            target=target_array.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                           
        perf_rep=classification_report(all_targets,
                                      all_predicts,
                                      zero_division=1,
                                      digits=4,
                                      output_dict=True)
        perf_rep=classification_report(all_targets,all_predicts,zero_division=1,digits=4,output_dict=True)
        
        return perf_rep
            



s