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


class MLP16(MLP):
    """
    Variante MLP16 :
    - hidden_dim = 128
    - dropout = 0.3
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP16, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP16/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.3)


class MLP17(MLP):
    """
    Variante MLP17 :
    - hidden_dim = 128
    - dropout = 0.4
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP17, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP17/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.4)


class MLP18(MLP):
    """
    Variante MLP18 :
    - hidden_dim = 128
    - dropout = 0.5
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP18, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP18/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.5)


class MLP19(MLP):
    """
    Variante MLP19 :
    - hidden_dim = 128
    - dropout = 0.7
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP19, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP19/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.7)


class MLP20(MLP):
    """
    Variante MLP20 :
    - hidden_dim = 128
    - dropout = 0.9
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP20, self).__init__(conf)
        # chemin de sauvegarde spécifique à cette variante
        self.path_save = conf["path_save"] + "_MLP20/"

        # Redéfinition de l'architecture
        self.lin1 = nn.Linear(self.board_size * self.board_size, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.9)

class MLP21(MLP):
    """
    Variante MLP21 :
    - hidden_dim = 160
    - dropout = 0.0
    (un seul hyperparamètre conceptuel est différent de la MLP de base)
    """
    def __init__(self, conf):
        super(MLP21, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP21/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.0)


class MLP22(MLP):
    """
    Variante MLP22 :
    - hidden_dim = 160
    - dropout = 0.05
    """
    def __init__(self, conf):
        super(MLP22, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP22/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.05)


class MLP23(MLP):
    """
    Variante MLP23 :
    - hidden_dim = 160
    - dropout = 0.15
    """
    def __init__(self, conf):
        super(MLP23, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP23/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.15)


class MLP24(MLP):
    """
    Variante MLP24 :
    - hidden_dim = 160
    - dropout = 0.2
    """
    def __init__(self, conf):
        super(MLP24, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP24/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.2)


class MLP25(MLP):
    """
    Variante MLP25 :
    - hidden_dim = 160
    - dropout = 0.25
    """
    def __init__(self, conf):
        super(MLP25, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP25/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.25)


class MLP26(MLP):
    """
    Variante MLP26 :
    - hidden_dim = 160
    - dropout = 0.3
    """
    def __init__(self, conf):
        super(MLP26, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP26/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.3)


class MLP27(MLP):
    """
    Variante MLP27 :
    - hidden_dim = 160
    - dropout = 0.4
    """
    def __init__(self, conf):
        super(MLP27, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP27/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.4)


class MLP28(MLP):
    """
    Variante MLP28 :
    - hidden_dim = 160
    - dropout = 0.5
    """
    def __init__(self, conf):
        super(MLP28, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP28/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.5)


class MLP29(MLP):
    """
    Variante MLP29 :
    - hidden_dim = 160
    - dropout = 0.7
    """
    def __init__(self, conf):
        super(MLP29, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP29/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.7)


class MLP30(MLP):
    """
    Variante MLP30 :
    - hidden_dim = 160
    - dropout = 0.9
    """
    def __init__(self, conf):
        super(MLP30, self).__init__(conf)
        self.path_save = conf["path_save"] + "_MLP30/"
        self.lin1 = nn.Linear(self.board_size * self.board_size, 160)
        self.lin2 = nn.Linear(160, 160)
        self.lin3 = nn.Linear(160, self.board_size * self.board_size)
        self.dropout = nn.Dropout(p=0.9)
        
        
class MLP_Best(MLP):

    def __init__(self, conf):

        super().__init__(conf)



        self.lin1 = nn.Linear(self.board_size*self.board_size, 512)

        self.bn1 = nn.BatchNorm1d(512)



        self.lin2 = nn.Linear(512, 512)

        self.bn2 = nn.BatchNorm1d(512)



        self.lin3 = nn.Linear(512, 256)

        self.bn3 = nn.BatchNorm1d(256)



        self.lin4 = nn.Linear(256, self.board_size*self.board_size)



        self.dropout = nn.Dropout(p=0.3)



        
