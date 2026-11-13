import torch

from torch.utils.data import DataLoader



from data import CustomDatasetMany

from utile import BOARD_SIZE

from networks_e2404931 import LSTM_Optimized2 #LSTMs




#  AJOUT : argparse



import argparse

parser = argparse.ArgumentParser()



parser.add_argument("--lr", type=float, default=0.01)

parser.add_argument("--batch_size", type=int, default=1000)

parser.add_argument("--hidden_dim", type=int, default=256)

parser.add_argument("--len_samples", type=int, default=5)

parser.add_argument("--dropout", type=float, default=0.1)



args = parser.parse_args()





if torch.cuda.is_available():

    device = torch.device("cuda:0")

else:

    device = torch.device("cpu")

    

print('Running on ' + str(device))




# REMPLACEMENT : len_samples

len_samples = args.len_samples




dataset_conf={}  

# self.filelist : a list of all games for train/dev/test

dataset_conf["filelist"]="train.txt"

#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling

dataset_conf["len_samples"]=len_samples

dataset_conf["path_dataset"]="dataset/"




# REMPLACEMENT : batch_size-

dataset_conf['batch_size']=args.batch_size



print("Training Dataste ... ")

ds_train = CustomDatasetMany(dataset_conf)




# REMPLACEMENT : batch_size

trainSet = DataLoader(ds_train, batch_size=args.batch_size)




dataset_conf={}  

# self.filelist : a list of all games for train/dev/test

dataset_conf["filelist"]="dev.txt"

#len_samples is 1 for one2one but it can be more than 1 for seq2one modeling

dataset_conf["len_samples"]=len_samples

dataset_conf["path_dataset"]="dataset/"

dataset_conf['batch_size']=args.batch_size



print("Development Dataste ... ")

ds_dev = CustomDatasetMany(dataset_conf)



devSet = DataLoader(ds_dev, batch_size=args.batch_size)




conf={}

conf["board_size"]=BOARD_SIZE

conf["path_save"]="save_models"

conf['epoch']=40

conf["earlyStopping"]=20

conf["len_inpout_seq"]=len_samples

conf["LSTM_conf"]={}



# REMPLACEMENT : hidden_dim-

conf["LSTM_conf"]["hidden_dim"]=args.hidden_dim




model = LSTM_Optimized2(conf).to(device)



# Afficher le nombre de paramètres entraînables

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Nombre de paramètres entraînables LSTM : {total_params}")






# REMPLACEMENT : learning rate

opt = torch.optim.Adam(model.parameters(), lr=args.lr)





def count_parameters(model):

    return sum(p.numel() for p in model.parameters() if p.requires_grad)



n = count_parameters(model)

print("Number of parameters: %s" % n)



best_epoch=model.train_all(trainSet,

                       devSet,

                       conf['epoch'],

                       device, opt)



# model = torch.load(conf["path_save"] + '/model_2.pt')

# model.eval()

# train_clas_rep=model.evalulate(trainSet, device)

# acc_train=train_clas_rep["weighted avg"]["recall"]

# print(f"Accuracy Train:{round(100*acc_train,2)}%")
