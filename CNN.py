class CNN(nn.Module):
    def _init_(self, conf):
        super(CNN, self)._init_()
        self.board_size = conf["board_size"]
        self.path_save = conf["path_save"] + "_CNN/"
        self.earlyStopping = conf["earlyStopping"]
        self.len_inpout_seq = conf["len_inpout_seq"]

        # --- ARCHITECTURE CNN ---
        # Couche 1 : Détecte des motifs simples
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) 

        # Couche 2 : Détecte des formes
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Couche 3 : Encore plus profond
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Couches linéaires pour la décision finale
        self.lin1 = nn.Linear(64 * 8 * 8, 128)
        self.lin2 = nn.Linear(128, self.board_size * self.board_size)
        
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        # Gestion du format de l'image (Batch, 1, 8, 8)
        if x.dim() == 2:
            x = x.view(-1, 1, 8, 8)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Aplatir pour les couches linéaires
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        outp = self.lin2(x)
        
        return F.softmax(outp, dim=-1)

    # --- MÉTHODES D'ENTRAÎNEMENT (Celles qui manquaient) ---
    
    def train_all(self, train, dev, num_epoch, device, optimizer):
        if not os.path.exists(f"{self.path_save}"):
            os.mkdir(f"{self.path_save}")
        best_dev = 0.0
        notchange = 0 
        train_acc_list = []
        dev_acc_list = []
        
        init_time = time.time()
        
        for epoch in range(1, num_epoch+1):
            start_time = time.time()
            nb_batch = 0
            loss_batch = 0
            
            self.train() # Mode entraînement
            for batch, labels, _ in tqdm(train):
                outputs = self(batch.float().to(device))
                loss = loss_fnc(outputs, labels.clone().detach().float().to(device))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                nb_batch += 1
                loss_batch += loss.item()
            
            print("epoch : " + str(epoch) + "/" + str(num_epoch) + ' - loss = ' + str(loss_batch/nb_batch))
            last_training = time.time() - start_time

            self.eval() # Mode évaluation
            
            train_clas_rep = self.evalulate(train, device)
            acc_train = train_clas_rep["weighted avg"]["recall"]
            train_acc_list.append(acc_train)
            
            dev_clas_rep = self.evalulate(dev, device)
            acc_dev = dev_clas_rep["weighted avg"]["recall"]
            dev_acc_list.append(acc_dev)
            
            last_prediction = time.time() - last_training - start_time
            
            print(f"Accuracy Train:{round(100*acc_train,2)}%, Dev:{round(100*acc_dev,2)}% ;",
                  f"Time:{round(time.time()-init_time)}",
                  f"(last_train:{round(last_training)}, last_pred:{round(last_prediction)})")

            if acc_dev > best_dev or best_dev == 0.0:
                notchange = 0
                torch.save(self, self.path_save + '/model_' + str(epoch) + '.pt')
                best_dev = acc_dev
                best_epoch = epoch
                print(f"***** The best score on DEV {best_epoch} :{round(100*best_dev,3)}%")
            else:
                notchange += 1
                if notchange > self.earlyStopping:
                    break
            
        self = torch.load(self.path_save + '/model_' + str(best_epoch) + '.pt', weights_only=False)
        self.eval()
        _clas_rep = self.evalulate(dev, device)
        print(f"Recalculing the best DEV: WAcc : {100*_clas_rep['weighted avg']['recall']}%")
        return best_epoch

    def evalulate(self, test_loader, device):
        all_predicts = []
        all_targets = []
        
        for data, target, _ in tqdm(test_loader):
            output = self(data.float().to(device))
            predicted = output.argmax(dim=-1).cpu().detach().numpy()
            target = target.argmax(dim=-1).numpy()
            for i in range(len(predicted)):
                all_predicts.append(predicted[i])
                all_targets.append(target[i])
                            
        perf_rep = classification_report(all_targets, all_predicts, zero_division=1, digits=4, output_dict=True)
        return perf_rep
        
        
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
