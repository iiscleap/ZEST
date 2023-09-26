import os
import torch
import torchaudio
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import torch.nn.functional as F
import pickle5 as pickle
import ast
from torch.autograd import Function

torch.set_printoptions(profile="full")
#Logger set
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
torch.autograd.set_detect_anomaly(True)
#CUDA devices enabled
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

class MyDataset(Dataset):
    def __init__(self, folder, speaker_folder):
        self.folder = folder
        self.speaker_folder = speaker_folder
        wav_files = os.listdir(speaker_folder)
        wav_files = [x for x in wav_files if ".npy" in x]
        wav_files = [x.replace("_gen.wav", ".wav") for x in wav_files]
        speaker_features = os.listdir(speaker_folder)
        speaker_features = [x for x in speaker_features if ".npy" in x]
        self.wav_files = wav_files
        self.speaker_features = speaker_features
        self.sr = 16000
        self.speaker_dict = {}
        for ind in range(11, 21):
            self.speaker_dict["00"+str(ind)] = ind-11

    def __len__(self):
        return len(self.wav_files) 

    def getspkrlabel(self, file_name):
        spkr_name = file_name[-15:][:4]
        spkr_label = self.speaker_dict[spkr_name]

        return spkr_label

    def getemolabel(self, file_name):
        file_name = int(file_name[5:-4])
        if file_name <=350:
            return 0
        elif file_name > 350 and file_name <=700:
            return 1
        elif file_name > 700 and file_name <= 1050:
            return 2
        elif file_name > 1050 and file_name <= 1400:
            return 3
        else:
            return 4
        
    def __getitem__(self, audio_ind):
        speaker_feat = np.load(os.path.join(self.speaker_folder, self.wav_files[audio_ind].replace(".wav", ".npy")))
        speaker_label = self.getspkrlabel(self.wav_files[audio_ind][-15:])
        class_id = self.getemolabel(self.wav_files[audio_ind]) 

        return speaker_feat, speaker_label, class_id, self.wav_files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class SpeakerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(192, 128)
        self.fc = nn.Linear(128, 128)
        self.fc_embed = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_embed_1 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)
        self.fc4 = nn.Linear(128, 128)
        self.fc_embed_2 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 5)
    
    def forward(self, feat, alpha=1.0):
        feat = self.fc(self.fc_embed(self.fc1(feat)))
        reverse = ReverseLayerF.apply(feat, alpha)
        out = self.fc3(self.fc_embed_1(self.fc2(feat)))
        emo_out = self.fc5(self.fc_embed_2(self.fc4(reverse)))
        
        return out, emo_out, feat

def create_dataset(mode, bs=32):
    speaker_folder = "/folder/to/x-vectors"
    if mode == 'train':
        folder = "/folder/to/train/audio/files"
    elif mode == 'val':
        folder = "/folder/to/validation/audio/files"
    elif mode =="test":
        folder = "/folder/to/test/audio/files"
    dataset = MyDataset(folder, speaker_folder)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False)

    return loader

def train():
    
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = SpeakerModel()
    model.to(device)
    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params':parameters, 'lr':base_lr}])
    final_val_loss = 1e20

    for e in range(10):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs, out_emo, _ = model(speaker_feat, alpha)
            loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
            loss_emo = nn.CrossEntropyLoss(reduction='mean')(out_emo, emo_labels)
            loss += 10*loss_emo
            tot_loss += loss.detach().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred = torch.argmax(outputs, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(val_loader)):
                speaker_feat, labels, emo_labels = data[0].to(device), data[1].to(device), data[2].to(device)
                outputs, out_emo, _ = model(speaker_feat)
                loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
                val_loss += loss.detach().item()
                pred = torch.argmax(outputs, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save(model, 'EASE.pth')
            final_val_loss = val_loss
        train_loss = tot_loss/len(train_loader)
        train_f1 = accuracy_score(gt_tr, pred_tr)
        val_loss_log = val_loss/len(val_loader)
        val_f1 = accuracy_score(gt_val, pred_val)
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")

def get_embedding():
    train_loader = create_dataset("train", 1)
    val_loader = create_dataset("val", 1)
    test_loader = create_dataset("test", 1)
    model = torch.load('EASE.pth', map_location=device)
    model.to(device)
    model.eval()
    os.makedirs("EASE_embeddings", exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join("EASE_embeddings", target_file_name), embedded[ind, :].cpu().detach().numpy())

        for i, data in enumerate(tqdm(val_loader)):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join("EASE_embeddings", target_file_name), embedded[ind, :].cpu().detach().numpy())  
        
        for i, data in enumerate(tqdm(test_loader)):
            speaker_feat, labels = data[0].to(device), data[1].to(device)
            names = data[3]
            _, _, embedded = model(speaker_feat)
            for ind in range(len(names)):
                target_file_name = names[ind].replace("wav", "npy")
                np.save(os.path.join("EASE_embeddings", target_file_name), embedded[ind, :].cpu().detach().numpy())  

if __name__ == "__main__":
    train()
    get_embedding()
