import os
import torch
import torchaudio
from einops.layers.torch import Rearrange
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import logging
import numpy as np
import json
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import torch.nn as nn
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
import random
import torch.nn.functional as F
from config import hparams
from config import train_tokens_orig, val_tokens_orig, test_tokens_orig, f0_file
import pickle5 as pickle
import ast
import math
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

    def __init__(self, folder, token_file):
        self.folder = folder
        wav_files = os.listdir(folder)
        wav_files = [x for x in wav_files if ".wav" in x]
        self.wav_files = wav_files
        self.sr = 16000
        self.tokens = {}
        with open(f0_file, 'rb') as handle:
            self.f0_feat = pickle.load(handle)
        with open(token_file) as f:
            lines = f.readlines()
            for l in lines:
                d = ast.literal_eval(l)
                name, tokens = d["audio"], d["hubert"]
                tokens_l = tokens.split(" ")
                self.tokens[name.split(os.sep)[-1]] = np.array(tokens_l).astype(int)

    def __len__(self):
        return len(self.wav_files) 

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

    def getspkrlabel(self, file_name):
        spkr_name = file_name[:4]
        speaker_dict = {}
        for ind in range(11, 21):
            speaker_dict["00"+str(ind)] = ind-11
        speaker_feature = np.load(os.path.join("/folder/to/EASE/embeddings", file_name.replace(".wav", ".npy")))

        return speaker_feature, speaker_dict[spkr_name]
        
    def __getitem__(self, audio_ind): 
        class_id = self.getemolabel(self.wav_files[audio_ind])  
        audio_path = os.path.join(self.folder, self.wav_files[audio_ind])
        (sig, sr) = torchaudio.load(audio_path)
        
        sig = sig.numpy()[0, :]
        tokens = self.tokens[self.wav_files[audio_ind]]
        speaker_feat, speaker_label = self.getspkrlabel(self.wav_files[audio_ind])
        
        final_sig = sig
        f0 = self.f0_feat[self.wav_files[audio_ind]]

        return final_sig, f0, tokens, class_id, speaker_feat, speaker_label, self.wav_files[audio_ind]

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class WAV2VECModel(nn.Module):
    def __init__(self,
                 wav2vec,
                 output_dim,
                 hidden_dim_emo):
        
        super().__init__()
        
        self.wav2vec = wav2vec
        
        embedding_dim = wav2vec.config.to_dict()['hidden_size']
        self.out = nn.Linear(hidden_dim_emo, output_dim)
        self.out_spkr = nn.Linear(hidden_dim_emo, 10)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim_emo, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(in_channels=hidden_dim_emo, out_channels=hidden_dim_emo, kernel_size=5, padding=2)

        self.relu = nn.ReLU()
        
    def forward(self, aud, alpha):
        aud = aud.squeeze(0)
        hidden_all = list(self.wav2vec(aud).hidden_states)
        embedded = sum(hidden_all)
        embedded = embedded.permute(0, 2, 1)

        emo_embedded = self.relu(self.conv1(embedded))
        emo_embedded = self.relu(self.conv2(emo_embedded))
        emo_embedded = emo_embedded.permute(0, 2, 1)
        emo_hidden = torch.mean(emo_embedded, 1).squeeze(1)

        out_emo = self.out(emo_hidden)

        reverse_feature = ReverseLayerF.apply(embedded, alpha)

        embedded_spkr = self.relu(self.conv3(reverse_feature))
        embedded_spkr = self.relu(self.conv4(embedded_spkr))
        hidden_spkr = torch.mean(embedded_spkr, -1).squeeze(-1)
        output_spkr = self.out_spkr(hidden_spkr)
        
        return out_emo, output_spkr, emo_hidden, emo_embedded

class CrossAttentionModel(nn.Module):
    def __init__(self, hidden_dim_q, hidden_dim_k):
        super().__init__()
        HIDDEN_SIZE = 256
        NUM_ATTENTION_HEADS = 4
        self.inter_dim = HIDDEN_SIZE//NUM_ATTENTION_HEADS
        self.num_heads = NUM_ATTENTION_HEADS
        self.fc_q = nn.Linear(hidden_dim_q, self.inter_dim*self.num_heads)
        self.fc_k = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)
        self.fc_v = nn.Linear(hidden_dim_k, self.inter_dim*self.num_heads)

        self.multihead_attn = nn.MultiheadAttention(self.inter_dim*self.num_heads,
                                                    self.num_heads,
                                                    dropout = 0.5,
                                                    bias = True,
                                                    batch_first=True)
                                                                                                           
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim_q, eps = 1e-6)
        self.fc = nn.Linear(self.inter_dim*self.num_heads, hidden_dim_q)
        self.fc_1 = nn.Linear(hidden_dim_q, hidden_dim_q)
        self.relu = nn.ReLU()
    
    def forward(self, query_i, key_i, value_i):
        query = self.fc_q(query_i)
        key = self.fc_k(key_i)
        value = self.fc_v(value_i)
        cross, _ = self.multihead_attn(query, key, value, need_weights = True)
        skip = self.fc(cross)
 
        skip += query_i
        skip = self.relu(skip)
        skip = self.layer_norm(skip)

        new = self.fc_1(skip)
        new += skip
        new = self.relu(new)
        out = self.layer_norm_1(new)
        
        return out

class PitchModel(nn.Module):
    def __init__(self, hparams):
        super(PitchModel, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
        self.wav2vec = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h", output_hidden_states=True)
        self.encoder = WAV2VECModel(self.wav2vec, 5, hparams["emotion_embedding_dim"])
        self.embedding = nn.Embedding(101, 128, padding_idx=100)        
        self.fusion = CrossAttentionModel(128, 128)
        self.linear_layer = nn.Linear(128, 1)
        self.leaky = nn.LeakyReLU()
        self.cnn_reg1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.cnn_reg2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.speaker_linear = nn.Linear(128, 128)

    def forward(self, aud, tokens, speaker, lengths, alpha=1.0):
        hidden = self.embedding(tokens.int())
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        emo_out, spkr_out, _, emo_embedded = self.encoder(inputs['input_values'].to(device), alpha)
        speaker_temp = speaker.unsqueeze(1).repeat(1, emo_embedded.shape[1], 1)
        speaker_temp = self.speaker_linear(speaker_temp)
        emo_embedded = emo_embedded + speaker_temp
        pred_pitch = self.fusion(hidden, emo_embedded, emo_embedded)
        pred_pitch = pred_pitch.permute(0, 2, 1)
        pred_pitch = self.cnn_reg2(self.leaky(self.cnn_reg1(pred_pitch)))
        pred_pitch = pred_pitch.squeeze(1)
        mask = torch.arange(hidden.shape[1]).expand(hidden.shape[0], hidden.shape[1]).to(device) < lengths.unsqueeze(1)
        pred_pitch = pred_pitch.masked_fill(~mask, 0.0)
        mask = mask.int()

        return pred_pitch, emo_out, spkr_out, mask

def custom_collate(data):
    batch_len = len(data)
    new_data = {"audio":[], "mask":[], "labels":[], "hubert":[], "f0":[], "speaker":[], "speaker_label":[], "names":[]}
    max_len_f0, max_len_hubert, max_len_aud = 0, 0, 0
    for ind in range(len(data)):
        max_len_aud = max(data[ind][0].shape[-1], max_len_aud)
        max_len_f0 = max(data[ind][1].shape[-1], max_len_f0)
        max_len_hubert = max(data[ind][2].shape[-1], max_len_hubert)
    for i in range(len(data)):
        final_sig = np.concatenate((data[i][0], np.zeros((max_len_aud-data[i][0].shape[-1]))), -1)
        f0_feat = np.concatenate((data[i][1], np.zeros((max_len_f0-data[i][1].shape[-1]))), -1)
        mask = data[i][2].shape[-1]
        hubert_feat = np.concatenate((data[i][2], 100*np.ones((max_len_f0-data[i][2].shape[-1]))), -1)
        labels = data[i][3]
        speaker_feat = data[i][4]
        speaker_label = data[i][5]
        names = data[i][6]
        new_data["audio"].append(final_sig)
        new_data["f0"].append(f0_feat)
        new_data["mask"].append(torch.tensor(mask))
        new_data["hubert"].append(hubert_feat)
        new_data["labels"].append(torch.tensor(labels))
        new_data["speaker"].append(speaker_feat)
        new_data["speaker_label"].append(speaker_label)
        new_data["names"].append(names)
    new_data["audio"] = np.array(new_data["audio"])
    new_data["mask"] = np.array(new_data["mask"])
    new_data["hubert"] = np.array(new_data["hubert"])
    new_data["f0"] = np.array(new_data["f0"])
    new_data["labels"] = np.array(new_data["labels"])
    new_data["speaker_label"] = np.array(new_data["speaker_label"])
    new_data["speaker"] = np.array(new_data["speaker"])
    return new_data

def create_dataset(mode, bs=24):
    if mode == 'train':
        folder = "/folder/to/train/audio/files"
        token_file = train_tokens_orig["ESD"]
    elif mode == 'val':
        folder = "/folder/to/validation/audio/files"
        token_file = val_tokens_orig["ESD"]
    else:
        folder = "/folder/to/test/audio/files"
        token_file = test_tokens_orig["ESD"]
    dataset = MyDataset(folder, token_file)
    loader = DataLoader(dataset,
                    batch_size=bs,
                    pin_memory=False,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=custom_collate)
    return loader

def l2_loss(input, target):
    return F.l1_loss(
        input=input.float(),
        target=target.float(),
        reduction='none'
    )

def train():
    
    train_loader = create_dataset("train")
    val_loader = create_dataset("val")
    model = PitchModel(hparams)
    unfreeze = [i for i in range(0, 24)]
    for name, param in model.named_parameters():
        if 'wav2vec' in name:
            param.requires_grad = False
        for num in unfreeze:
            if str(num) in name and 'conv' not in name:
                param.requires_grad = True
    model.to(device)
    base_lr = 1e-4
    parameters = list(model.parameters()) 
    optimizer = Adam([{'params':parameters, 'lr':base_lr}])
    final_val_loss = 1e20
    for e in range(500):
        model.train()
        tot_loss, tot_correct = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0
        val_correct = 0.0
        pred_tr = []
        gt_tr = []
        pred_val = []
        gt_val = []
        tot_train_samples = 0
        tot_val_samples = 0
        for i, data in enumerate(tqdm(train_loader)):
            model.train()
            p = float(i + e * len(train_loader)) / 100 / len(train_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
            speaker_label = torch.tensor(data["speaker_label"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            pitch_pred, emo_out, spkr_out, mask_loss = model(inputs, tokens, speaker, mask, alpha)
            pitch_pred = torch.exp(pitch_pred) - 1
            loss1 = (mask_loss * nn.L1Loss(reduction='none')(pitch_pred, f0_trg.float().detach())).sum()
            loss2 = nn.CrossEntropyLoss(reduction='none')(emo_out, labels).sum()
            loss3 = nn.CrossEntropyLoss(reduction='none')(spkr_out, speaker_label).sum()
            cur_train_samples = (mask_loss != 0).sum()
            tot_train_samples += cur_train_samples
            loss = loss1 + 1000*loss2 + 1000*loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            f0_loss = loss1
            tot_loss += f0_loss.detach().item()
            pred = torch.argmax(emo_out, dim = 1)
            pred = pred.detach().cpu().numpy()
            pred = list(pred)
            pred_tr.extend(pred)
            labels = labels.detach().cpu().numpy()
            labels = list(labels)
            gt_tr.extend(labels)
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
                speaker = torch.tensor(data["speaker"]).to(device)
                pitch_pred, emo_out, spkr_out, mask_loss = model(inputs, tokens, speaker, mask)
                pitch_pred = torch.exp(pitch_pred) - 1
                loss1 = (mask_loss * nn.L1Loss(reduction='none')(pitch_pred, f0_trg.float().detach())).sum()
                loss2 = nn.CrossEntropyLoss(reduction='none')(emo_out, labels).sum()
                cur_val_samples = (mask_loss != 0).sum()
                tot_val_samples += cur_val_samples
                loss = loss1 + 1000*loss2
                f0_loss = loss1
                val_loss += f0_loss.detach().item()
                pred = torch.argmax(emo_out, dim = 1)
                pred = pred.detach().cpu().numpy()
                pred = list(pred)
                pred_val.extend(pred)
                labels = labels.detach().cpu().numpy()
                labels = list(labels)
                gt_val.extend(labels)
        if val_loss < final_val_loss:
            torch.save(model, 'f0_predictor.pth')
            final_val_loss = val_loss
        train_loss = tot_loss/tot_train_samples
        train_f1 = f1_score(gt_tr, pred_tr, average='weighted')
        val_loss_log = val_loss/tot_val_samples
        val_f1 = f1_score(gt_val, pred_val, average='weighted')
        e_log = e + 1
        logger.info(f"Epoch {e_log}, \
                    Training Loss {train_loss},\
                    Training Accuracy {train_f1}")
        logger.info(f"Epoch {e_log}, \
                    Validation Loss {val_loss_log},\
                    Validation Accuracy {val_f1}")


if __name__ == "__main__":
    train()

