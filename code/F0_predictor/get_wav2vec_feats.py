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
from config import hparams, f0_stats
#from config import train_tokens, val_tokens, test_tokens, f0_file
from config import train_tokens_orig, val_tokens_orig, test_tokens_orig, f0_file
import pickle5 as pickle
import ast
import math
from torch.autograd import Function
from pitch_convert import crema_dataset

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
        # if self.training:  # This masks part of the sequence to avoid overfit and encourage disentanglement
        #     mask = torch.cuda.FloatTensor(query_i.shape[0], query_i.shape[1]).uniform_() > 0.6
        #     query_i[mask] = 0
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
        self.encoder = WAV2VECModel(self.wav2vec, 3, hparams["emotion_embedding_dim"])
        self.embedding = nn.Embedding(101, 128, padding_idx=100)        
        self.fusion = CrossAttentionModel(128, 128)
        self.linear_layer = nn.Linear(128, 1)
        self.leaky = nn.LeakyReLU()
        self.cnn_reg1 = nn.Conv1d(128, 128, kernel_size=(3,), padding=1)
        self.cnn_reg2 = nn.Conv1d(128, 1, kernel_size=(1,), padding=0)
        self.speaker_linear = nn.Linear(128, 128)
        # self.pe_spk = PositionalEncoding(192)

    def forward(self, aud, tokens=None, speaker=None, lengths=None, alpha=1.0):
        inputs = self.processor(aud, sampling_rate=16000, return_tensors="pt")
        _, _, emo_hidden, _ = self.encoder(inputs['input_values'].to(device), alpha)
        
        return emo_hidden

def train():
    wav2vec_feats_folder = "wav2vec_feats"
    os.makedirs(wav2vec_feats_folder, exist_ok=True)
    train_loader = create_dataset("train", 1)
    val_loader = create_dataset("val", 1)
    test_loader = create_dataset("test", 1)
    model = torch.load('f0_predictor.pth', map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm(train_loader)):
            inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
            speaker_label = torch.tensor(data["speaker_label"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            name = data["names"]
            embedded = model(inputs, tokens, speaker, mask)
            for ind in range(len(name)):
                target_file_name = name[ind].replace("wav", "npy")
                np.save(os.path.join(wav2vec_feats_folder, target_file_name), embedded[ind, :].cpu().detach().numpy()) 

        for i, data in enumerate(tqdm(val_loader)):
            inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
            speaker_label = torch.tensor(data["speaker_label"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            name = data["names"]
            embedded = model(inputs, tokens, speaker, mask)
            for ind in range(len(name)):
                target_file_name = name[ind].replace("wav", "npy")
                np.save(os.path.join(wav2vec_feats_folder, target_file_name), embedded[ind, :].cpu().detach().numpy()) 
        
        for i, data in enumerate(tqdm(test_loader)):
            inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
            speaker_label = torch.tensor(data["speaker_label"]).to(device)
            speaker = torch.tensor(data["speaker"]).to(device)
            name = data["names"]
            embedded = model(inputs, tokens, speaker, mask)
            for ind in range(len(name)):
                target_file_name = name[ind].replace("wav", "npy")
                np.save(os.path.join(wav2vec_feats_folder, target_file_name), embedded[ind, :].cpu().detach().numpy()) 

if __name__ == "__main__":
    crema()