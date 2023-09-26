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
import pickle5 as pickle
import ast
from pitch_attention_adv import create_dataset
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
        self.encoder = WAV2VECModel(self.wav2vec, hparams["output_classes"], hparams["emotion_embedding_dim"])
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

def get_f0():
    os.makedirs("pred_DSDT_f0", exist_ok=True)
    test_loader = create_dataset("test", 1)
    model = PitchModel(hparams)
    model = torch.load('f0_predictor.pth', map_location=device)
    model.to(device)
    model.eval()
    sources = ["0011_000021.wav", "0012_000022.wav", "0013_000025.wav",
               "0014_000032.wav", "0015_000034.wav", "0016_000035.wav",
               "0017_000038.wav", "0018_000043.wav", "0019_000023.wav",
               "0020_000047.wav"]

    with torch.no_grad():
        for source in sources:
            for i, data in enumerate(test_loader):
                inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                   torch.tensor(data["mask"]).to(device),\
                                                   torch.tensor(data["hubert"]).to(device),\
                                                   torch.tensor(data["f0"]).to(device),\
                                                   torch.tensor(data["labels"]).to(device)
                
                speaker = torch.tensor(data["speaker"]).to(device)
            
                names = data["names"] 
                if names[0] == source:
                    source_name = names[0].replace(".wav", "")
                    tokens_s, mask_s, speaker_s = tokens, mask, speaker

            for i, data in enumerate(test_loader):
                inputs, mask ,tokens, f0_trg, labels = torch.tensor(data["audio"]).to(device), \
                                                    torch.tensor(data["mask"]).to(device),\
                                                    torch.tensor(data["hubert"]).to(device),\
                                                    torch.tensor(data["f0"]).to(device),\
                                                    torch.tensor(data["labels"]).to(device)
                names = data["names"] 
                speaker = source[:5]
                if speaker not in names[0] and labels[0] > 0 and (int(names[0][5:11]) - int(source[5:11]))%350 != 0:
                    inputs_t = inputs
                    pitch_pred, _, _, _ = model(inputs_t, tokens_s, speaker_s, mask_s)
                    pitch_pred = torch.exp(pitch_pred) - 1
                    final_name = source_name + names[0]
                    final_name = final_name.replace(".wav", ".npy")
                    np.save(os.path.join("pred_DSDT_f0", final_name), pitch_pred[0, :].cpu().detach().numpy()) 

if __name__ == "__main__":
    get_f0()