import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import spacy
import pickle
import io
import tarfile
import glob
import re
import os
from collections import Counter
import tqdm
from torchtext.vocab import build_vocab_from_iterator
from spacy.cli import download

def get_seq_num_annotations(file_path):

    annotations = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip the first line (header)
            seq_num, text = line.split('_')  # can't remember the separators
            annotations[int(seq_num)] = text.strip()
    return annotations

# Ensure Spacy's German model is downloaded
spacy_de = spacy.load('de_core_news_sm')

# 1. Tokenization and Vocabulary Building
def tokenize(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


import re

from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import re

def build_vocab(texts, min_freq=2):
    """
    Build vocabulary from a list of text sequences.
    Args:
        texts: List of text sequences (sentences).
        min_freq: Minimum frequency of words to be included in the vocabulary.
    Returns:
        vocab: TorchText vocabulary object.
    """
    counter = Counter()
    
    # Preprocess each text
    texts = [sentence.lower() for sentence in texts]  # Lowercase all texts
    texts = [re.sub(r'[^\w\s]', '', sentence) for sentence in texts]  # Remove punctuation
    
    token_lists = []
    
    for text in texts:
        tokens = tokenize(text)  # Tokenize text using provided tokenizer
        
        # Ensure only string tokens
        to_add = [tok for tok in tokens if isinstance(tok, str)]
        
        # Remove tokens containing any digits
        to_add = [re.sub(r'\d+', '', tok) for tok in to_add]
        
        # print(f"Tokens to add: {to_add}")  # For debugging
        
        # Append to token lists for building vocab
        token_lists.append(to_add)
        
        # Update counter with tokens
        counter.update(to_add)
    
    # Filter out any empty strings that might result from token removal
    counter = {k: v for k, v in counter.items() if k}

    # Now pass the token lists (not the counter) to build_vocab_from_iterator
    vocab = build_vocab_from_iterator(token_lists, specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=min_freq)

    # Set the default index for out-of-vocabulary tokens
    vocab.set_default_index(vocab['<unk>'])
    
    return vocab


def get_seq_num_keypoints(tar):
    dict = {}
    count = 0
    for name, member in zip(tar.getnames(), tar.getmembers()):
        if "keypoints" in name:
            # count += 1
            # if count > 1000:
            #     break
            kps = tar.extractfile(member)
            kps = kps.read()
            kps = pickle.load(io.BytesIO(kps)) * 256
            seq = int(name.split('.')[0])
            if seq not in dict:
                dict[seq] = []
            dict[seq].append(kps)
    return dict


def get_seq_num_keypoints(tar1, tar2, tar3):
    dict = {}
    count = 0
    for name, member in zip(tar1.getnames(), tar1.getmembers()):
        if "keypoints" in name:
            # count += 1
            # if count > 1000:
            #     break
            kps = tar1.extractfile(member)
            kps = kps.read()
            kps = pickle.load(io.BytesIO(kps)) * 256
            # print(name)
            seq = int(name.split('/')[0].split('-')[-1])
            if seq not in dict:
                dict[seq] = []
            dict[seq].append(kps)
    count = 0
    for name, member in zip(tar2.getnames(), tar2.getmembers()):
        if "keypoints" in name:
            # count += 1
            # if count > 1000:
            #     break
            kps = tar2.extractfile(member)
            kps = kps.read()
            kps = pickle.load(io.BytesIO(kps)) * 256
            seq = int(name.split('/')[0].split('-')[-1])
            if seq not in dict:
                dict[seq] = []
            dict[seq].append(kps)
    count = 0
    for name, member in zip(tar3.getnames(), tar3.getmembers()):
        if "keypoints" in name:
            count += 1
            if count > 1000:
                break
            kps = tar3.extractfile(member)
            kps = kps.read()
            kps = pickle.load(io.BytesIO(kps)) * 256
            seq = int(name.split('/')[0].split('-')[-1])
            if seq not in dict:
                dict[seq] = []
            dict[seq].append(kps)
    return dict


sentence_to_seq = {}
tokenized_to_seq = {}

# 2. Dataset Preparation
class PoseDataset(Dataset):
    def __init__(self, data_dict, vocab):
        self.data_dict = data_dict
        self.vocab = vocab
        self.examples = self.load_data()

    def load_data(self):
        examples = []

        for text, frames in self.data_dict.items():
            keypoints_list = [torch.tensor(keypoints).float() for keypoints in frames]
            tokenized_text = self.preprocess_text(text)
            # print(f"tokenized_text: {tokenized_text} for text: {text}")
            temp = tokenized_text
            # padd the tokenized_text
            if tokenized_text.size(0) < 250:
                pad_size = 250 - tokenized_text.size(0)
                padding = torch.full((pad_size,), 1)
                tokenized_text = torch.cat([tokenized_text, padding], dim=0)
            else:
                tokenized_text = tokenized_text[:250]
            # print(f"tokenized_text: {tokenized_text} for text: {text}")
            # print(f"shape: {tokenized_text.shape}")
            tokenized_to_seq[tuple(tokenized_text.tolist())] = sentence_to_seq[text]
            examples.append((tokenized_text, torch.stack(keypoints_list)))
        return examples

    def preprocess_text(self, text):
        tokens = tokenize(text)
        # print(f"tokens: {tokens}")
        return torch.tensor([self.vocab['<sos>']] + [self.vocab[token] for token in tokens] + [self.vocab['<eos>']])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# 3. Custom DataLoader
def collate_fn(batch, max_seq_length=250):
    texts, keypoints = zip(*batch)
    
    # Pad or truncate texts to max_seq_length
    padded_texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=1)
    if padded_texts.size(1) > max_seq_length:
        padded_texts = padded_texts[:, :max_seq_length]
    else:
        pad_size = max_seq_length - padded_texts.size(1)
        padding = torch.full((padded_texts.size(0), pad_size), 1)  # Padding token index 1
        padded_texts = torch.cat([padded_texts, padding], dim=1)
    
    # Pad or truncate keypoints to max_seq_length
    padded_keypoints = nn.utils.rnn.pad_sequence(keypoints, batch_first=True, padding_value=0)
    if padded_keypoints.size(1) > max_seq_length:
        padded_keypoints = padded_keypoints[:, :max_seq_length]
    else:
        pad_size = max_seq_length - padded_keypoints.size(1)
        padding = torch.zeros(padded_keypoints.size(0), pad_size, padded_keypoints.size(2), padded_keypoints.size(3))
        padded_keypoints = torch.cat([padded_keypoints, padding], dim=1)
    
    return padded_texts, padded_keypoints

# 4. Model Definition
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PoseTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, num_keypoints=96):
        super(PoseTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 2048)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc = nn.Linear(d_model, num_keypoints * 2)
        self.num_keypoints = num_keypoints

    def forward(self, src, src_mask=None):
        batch_size, seq_length = src.size(0), src.size(1)
        
        # Embedding and positional encoding
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Fully connected layer
        output = self.fc(output)
        
        # Reshaping to the desired output shape
        output = output.view(batch_size, seq_length, self.num_keypoints, 2)
        
        return output

# 5. Custom Loss Function with Masking
def masked_mse_loss(pred, target, mask):
    """
    pred: [batch_size, seq_len, num_keypoints, 2]
    target: [batch_size, seq_len, num_keypoints, 2]
    mask: [batch_size, seq_len] with 1 for valid frames and 0 for padding
    """
    loss = (pred - target) ** 2  # MSE loss
    loss = loss.mean(dim=[2, 3])  # Average over keypoints and coordinates (x, y)
    return (loss * mask).sum() / mask.sum()  # Return mean loss over the sequence

# 6. Training Loop
def train(model, iterator, optimizer, criterion, clip, alpha):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        
        output = model(src)  # [batch_size, seq_len, num_keypoints, 2]

        # Create a mask for valid frames based on the source sequence
        mask = (src != 1).float().sum(dim=-1) > 0  # Assumes padding token has index 1

        output_seq_len = output.size(1)
        target_seq_len = trg.size(1)
        # print(f"sample {i}, output_seq_len: {output_seq_len}, target_seq_len: {target_seq_len}")

        # Pad the output if it's shorter
        if output_seq_len < target_seq_len:
            pad_size = target_seq_len - output_seq_len
            padding = torch.zeros(output.size(0), pad_size, output.size(2), output.size(3)).to(output.device)
            output = torch.cat([output, padding], dim=1)
        elif output_seq_len > target_seq_len:
            pad_size = output_seq_len - target_seq_len
            trg = torch.cat([trg, torch.zeros(trg.size(0), pad_size, trg.size(2), trg.size(3)).to(trg.device)], dim=1)
        
        # Mask for valid frames
        mask = mask.unsqueeze(1).repeat(1, output.size(1))
        mask[output_seq_len:-1] = 0

        # Calculate loss with the criterion
        loss = criterion(output, trg, mask)

        # Length penalty if output sequence is shorter than target sequence
        length_diff = (target_seq_len - output_seq_len) ** 2
        length_penalty = alpha * length_diff
        loss += length_penalty

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def save_output(output, seq):
    pass

# 7. Training and Evaluation Setup
train_annotations = get_seq_num_annotations('/netscratch/abdelgawad/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/train.corpus.csv')
dev_annotations = get_seq_num_annotations('/netscratch/abdelgawad/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/dev.corpus.csv')
test_annotations = get_seq_num_annotations('/netscratch/abdelgawad/datasets/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/test.corpus.csv')

for seq in train_annotations:
    sentence_to_seq[train_annotations[seq]] = seq
for seq in dev_annotations:
    sentence_to_seq[dev_annotations[seq]] = seq
for seq in test_annotations:
    sentence_to_seq[test_annotations[seq]] = seq

tar1_path = '/netscratch/abdelgawad/datasets/train_kps.tar.gz'
tar2_path = '/netscratch/abdelgawad/datasets/new_test.tar.gz'
tar3_path = '/netscratch/abdelgawad/datasets/dev_kps.tar.gz'
tar1 = tarfile.open(tar1_path)
tar2 = tarfile.open(tar2_path)
tar3 = tarfile.open(tar3_path)
dict = get_seq_num_keypoints(tar1, tar2, tar3)

# keypoints_path = '/netscratch/abdelgawad/datasets/org/phoenix_keypoints.tar.gz'
# dict = {}
# with tarfile.open(keypoints_path, 'r:gz') as tar:
#         try:    
#             for member in tqdm(tar.getmembers(), desc="Extracting keypoints"):
#                 if ".keypoint" in member.name:
#                     # print(member.name)
#                     seq_number = member.name.split('/')[-2].split('-')[-1]
#                     if seq_number not in dict:
#                         dict[seq_number] = []
#                     kps = tar.extractfile(member)
#                     kps = kps.read()
#                     kps = pickle.load(io.BytesIO(kps)) * 256
#                     dict[seq_number].append(kps)
#                     pass
#         except EOFError:
#             print("Encountered an incomplete file. Skipping...")

print("train annotations: ", len(train_annotations))
print("dev annotations: ", len(dev_annotations))
print("Number of sequences with keypoints: ", len(dict))
print("missing sequences: ", len(train_annotations) + len(dev_annotations) + len(test_annotations) - len(dict))
# print(dict.keys())

tar1.close()
tar2.close()
tar3.close()
# Process annotations and filter sequences without keypoints
seqs = list(train_annotations.keys())

for seq in seqs:
    if seq not in dict:
        del train_annotations[seq]

seqs = list(dev_annotations.keys())
for seq in seqs:
    if seq not in dict:
        del dev_annotations[seq]

seqs = list(test_annotations.keys())
for seq in seqs:
    if seq not in dict:
        del test_annotations[seq]
    


# Prepare the training and development data
data_dict_train = {seq: dict[seq] for seq in train_annotations}
data_dict_dev = {seq: dict[seq] for seq in dev_annotations}

data_dict_test = {seq: dict[seq] for seq in test_annotations}

train_data = {train_annotations[seq]: data_dict_train[seq] for seq in data_dict_train}
dev_data = {dev_annotations[seq]: data_dict_dev[seq] for seq in data_dict_dev}
test_data = {test_annotations[seq]: data_dict_test[seq] for seq in data_dict_test}

# #only keep the first 2000 train sequences
# train_data = {k: train_data[k] for k in list(train_data)[:2000]}

# Build vocabulary

TEXT_VOCAB = build_vocab(list(train_data.keys()) + list(dev_data.keys()) + list(test_data.keys()), min_freq=2)

# Create datasets and dataloaders
pose_dataset_train = PoseDataset(train_data, TEXT_VOCAB)
pose_dataset_dev = PoseDataset(dev_data, TEXT_VOCAB)
pose_dataset_test = PoseDataset(test_data, TEXT_VOCAB)

batch_size = 32

train_loader = DataLoader(pose_dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(pose_dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(pose_dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model, optimizer, and criterion
model = PoseTransformer(vocab_size=len(TEXT_VOCAB), d_model=512, nhead=8, num_layers=6, num_keypoints=96)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = masked_mse_loss

# Training the model
N_EPOCHS = 5000
CLIP = 1
alpha = 0.1 
# print(len(train_loader))

for epoch in range(N_EPOCHS):
    model.train()
    train_loss = train(model, train_loader, optimizer, criterion, CLIP, alpha)
    path = f'/netscratch/abdelgawad/trained-models/pose_transformer_epochs{epoch}_batch-size{batch_size}.pth'
    model.eval()
    kps_dict = {}
    with torch.no_grad():
        for i, (src, trg) in enumerate(test_loader):
            output = model(src)
            print(f"sample output frame:")
            for j in range(len(output[0][0])):
                print(output[0][0][j])
            for j in range(len(output)):
                # print(tokenized_to_seq[tuple(src[i].tolist())])
                kps_dict[tokenized_to_seq[tuple(src[j].tolist())]] = output[j].tolist()
    with torch.no_grad():
        for i, (src, trg) in enumerate(valid_loader):
            output = model(src)
            for j in range(len(output)):
                # print(tokenized_to_seq[tuple(src[i].tolist())])
                kps_dict[tokenized_to_seq[tuple(src[j].tolist())]] = output[j].tolist()

    #save dict
    path_to_save = f'/netscratch/abdelgawad/datasets/pose_transformer_keypoints_epoch{epoch}_batch_size{batch_size}.pickle'
    with open(path_to_save, 'wb') as f:
        pickle.dump(kps_dict, f)
    # Save the model
    # torch.save(model.state_dict(), path)
    # Save the optimizer
    # path = f'/netscratch/abdelgawad/trained-models/pose_transformer_optimizer_epochs{epoch}_batch-size{batch_size}.pth'
    # torch.save(optimizer.state_dict(), path)
    
    # load the model
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, path: {path}')

# Save the model


# epoch = 30
# batch_size = 32 

# model_path = f'/netscratch/abdelgawad/models/pose_transformer_epochs{epoch}_batch-size{batch_size}.pth'
        

# inference


print(f"Saved keypoints to {path_to_save}")
        # print(len(src))
        # print(len(output))
        # print(f"src: {src}")
        # print(f"src[0] shape: {src[0].shape}")
        # print(f"output shape: {output.shape}")
        # print(tokenized_to_seq[tuple(src[0].tolist())])
        # print(f"target shape: {trg.shape}")

