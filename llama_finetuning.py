import torch
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from llama_recipes.utils.train_utils import train, evaluation
from collections import Counter
import pandas as pd
from torch.nn.utils import clip_grad_norm_
from llama_recipes.utils.config_utils import get_dataloader_kwargs
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from llama_recipes.configs import lora_config as LORA_CONFIG
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import Dataset, DatasetDict, Features, Value
from llama_recipes.data.concatenator import ConcatDataset
import evaluate
from huggingface_hub import login


# Download the necessary NLTK data files
nltk.download('punkt')

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        print(f"Prediction: {pred}")
        print(f"Reference: {ref}")
        score = scorer.score(ref, pred)
        scores["rouge1"].append(score['rouge1'].fmeasure)
        scores["rouge2"].append(score['rouge2'].fmeasure)
        scores["rougeL"].append(score['rougeL'].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

def replace_negative_100(tensor_list, pad_token_id):
    """
    Replace all occurrences of -100 in the list of tensors with the pad_token_id.

    Args:
    tensor_list (list of torch.Tensor): List of tensors to be processed.
    pad_token_id (int): The pad token ID to replace -100 with.

    Returns:
    list of torch.Tensor: List of tensors with -100 replaced by pad_token_id.
    """
    return [torch.where(tensor == -100, torch.tensor(pad_token_id), tensor) for tensor in tensor_list]

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(preds, labels):
    preds = replace_negative_100(preds, tokenizer.pad_token_id)
    labels = replace_negative_100(labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)

    return result

def get_preprocessed_dataset(dataset):
    prompt = "translate the following german text to its corresponding german gloss:\n{text}.\n---\ntranslation:\n"
    
    def apply_prompt_template(sample):
        return {
            "text": prompt.format(text=sample["text"]),
            "gloss_translation": sample["gloss"]
        }

    dataset = dataset.map(apply_prompt_template)
    
    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["text"], add_special_tokens=False)
        gloss_translation = tokenizer.encode(sample["gloss_translation"] + tokenizer.eos_token, add_special_tokens=False)
        
        sample = {
            "input_ids": prompt + gloss_translation,
            "attention_mask": [1] * (len(prompt) + len(gloss_translation)),
            "labels": prompt + gloss_translation
        }
        return sample

    dataset = dataset.map(tokenize_add_label)
    return dataset

# Main code
login(token="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")
batch_size = 16
train_config = TRAIN_CONFIG()
train_config.model_name = "/netscratch/abdelgawad/models/meta-llama-3-8b/model"
train_config.num_epochs = 30
train_config.run_validation = False
train_config.gradient_accumulation_steps = 2
train_config.batch_size_training = batch_size
train_config.val_batch_size = batch_size
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048  # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.num_workers_dataloader = 6
print(train_config)

config = BitsAndBytesConfig(load_in_8bit=True)

model = LlamaForCausalLM.from_pretrained(
    train_config.model_name,
    device_map="auto",
    quantization_config=config,
    use_cache=False,
    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
    torch_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained("/netscratch/abdelgawad/models/meta-llama-3-8b/tokenizer")
tokenizer.pad_token = tokenizer.eos_token
model.generation_config.pad_token_id = tokenizer.eos_token_id

eval_prompt = """
translate the following german text to its corresponding german gloss:
die halten sich morgen recht zäh
---
translation:
"""

model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

model.eval()
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

# Read CSV files
file_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/'
# Load data
# text_df_test = pd.read_csv(file_path + 'test.csv', header=None, names=['text'])
# gloss_df_test = pd.read_csv(file_path + 'test_gloss.csv', header=None, names=['gloss'])
# text_df_dev = pd.read_csv(file_path + 'dev.csv', header=None, names=['text'])
# gloss_df_dev = pd.read_csv(file_path + 'dev_gloss.csv', header=None, names=['gloss'])
# text_df_train = pd.read_csv(file_path + 'train.csv', header=None, names=['text'])
# gloss_df_train = pd.read_csv(file_path + 'train_gloss.csv', header=None, names=['gloss'])

# # Combine text and gloss dataframes
# test_data = pd.concat([text_df_test, gloss_df_test], axis=1)
# dev_data = pd.concat([text_df_dev, gloss_df_dev], axis=1)
# train_data = pd.concat([text_df_train, gloss_df_train], axis=1)

# # Define features
# features = Features({
#     'text': Value('string'),
#     'gloss': Value('string')
# })

# # Convert to Hugging Face Dataset
# test_dataset = Dataset.from_pandas(test_data, features=features)
# dev_dataset = Dataset.from_pandas(dev_data, features=features)
# train_dataset = Dataset.from_pandas(train_data, features=features)

# train_dataset = get_preprocessed_dataset(train_dataset)
# dev_dataset = get_preprocessed_dataset(dev_dataset)
# test_dataset = get_preprocessed_dataset(test_dataset)


# # Create a DatasetDict
# dataset_dict = DatasetDict({
#     'train': train_dataset,
#     'dev': dev_dataset,
#     'test': test_dataset
# })

# # Save the dataset (optional)
# dataset_dict.save_to_disk(file_path + 'huggingface_dataset_llama')


# Load the dataset
dataset_dict = DatasetDict.load_from_disk(file_path + 'huggingface_dataset_llama')
# text_df_test = pd.read_csv(file_path + 'test.csv', header=None, names=['text'])
# gloss_df_test = pd.read_csv(file_path + 'test_gloss.csv', header=None, names=['gloss'])
# text_df_dev = pd.read_csv(file_path + 'dev.csv', header=None, names=['text'])
# gloss_df_dev = pd.read_csv(file_path + 'dev_gloss.csv', header=None, names=['gloss'])
# text_df_train = pd.read_csv(file_path + 'train.csv', header=None, names=['text'])
# gloss_df_train = pd.read_csv(file_path + 'train_gloss.csv', header=None, names=['gloss'])

# # Drop any rows with missing values
# text_df_test = text_df_test.dropna(subset=['text'])
# gloss_df_test = gloss_df_test.dropna(subset=['gloss'])
# text_df_dev = text_df_dev.dropna(subset=['text'])
# gloss_df_dev = gloss_df_dev.dropna(subset=['gloss'])
# text_df_train = text_df_train.dropna(subset=['text'])
# gloss_df_train = gloss_df_train.dropna(subset=['gloss'])

# texts_test = text_df_test['text'].tolist()
# glosses_test = gloss_df_test['gloss'].tolist()
# texts_dev = text_df_dev['text'].tolist()
# glosses_dev = gloss_df_dev['gloss'].tolist()
# texts_train = text_df_train['text'].tolist()
# glosses_train = gloss_df_train['gloss'].tolist()

# # Create a dict and then a dataset from that dict
# train_dict = {"text": texts_train, "gloss": glosses_train}
# dev_dict = {"text": texts_dev, "gloss": glosses_dev}
# test_dict = {"text": texts_test, "gloss": glosses_test}

# train_dataset = Dataset.from_dict(train_dict)
# dev_dataset = Dataset.from_dict(dev_dict)
# test_dataset = Dataset.from_dict(test_dict)

# train_dataset = get_preprocessed_dataset(train_dataset)
# dev_dataset = get_preprocessed_dataset(dev_dataset)
# test_dataset = get_preprocessed_dataset(test_dataset)

#save in a json file
# train_dataset.save_to_disk("/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/train_dataset")
# dev_dataset.save_to_disk("/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/dev_dataset")
# test_dataset.save_to_disk("/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/test_dataset")

train_dataset = dataset_dict['train']
dev_dataset = dataset_dict['dev']
test_dataset = dataset_dict['test']

# Get dataloader kwargs
train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
dev_dl_kwargs = get_dataloader_kwargs(train_config, dev_dataset, tokenizer, "dev")
test_dl_kwargs = get_dataloader_kwargs(train_config, test_dataset, tokenizer, "test")

if train_config.batching_strategy == "packing":
    train_dataset = ConcatDataset(train_dataset, chunk_size=train_config.context_length)
    dev_dataset = ConcatDataset(dev_dataset, chunk_size=train_config.context_length)
    test_dataset = ConcatDataset(test_dataset, chunk_size=train_config.context_length)

# Create DataLoaders for the training and validation datasets
train_dataloader = DataLoader(train_dataset, num_workers=train_config.num_workers_dataloader, pin_memory=True, **train_dl_kwargs)
dev_dataloader = DataLoader(dev_dataset, num_workers=train_config.num_workers_dataloader, pin_memory=True, **dev_dl_kwargs)
test_dataloader = DataLoader(test_dataset, num_workers=train_config.num_workers_dataloader, pin_memory=True, **test_dl_kwargs)

lora_config = LORA_CONFIG()
lora_config.r = 64
lora_config.lora_alpha = lora_config.r * 2
lora_config.lora_dropout =  0.05

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Training loop
model.train()

optimizer = optim.AdamW(
    model.parameters(),
    lr=train_config.lr,
    weight_decay=train_config.weight_decay,
)
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

results = train(
    model,
    train_dataloader,
    dev_dataset,
    tokenizer,
    optimizer,
    scheduler,
    train_config.gradient_accumulation_steps,
    train_config,
    None,
    None,
    None,
    wandb_run=None,
)

print(results)

# Evaluation phase

eval_results = evaluation(
    model,
    train_config,
    test_dataloader,
    None,
    tokenizer,
    None,
)

model.eval()
for sample in test_dataset:
    print(sample)
with torch.no_grad():
    print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))

suffix = 'num_epochs_' + str(train_config.num_epochs) + '_batch_size_' + str(batch_size) + 'lora_r_' + str(lora_config.r) + '_lora_alpha_' + str(lora_config.lora_alpha) + '_lora_dropout_' + str(lora_config.lora_dropout)

model.save_pretrained("/netscratch/abdelgawad/trained-models/meta-llama-3-8b/model"+suffix)
