import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset, DatasetDict
from rouge_score import rouge_scorer
from huggingface_hub import login
login(token="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")
from transformers import BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments 
from torch.optim import AdamW
from transformers import get_scheduler, default_data_collator
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import evaluate
from transformers import LongformerTokenizer, LongformerModel
from transformers import LongformerForMaskedLM

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):

        score = scorer.score(ref, pred)
        scores["rouge1"].append(score['rouge1'].fmeasure)
        scores["rouge2"].append(score['rouge2'].fmeasure)
        scores["rougeL"].append(score['rougeL'].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

def extract_output_from_translation(translation):
    # Extract the output from the translation
    output = translation.split('translation:\n')[1].strip()
    return output        


def print_metrics(dataset, name, tokenizer, model):
    model.eval()
    #to compute metrics
    max_input = 512 
    with torch.no_grad():
        references, predictions, sentences = [], [], []
        for sample in dataset:
            model_inputs = tokenizer(sample['text'], max_length=max_input, padding='max_length', truncation=True, return_tensors="pt")
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            
            with torch.no_grad():
                # Make prediction
                outputs = model.generate(**model_inputs)
                # Decode the output
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"sentence: {sample['text']}")
                print(f"Reference: {sample['gloss']}")
                print(f"Generated text: {decoded_output}")
                references.append(sample["gloss"])
                predictions.append(decoded_output)
                sentences.append(sample["text"])

        metrics = compute_metrics(predictions, references)

        print(f"{name} metrics: {metrics}")

        rouge_score = compute_rouge(predictions, references)
        print(f"{name} rouge score: {rouge_score}")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(preds, labels):
    decoded_preds = preds
    decoded_labels = labels

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

def extract_output_from_translation(translation):
    # Extract the output from the translation
    output = translation.split('translation:')[1].strip()
    return output   

def get_preprocessed_dataset(dataset):
    
    def preprocess_data(data_to_process):
        max_input = 512
        max_target = 512

        inputs = data_to_process['text']

        model_inputs = tokenizer(inputs,  max_length=max_input, padding='max_length', truncation=True)

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(data_to_process['gloss'], max_length=max_target, padding='max_length', truncation=True)
            
        #set labels
        model_inputs['labels'] = targets['input_ids']
        #return the tokenized data
        #input_ids, attention_mask and labels
        return model_inputs


    dataset = dataset.map(preprocess_data)

    return dataset


def get_datasets(file_path):
    # Load data
    text_df_test = pd.read_csv(file_path + 'test.csv', header=None, names=['text'])
    gloss_df_test = pd.read_csv(file_path + 'test_gloss.csv', header=None, names=['gloss'])
    text_df_dev = pd.read_csv(file_path + 'dev.csv', header=None, names=['text'])
    gloss_df_dev = pd.read_csv(file_path + 'dev_gloss.csv', header=None, names=['gloss'])
    text_df_train = pd.read_csv(file_path + 'train.csv', header=None, names=['text'])
    gloss_df_train = pd.read_csv(file_path + 'train_gloss.csv', header=None, names=['gloss'])

    # Combine text and gloss dataframes
    test_data = pd.concat([text_df_test, gloss_df_test], axis=1)
    dev_data = pd.concat([text_df_dev, gloss_df_dev], axis=1)
    train_data = pd.concat([text_df_train, gloss_df_train], axis=1)

    # Convert to Hugging Face Dataset
    test_dataset = Dataset.from_pandas(test_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    train_dataset = Dataset.from_pandas(train_data)

    return train_dataset, dev_dataset, test_dataset

# File paths
file_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/'

# Load data
train_dataset, dev_dataset, test_dataset = get_datasets(file_path)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
print(f"tokenizer: {tokenizer}")

train_dataset = get_preprocessed_dataset(train_dataset)
test_dataset = get_preprocessed_dataset(test_dataset)
dev_dataset = get_preprocessed_dataset(dev_dataset)

# models file path
models_path = "/netscratch/abdelgawad/trained-models/bart/"
print(f"list of models: {os.listdir(models_path)}")
done = ['model_epochs_20_batch_size_8', 'model_epochs_30_batch_size_16', 'model_epochs_30_batch_size_32', 'model_epochs_50_batch_size_32', 'model_epochs_20_batch_size_16', 'model_epochs_50_batch_size_64', 'model_epochs_20_batch_size_32', 'model_epochs_30_batch_size_64', 'model_epochs_20_batch_size_64']
for model_name in os.listdir(models_path):
    if model_name == "model":
        continue
    if model_name in done:
        continue
    model = AutoModelForMaskedLM.from_pretrained(
        models_path + model_name
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(f"Model: {model_name}")

    model.eval()
    # with torch.no_grad():
    #     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
    print_metrics(test_dataset, "test", tokenizer, model)
    print_metrics(dev_dataset, "dev", tokenizer, model)
    print('-------------------------------------')
    del model
