import pandas as pd
import numpy as np
import os
import torch
from datasets import Dataset, DatasetDict
from rouge_score import rouge_scorer
from huggingface_hub import login
from transformers import BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from torch.optim import AdamW
from transformers import get_scheduler, default_data_collator
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import evaluate
from transformers import LongformerTokenizer, LongformerModel
from transformers import LongformerForMaskedLM
from scipy.spatial.distance import cosine
from helpers import get_all_datasetwords
from nltk.translate.meteor_score import single_meteor_score
from nltk import download, word_tokenize
from bleu import raw_corpus_bleu
# Login to Hugging Face
login(token="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")

# Functions



# Ensure necessary NLTK resources are downloaded
download('wordnet')
download('punkt')  # Required for word_tokenize


def calculate_meteor(candidates, references):
    """
    Calculate the average METEOR score for a list of candidate sentences against a list of reference sentences.
    
    :param candidates: A list of strings representing candidate sentences.
    :param references: A list of strings representing reference sentences.
    :return: The average METEOR score.
    """
    assert len(candidates) == len(references), "The number of candidates and references must be the same."
    
    scores = []
    for candidate, reference in zip(candidates, references):
        # Tokenize the sentences
        candidate_tokens = word_tokenize(candidate)
        reference_tokens = word_tokenize(reference)
        
        # Calculate the METEOR score using tokenized sentences
        score = single_meteor_score(reference_tokens, candidate_tokens)
        scores.append(score)
    
    return sum(scores) / len(scores)


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

def removeIX(s):
    return s.replace('IX', '')

def get_word_embedding(word, model, tokenizer, device):
    input_ids = tokenizer(word, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        outputs = model.model.encoder(input_ids)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def get_dataset_word_embeddings(dataset_words, model, tokenizer, device):
    dataset_word_embeddings = []
    for word in dataset_words:
        embedding = get_word_embedding(word, model, tokenizer, device)
        dataset_word_embeddings.append(embedding)
    return dataset_word_embeddings

def find_closest_word(embedding, dataset_word_embeddings, dataset_words):
    # Compute dot product between the embedding and all dataset word embeddings
    dot_products = [np.dot(embedding, dw_emb) for dw_emb in dataset_word_embeddings]
    # Find the index of the word with the highest dot product
    closest_word_idx = np.argmax(dot_products)
    result = dataset_words[closest_word_idx].split()
    return result[0]

def find_closest_word_cosine(embedding, dataset_word_embeddings, dataset_words):
    similarities = [1 - cosine(embedding, dw_emb) for dw_emb in dataset_word_embeddings]
    closest_word_idx = np.argmax(similarities)
    return dataset_words[closest_word_idx]

def fix_output_glosses(glosses, model, tokenizer, dataset_word_embeddings, dataset_words, device):
    glosses_list = glosses.split()
    result = ''
    for i in range(len(glosses_list)):
        if glosses_list[i] in dataset_words:
            result += glosses_list[i]
            if i < len(glosses_list) - 1:
                result += ' '
            continue
        if glosses_list[i] == 'NULL':
            print(glosses_list[i])
            print('failed')
            exit()
        cur_gloss_embedding = get_word_embedding(glosses_list[i], model, tokenizer, device)
        closest_word = find_closest_word(cur_gloss_embedding, dataset_word_embeddings, dataset_words)
        result += closest_word
        if i < len(glosses_list) - 1:
            result += ' '
    return result

def count_non_existing_words(glosses, dataset_words):
    glosses_list = glosses.split()
    count = 0
    for gloss in glosses_list:
        if gloss not in dataset_words:
            count += 1
    return count

def print_metrics(dataset, name, tokenizer, model, dataset_word_embeddings, all_glosses, device):
    model.eval()
    max_input = 1024
    max_output_length = 1024
    total_count_bad = 0
    total = 0
    with torch.no_grad():
        references, predictions, sentences = [], [], []
        for sample in dataset:
            model_inputs = tokenizer(sample['text'], max_length=max_input, padding='max_length', return_tensors="pt")
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            
            outputs = model.generate(**model_inputs, max_length=max_output_length)
            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Sentence: {sample['text']}")
            print(f"Reference: {sample['gloss']}")
            print(f"Generated text: {decoded_output}")
            total_count_bad += count_non_existing_words(decoded_output, all_glosses)
            total += (len(decoded_output.split()))
            decoded_output = fix_output_glosses(decoded_output, model, tokenizer, dataset_word_embeddings, all_glosses, device)
            print(f"Fixed generated text: {decoded_output}")
            references.append(sample["gloss"])
            predictions.append(decoded_output)
            sentences.append(sample["text"])
        # print("DONE")
        # metrics = compute_metrics(predictions, references, True)
        # print(f"{name} metrics: {metrics}")
        # metrics = compute_metrics(predictions, references, False)
        # print(f"{name} metrics without effective order: {metrics}")
        results = raw_corpus_bleu(predictions, references)
        print(f"{name} bleu score: {results.scores}")
        rouge_score = compute_rouge(predictions, references)
        print(f"{name} rouge score: {rouge_score}")

        
        #sort for bleu1 scores and print the worst 10
    
        # for i in range(4):
            # bleu_i_scores = []
            # for pred, ref, sentence in zip(predictions, references, sentences):
            #     cur_metric = compute_single(pred, ref)
            #     # add tuple of (bleu_i, pred, ref) to the list
            #     bleu_i_scores.append((cur_metric['precisions'][i], pred, ref, sentence))

            # # sort the list based on the bleu_i score
            # bleu_i_scores.sort(key=lambda x: x[0])
            # print(f"worst 10 bleu{i+1} scores:")

            # for j in range(10):
            #     print(f"bleu{i+1}: {bleu_i_scores[j][0]}")
            #     print(f"sentence: {bleu_i_scores[j][3]}")
            #     print(f"prediction: {bleu_i_scores[j][1]}")
            #     print(f"reference: {bleu_i_scores[j][2]}")
            #     print("")
        


        # print(f"Total count of bad words: {total_count_bad / total}")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(preds, labels, effective_order):
    decoded_preds = preds
    decoded_labels = labels
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    metric = evaluate.load("sacrebleu")
    result = metric._compute(predictions=decoded_preds, references=decoded_labels, smooth_method="floor",
        smooth_value=0,
        force=True,
        tokenize="none",
        use_effective_order=effective_order)
    return result

def compute_single(preds, labels):
    decoded_preds = [preds]
    decoded_labels = [labels]
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    metric = evaluate.load("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return result

def get_preprocessed_dataset(dataset):
    def preprocess_data(data_to_process):
        max_input = 1024
        max_target = 1024
        inputs = data_to_process['text']
        model_inputs = tokenizer(inputs, max_length=max_input, padding='max_length')
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(data_to_process['gloss'], max_length=max_target, padding='max_length')
        model_inputs['labels'] = targets['input_ids']
        return model_inputs
    dataset = dataset.map(preprocess_data)
    return dataset

def get_all_glosses(file_path):
    all_lines = open(file_path).readlines()
    glosses = []
    for line in all_lines:
        glosses.append(line.strip())
    print(glosses)
    return glosses

def get_datasets(file_path):
    text_df_test = pd.read_csv(file_path + 'test.csv', header=None, names=['text'])
    gloss_df_test = pd.read_csv(file_path + 'test_gloss.csv', header=None, names=['gloss'])
    text_df_dev = pd.read_csv(file_path + 'dev.csv', header=None, names=['text'])
    gloss_df_dev = pd.read_csv(file_path + 'dev_gloss.csv', header=None, names=['gloss'])
    text_df_train = pd.read_csv(file_path + 'train.csv', header=None, names=['text'])
    gloss_df_train = pd.read_csv(file_path + 'train_gloss.csv', header=None, names=['gloss'])
    test_data = pd.concat([text_df_test, gloss_df_test], axis=1)
    dev_data = pd.concat([text_df_dev, gloss_df_dev], axis=1)
    train_data = pd.concat([text_df_train, gloss_df_train], axis=1)
    test_dataset = Dataset.from_pandas(test_data)
    dev_dataset = Dataset.from_pandas(dev_data)
    train_dataset = Dataset.from_pandas(train_data)
    return train_dataset, dev_dataset, test_dataset

# File paths
file_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/'

# Load data
train_dataset, dev_dataset, test_dataset = get_datasets(file_path)

# Assume dataset_words is a list of words
dataset_words = get_all_datasetwords()

# Load the BART tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

all_glosses = get_all_glosses(file_path + 'all_glosses.csv')


test_dataset = get_preprocessed_dataset(test_dataset)
dev_dataset = get_preprocessed_dataset(dev_dataset)


# Models file path
models_path = "/netscratch/abdelgawad/trained-models/bart/"
print(f"List of models: {os.listdir(models_path)}")


for model_name in os.listdir(models_path):

    model = AutoModelForMaskedLM.from_pretrained(models_path + model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    dataset_word_embeddings = get_dataset_word_embeddings(all_glosses, model, tokenizer, device)

    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"Model: {model_name}")
    model.eval()
    
    print_metrics(test_dataset, "test", tokenizer, model, dataset_word_embeddings, all_glosses , device)
    print_metrics(dev_dataset, "dev", tokenizer, model, dataset_word_embeddings, all_glosses, device)
    
    print('-------------------------------------')
    del model
