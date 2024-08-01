import torch
import numpy as np
import os
import pandas as pd
import nltk
from rouge_score import rouge_scorer
from datasets import Dataset
import evaluate
from huggingface_hub import login
# Download the necessary NLTK data files
nltk.download('punkt')
login(token="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")
from transformers import LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
        
def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for pred, ref in zip(predictions, references):
        # print(f"Prediction: {pred}")
        # print(f"Reference: {ref}")
        score = scorer.score(ref, pred)
        scores["rouge1"].append(score['rouge1'].fmeasure)
        scores["rouge2"].append(score['rouge2'].fmeasure)
        scores["rougeL"].append(score['rougeL'].fmeasure)

    avg_scores = {k: np.mean(v) for k, v in scores.items()}
    return avg_scores

def print_metrics(dataset, name, tokenizer, model):
    model.eval()
    #to compute metrics
    with torch.no_grad():
        references, predictions, sentences = [], [], []
        for sample in dataset:
            model_input = tokenizer(sample["text"], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(**model_input, max_length=8192, num_beams=4, no_repeat_ngram_size=2, early_stopping=True)
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                extracted_output = extract_output_from_translation(generated_text)
                print(f"Generated text: {generated_text}")
                print(f"sentence: {sample['text']}")
                print(f"Extracted output: {extracted_output}")
                print(f"Reference: {sample['gloss']}")
                references.append(sample["gloss"])
                predictions.append(extracted_output)
                sentences.append(sample["text"])

        metrics = compute_metrics(predictions, references)

        print(f"{name} metrics: {metrics}")

        rouge_score = compute_rouge(predictions, references)
        print(f"{name} rouge score: {rouge_score}")

def extract_output_from_translation(translation):
    # Extract the output from the translation
    output = translation.split('translation:\n')[1].strip()
    return output             

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
    decoded_preds = preds
    decoded_labels = labels

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = evaluate.load("bleu")
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

train_config = TRAIN_CONFIG()
train_config.model_name = "/netscratch/abdelgawad/models/meta-llama-3-8b/model"
train_config.num_epochs = 23
train_config.run_validation = False
train_config.gradient_accumulation_steps = 2
train_config.batch_size_training = 1
train_config.val_batch_size = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
# train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.num_workers_dataloader = 1
print(train_config)
config = BitsAndBytesConfig(load_in_8bit=True)

# model = LlamaForCausalLM.from_pretrained(
#     train_config.model_name,
#     device_map="auto",
#     quantization_config=config,
#     use_cache=False,
#     attn_implementation="sdpa" if train_config.use_fast_kernels else None,
#     torch_dtype=torch.float16,
# )


tokenizer = AutoTokenizer.from_pretrained("/netscratch/abdelgawad/models/meta-llama-3-8b/tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# Read CSV files
text_df_test = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/test.csv', header=None, names=['text'])
gloss_df_test = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/test_gloss.csv', header=None, names=['gloss'])
text_df_dev = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/dev.csv', header=None, names=['text'])
gloss_df_dev = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/dev_gloss.csv', header=None, names=['gloss'])
text_df_train = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/train.csv', header=None, names=['text'])
gloss_df_train = pd.read_csv('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/train_gloss.csv', header=None, names=['gloss'])

glosses_path = "/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/all_glosses.csv"
all_glosses= pd.read_csv(glosses_path, header=None, names=['gloss'])
all_glosses = all_glosses.dropna(subset=['gloss'])
glosses = all_glosses['gloss'].tolist()

# Drop any rows with missing values
text_df_test = text_df_test.dropna(subset=['text'])
gloss_df_test = gloss_df_test.dropna(subset=['gloss'])

text_df_dev = text_df_dev.dropna(subset=['text'])
gloss_df_dev = gloss_df_dev.dropna(subset=['gloss'])

text_df_train = text_df_train.dropna(subset=['text'])
gloss_df_train = gloss_df_train.dropna(subset=['gloss'])



texts_test = text_df_test['text'].tolist()
glosses_test = gloss_df_test['gloss'].tolist()

texts_dev = text_df_dev['text'].tolist()
glosses_dev = gloss_df_dev['gloss'].tolist()

texts_train = text_df_train['text'].tolist()
glosses_train = gloss_df_train['gloss'].tolist()

# assert len(text_df) == len(gloss_df), "The text and gloss files must have the same number of lines."

# print(f"Number of training samples: {len(texts_val)}")
#create a dict then a dataset from that dict
train_dict = {
    "text": texts_train,
    "gloss": glosses_train
}

dev_dict = {
    "text": texts_dev,
    "gloss": glosses_dev
}

test_dict = {
    "text": texts_test,
    "gloss": glosses_test
}

train_dataset = Dataset.from_dict(train_dict)
dev_dataset = Dataset.from_dict(dev_dict)
test_dataset = Dataset.from_dict(test_dict)


train_dataset = get_preprocessed_dataset(train_dataset)
dev_dataset = get_preprocessed_dataset(dev_dataset)
test_dataset = get_preprocessed_dataset(test_dataset)



# models file path
models_path = "/netscratch/abdelgawad/trained-models/meta-llama-3-8b/"
mylist = ['model']
for model_name in os.listdir(models_path):
    print(f"Model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(
        models_path + model_name,
        device_map="auto",
        quantization_config=config,
        use_cache=False,
        attn_implementation="sdpa" if train_config.use_fast_kernels else None,
        torch_dtype=torch.float16,
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id


    model.eval()
    # with torch.no_grad():
    #     print(tokenizer.decode(model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
    print_metrics(test_dataset, "test", tokenizer, model)
    print_metrics(dev_dataset, "dev", tokenizer, model)
    print('-------------------------------------')
    del model
