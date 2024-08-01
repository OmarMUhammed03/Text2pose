import pandas as pd
import os
import torch
from datasets import Dataset
from huggingface_hub import login
from transformers import BertTokenizer, AutoTokenizer, AutoModelForMaskedLM, TrainingArguments
from transformers import AdamW, get_scheduler, DataCollatorWithPadding
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
# Login to Hugging Face
login(token="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")

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
# file_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/'

# Get datasets
# train_dataset, dev_dataset, test_dataset = get_datasets(file_path)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")

# Function to preprocess dataset
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

# # Preprocess datasets
# train_dataset = get_preprocessed_dataset(train_dataset)
# dev_dataset = get_preprocessed_dataset(dev_dataset)
# test_dataset = get_preprocessed_dataset(test_dataset)

# #save the datasets
# train_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/train_dataset/')
# dev_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/dev_dataset/')
# test_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/test_dataset/')

# load the datasets

dataset_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/'

train_dataset = Dataset.load_from_disk(dataset_path+'train_dataset/')
dev_dataset = Dataset.load_from_disk(dataset_path+'dev_dataset/')
test_dataset = Dataset.load_from_disk(dataset_path+'test_dataset/')


batch_sizes = [32, 48, 64, 48, 48]
all_num_epochs = [15, 15, 15, 20, 30]

for i in range(len(batch_sizes)):
    # Initialize the dataloaders
    batch_size = batch_sizes[i]

    num_epochs = all_num_epochs[i]

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=data_collator)

    # Initialize the model
    model = AutoModelForMaskedLM.from_pretrained("netscratch/abdelgawad/models/bart/model")
    model.config.pad_token_id = tokenizer.pad_token_id


    # model.save_pretrained('netscratch/abdelgawad/models/bart/model')


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    eval_prompt = "die halten sich morgen recht zäh"

    max_input = 512

    suffix = "_epochs_" + str(num_epochs) + "_batch_size_" + str(batch_size)
    directory_path = "/netscratch/abdelgawad/trained-models/bart/model" + suffix

    training_args = Seq2SeqTrainingArguments(
        output_dir= directory_path,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
    )

    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    # Train the model
    trainer.train()

    # Prepare inputs for prediction
    model_inputs = tokenizer(eval_prompt, max_length=max_input, padding='max_length', truncation=True, return_tensors="pt")
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    # Make prediction
    outputs = model.generate(**model_inputs)
    # Decode the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(directory_path)
    print(decoded_output)

    os.makedirs(directory_path, exist_ok=True)
    model.save_pretrained(directory_path)
    del model
