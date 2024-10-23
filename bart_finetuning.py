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
import torch.nn as nn
from transformers import AutoTokenizer, BartForConditionalGeneration, PreTrainedTokenizerFast
import torch.nn as nn
# Login to Hugging Face
login(token="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")

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from huggingface_hub import login
from transformers import AutoTokenizer, BartForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorWithPadding
from torch.utils.data import DataLoader

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

# Get datasets
train_dataset, dev_dataset, test_dataset = get_datasets(file_path)

print(f"len(train_dataset): {len(train_dataset)}")
print(f"len(dev_dataset): {len(dev_dataset)}")
print(f"len(test_dataset): {len(test_dataset)}")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

# Function to preprocess dataset
def get_preprocessed_dataset(dataset):
    def preprocess_data(data_to_process):
        max_input = 1024
        max_target = 1024

        inputs = data_to_process['text']
        model_inputs = tokenizer(inputs, max_length=max_input, padding='max_length', truncation=True)

        with tokenizer.as_target_tokenizer():
            targets = tokenizer(data_to_process['gloss'], max_length=max_target, padding='max_length', truncation=True)
        target_ids = targets['input_ids']
        # Map the target tokens to the new vocabulary indices
        # target_ids = [token_to_idx.get(token, tokenizer.unk_token_id) for token in targets['input_ids']]
        # print(f"target_ids: {target_ids}")
        model_inputs['labels'] = target_ids
        return model_inputs

    processed_dataset = dataset.map(preprocess_data)
    print(f"Processed dataset size: {len(processed_dataset)}")
    # print(processed_dataset[0])  # Print the first item to inspect
    return processed_dataset

# Load the BART tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")


print(f"tokenizer: {tokenizer}")

# Preprocess datasets
train_dataset = get_preprocessed_dataset(train_dataset)
dev_dataset = get_preprocessed_dataset(dev_dataset)
test_dataset = get_preprocessed_dataset(test_dataset)

# #save the datasets
# train_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/train_dataset/')
# dev_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/dev_dataset/')
# test_dataset.save_to_disk('/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/test_dataset/')

# load the datasets

# dataset_path = '/netscratch/abdelgawad/datasets/phoenix_data/text_to_pose/bart/'

# train_dataset = Dataset.load_from_disk(dataset_path+'train_dataset/')
# dev_dataset = Dataset.load_from_disk(dataset_path+'dev_dataset/')
# test_dataset = Dataset.load_from_disk(dataset_path+'test_dataset/')



batch_sizes = [64, 32]
all_num_epochs = [20, 30]

for j in range(len(all_num_epochs)):
    for i in range(len(batch_sizes)):
        # Initialize the dataloaders
        batch_size = batch_sizes[i]
        num_epochs = all_num_epochs[j]
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=data_collator)

        # Initialize the model
        model = AutoModelForMaskedLM.from_pretrained("netscratch/abdelgawad/models/bart/model")
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Model: {model}")
        # Set the model to the appropriate device (GPU or CPU)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        eval_prompt = "die halten sich morgen recht zäh"
        max_input = 1024

        suffix = "_epochs_" + str(num_epochs) + "_batch_size_" + str(batch_size) + "_model"
        directory_path = "/netscratch/abdelgawad/m-trained-models/bart/model" + suffix
        print(torch.cuda.device_count())
        # Modify the Seq2SeqTrainingArguments to include distributed training settings
        training_args = Seq2SeqTrainingArguments(
            output_dir=directory_path,
            evaluation_strategy='epoch',
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            fp16=True,  # Enable mixed precision training
            gradient_accumulation_steps=1,
            num_train_epochs=num_epochs,
            dataloader_num_workers=4,
            # These flags enable distributed training
            deepspeed=None,
            local_rank=-1,  # Managed by the training launcher
            ddp_find_unused_parameters=False,  # Set this if facing any issues with unused parameters
            report_to="none",  # Disable logging, if not needed
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
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

        # Save the model and clean up
        os.makedirs(directory_path, exist_ok=True)
        model.save_pretrained(directory_path)
        del model
