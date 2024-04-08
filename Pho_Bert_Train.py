# import library
from transformers import (AutoTokenizer,
                          RobertaForSequenceClassification,
                          Trainer,
                          TrainingArguments,)

from transformers import DataCollatorWithPadding
from datasets import load_dataset

model_id = 'vinai/phobert-base-v2'

# Initialising Tokeniser and Model
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          cache_dir='./cache',
                                          model_max_length=256)

model = RobertaForSequenceClassification.from_pretrained(model_id,
                                                           num_labels=3,
                                                           cache_dir='./cache')
# Full finetune or model head finetune
model.roberta.requires_grad_(False)

# Print the trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable Parameters: {trainable_params}')

# Define preprocess_function (tokenise)
def preprocess_function(examples):
    return tokenizer(examples['text'], max_length=256, padding='max_length', truncation=True)

# Define Data Collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, )

# Load data and apply tokeniser by mapping
ds = load_dataset('csv', data_files='/home4/bachpt/domain_classification/Data/Train/All_train_segmented.csv', split = 'train')
ds = ds.map(preprocess_function, batched=True)

# Map string labels to int ids
label2id = {"Chat": 0, "Uni": 1, "Others": 2}

# Map labels to numerical IDs
def map_labels(example):
    example["label"] = label2id[example["label"]]
    return example

ds = ds.map(map_labels)

# Define training arguments
training_args = TrainingArguments( output_dir='./saved_checkpoints',
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=8,
                                  num_train_epochs=10,
                                  weight_decay=0.01,
                                #   report_to='tensorboard',
                                  save_strategy='epoch',
                                  logging_strategy='steps',
                                  logging_steps=100,
                                  save_total_limit=2,

)

# Train
trainer = Trainer( model=model,
                  args=training_args,
                  train_dataset=ds,
                  tokenizer=tokenizer,
                  data_collator=data_collator
                  )

trainer.train()