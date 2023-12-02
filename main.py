from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


# def preprocess(examples):
#     inputs = tokenizer(
#         examples['question'],
#         examples['document'],
#         padding='max_length',
#         truncation=True,
#         return_tensors='pt'
#     )
#     targets = tokenizer(
#
#     )


# Load model
model_name = "xyma/COSTA-wiki"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

dataset = load_dataset("natural_questions")
dataset = dataset.shuffle(seed=37)

train_data = dataset["train"]
val_data = dataset["validation"]

print(train_data['long_answer_candidates'])

training_args = TrainingArguments(output_dir="./finetuned_model",
                                  num_train_epochs=3,
                                  per_device_train_batch_size=8,
                                  per_device_eval_batch_size=8,
                                  warmup_steps=500,
                                  weight_decay=0.01,
                                  logging_dir="./logs")

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_data,
                  eval_dataset=val_data)

trainer.train()

results = trainer.evaluate()

print(results)

