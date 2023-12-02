import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AdamW
from datasets import load_dataset

# Load and preprocess the SQuAD dataset
dataset = load_dataset("squad")

model_name = "xyma/COSTA-wiki"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Preprocessing function for the dataset
def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",  # Truncate only the context part
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    # Extract start and end positions of answers
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        cls_index = tokenizer.cls_token_id
        sequence_ids = tokenized_examples.sequence_ids(i)

        # If no answers are provided, set the cls index as answer
        if len(answers[i]["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Extract start and end positions in the token space
            start_char = answers[i]["answer_start"][0]
            end_char = start_char + len(answers[i]["text"][0])
            token_start_index = 0
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            tokenized_examples["start_positions"].append(token_start_index)
            tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Load the COSTA model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create DataLoaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=16)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

for epoch in range(3):  # Example: 3 epochs
    model.train()
    for batch in train_loader:
        # Move batch data to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch} Loss {loss.item()}")

    # Validation Loop
    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            start_positions=start_positions, end_positions=end_positions)

        loss = outputs.loss
        total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss}")

# Save the model
model.save_pretrained("./costa_finetuned")
