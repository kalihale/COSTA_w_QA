import argparse

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training import COSTAForQA


# Preprocessing function for tokenization
def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    offset_mapping = tokenized_examples.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = tokenized_examples.sequence_ids(i)

        context_start = 0
        while sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while sequence_ids[context_end] != 1:
            context_end -= 1

        if offsets[context_start][0] > end_char or offsets[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_token = context_start
            while start_token < len(offsets) and offsets[start_token][0] <= start_char:
                start_token += 1
            end_token = context_end
            while end_token >= context_start and offsets[end_token][1] >= end_char:
                end_token -= 1

            start_positions.append(start_token - 1)
            end_positions.append(end_token + 1)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples


# Define a custom collate function for the DataLoader
def collate_fn(batch):
    return {
        'input_ids': torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch]),
        'attention_mask': torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch]),
        'start_positions': torch.stack([torch.tensor(item['start_positions'], dtype=torch.long) for item in batch]),
        'end_positions': torch.stack([torch.tensor(item['end_positions'], dtype=torch.long) for item in batch]),
    }


def compute_f1(start_logits, end_logits, start_positions, end_positions):
    f1_sum = 0
    for i in range(len(start_logits)):
        if start_logits[i] == end_logits[i]:
            if start_positions[i] == end_positions[i]:
                f1 = 1
            else:
                f1 = 0
        elif start_logits[i] > end_positions[i] or start_positions[i] > end_logits[i]:
            f1 = 0
        else:
            if start_logits[i] < start_positions[i]:
                if end_logits[i] < end_positions[i]:
                    shared = end_logits[i] - start_positions[i]
                else:
                    shared = end_positions[i] - start_positions[i]
            else:
                if end_logits[i] < end_positions[i]:
                    shared = end_logits[i] - start_logits[i]
                else:
                    shared = end_positions[i] - start_logits[i]
            precision = shared / (end_logits[i] - start_logits[i])
            recall = shared / (end_positions[i] - start_positions[i])
            f1 = (2 * precision * recall) / (precision + recall)
        f1_sum += f1
    return f1_sum, len(start_logits)


def exact_match(start_logits, end_logits, start_positions, end_positions):
    score = 0
    for i in range(len(start_logits)):
        if start_logits[i] == start_positions[i] and end_logits[i] == end_positions[i]:
            score += 1
    return score, len(start_logits)


if __name__ == "__main__":
    # Get command-line arguments
    parser = argparse.ArgumentParser(description="Process an array with a flag.")
    parser.add_argument('-lr', '--learningrates', nargs='+', type=float,
                        help='Array of floats, the learning rates to be tested.')
    parser.add_argument('-e', '--epochs', type=int, help='A single int argument, the number of epochs.')
    args = parser.parse_args()
    lr = args.learningrates
    epochs = args.epochs

    # Load the dataset
    dataset = load_dataset("squad")

    # Load a plaintext version of the dataset
    dataset_v = DataLoader(dataset["validation"], batch_size=8)

    # Load the COSTA model
    model_name = "xyma/COSTA-wiki"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = COSTAForQA(model_name)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True,
                                     remove_columns=dataset["validation"].column_names)

    # Create DataLoaders with the custom collate function
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=collate_fn)

    # Repeat for all learning rates from the CL
    for rate in lr:
        # Repeat for the number of epochs
        for epoch in range(epochs):
            try:
                model.load_pretrained("./costa_finetuned_squad_2", epoch, rate)
                # Move model to GPU if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print(device)
                model.to(device)
                model.eval()
                total_correct = 0
                total_val = 0
                f1_total_sum = 0
                with torch.no_grad():
                    for b in range(len(val_loader)):
                        batch = next(iter(val_loader))
                        v_batch = next(iter(dataset_v))
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        start_positions = batch['start_positions'].to(device)
                        end_positions = batch['end_positions'].to(device)
                        start_logits, end_logits = model(input_ids, attention_mask)
                        _, predicted_start = torch.max(start_logits, 1)
                        _, predicted_end = torch.max(end_logits, 1)
                        correct, batch_size = exact_match(predicted_start, predicted_end, start_positions, end_positions)
                        total_correct += correct
                        f1_batch, batch_size = compute_f1(predicted_start, predicted_end, start_positions, end_positions)
                        f1_total_sum += f1_batch
                        total_val += batch_size
                print("model_lr", rate, "_epoch", epoch, " exact match: ", total_correct / total_val, " f1 score: ",
                      f1_total_sum / total_val)
            except:
                continue
