import collections

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from training import COSTAForQA
import evaluate


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


# def predict_answers_and_evaluate(start_logits, end_logits, eval_set, examples):
#     examples_to_features = collections.defaultdict(list)
#     for idx, feature in enumerate(eval_set):
#         examples_to_features[feature["id"]].append(idx)
#
#     n_best =
#     _, predicted_start = torch.max(start_logits, 1)
#     _, predicted_end = torch.max(end_logits, 1)
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens())


def exact_match(start_logits, end_logits, start_positions, end_positions):
    score = 0
    for i in range(len(start_logits)):
        if start_logits[i] == start_positions[i] and end_logits[i] == end_positions[i]:
            score += 1
    return score, len(start_logits)



if __name__ == "__main__":
    dataset = load_dataset("squad")

    dataset_v = DataLoader(dataset["validation"], batch_size=8)

    model_name = "xyma/COSTA-wiki"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = COSTAForQA(model_name)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["validation"].column_names)

    # Create DataLoaders with the custom collate function
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=collate_fn)

    for models in range(2):
        for epoch in range(1, 4):  # Adjust the number of epochs as needed
            model.load_pretrained("./costa_finetuned_squad", epoch, models)
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(device)
            model.to(device)
            model.eval()
            total_correct = 0
            total_val = 0
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
                    total_val += batch_size
            print("model_", epoch, "_", models, " ", total_correct, " / ", total_val)
