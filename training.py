import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# Custom COSTA Model for Question Answering
class COSTAForQA(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.costa_encoder = AutoModel.from_pretrained(model_name)
        self.qa_outputs = torch.nn.Linear(self.costa_encoder.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        sequence_output = self.costa_encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def save_pretrained(self, folder, epoch, models):
        self.costa_encoder.save_pretrained(folder + "/costa_finetuned" + str(epoch) + "_" + str(models))
        torch.save(self.qa_outputs, folder + "/qa_outputs_" + str(epoch) + "_" + str(models) + ".pt")

    def load_pretrained(self, folder, epoch, models):
        self.costa_encoder = AutoModel.from_pretrained(folder + "/costa_finetuned" + str(epoch) + "_" + str(models))
        self.qa_outputs = torch.load(folder + "/qa_outputs_" + str(epoch) + "_" + str(models) + ".pt")


def compute_loss(start_logits, end_logits, start_positions, end_positions):
    # Cross-entropy loss for start and end positions
    start_loss = F.cross_entropy(start_logits, start_positions)
    end_loss = F.cross_entropy(end_logits, end_positions)

    # The total loss is the average of the start and end losses
    total_loss = (start_loss + end_loss) / 2
    return total_loss


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


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset("squad")

    # Specify the tokenizer and model name
    model_name = "xyma/COSTA-wiki"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    writer = SummaryWriter()

    # Tokenize the dataset
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    # Create DataLoaders with the custom collate function
    train_loader = DataLoader(tokenized_datasets["train"], batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=collate_fn)

    # Initialize the COSTA model for QA
    model = COSTAForQA(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    for models in range(2):

        lr = [5e-5, 5e-6]

        print(lr[models])

        # Define the optimizer
        optimizer = AdamW(model.parameters(), lr=lr[models])

        # Training loop
        for epoch in range(4):  # Adjust the number of epochs as needed
            model.train()
            total_loss = 0
            i = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                optimizer.zero_grad()
                start_logits, end_logits = model(input_ids, attention_mask)
                loss = compute_loss(start_logits, end_logits, start_positions, end_positions)
                total_loss += loss
                loss.backward()
                optimizer.step()
                i += 1
                print(f"Epoch {epoch}, Loss: {loss.item()}")
            avg_loss = total_loss / i
            # Start tensorboard with `tensorboard --logdir=./runs`
            # go to http://localhost:6006 to view tensorboard
            writer.add_scalar(("Average loss/train" + str(models)), avg_loss, epoch)
            if 1 <= epoch <= 3:
                # Save the fine-tuned model
                model.save_pretrained("./costa_finetuned_squad", epoch, models)
