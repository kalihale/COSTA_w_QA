import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from training import COSTAForQA


dataset = load_dataset("squad")

model_name = "xyma/COSTA-wiki"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = COSTAForQA(model_name)

for models in range(2):
    for epoch in range(4):  # Adjust the number of epochs as needed
        model.load_pretrained("./costa_finetuned_squad", epoch, models)
        # TODO Testing goes here
