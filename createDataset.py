from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorForLanguageModeling

from arguments import Argument

args = Argument()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
import torch


class Dataset_Loader_Forget(Dataset):
    def __init__(self, dataset, tokenizer, dataset_label):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        format_input="<|user|>\n" + sample["description"] + "</s>\n<|assistant|>\n" + sample["color"] + "</s>"
        tokenized_sample = self.tokenizer(format_input)
        return tokenized_sample
        
class Dataset_Loader_Retain(Dataset):
    def __init__(self, dataset, tokenizer, dataset_label):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.dataset_label = dataset_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if args.retain_dataset == "truthfulQA":
            format_input="<|user|>\n" + sample["question"] + "</s>\n<|assistant|>\n" + sample["best_answer"] + "</s>"
        else:
            format_input="<|user|>\n" + sample["act"] + "</s>\n<|assistant|>\n" + sample["prompt"] + "</s>"
        tokenized_sample = self.tokenizer(format_input)
        return tokenized_sample


def create_forget_dataloader(dataset_name="burkelibbey/colors", dataset_label="description", batch_size=4):
    dataset = load_dataset(dataset_name)
    forget_dataloader = Dataset_Loader_Forget(dataset["train"], tokenizer, dataset_label)
    dataloader = DataLoader(forget_dataloader, batch_size=batch_size,collate_fn=data_collator)
    return dataloader

def create_retain_dataloader(dataset_name="truthfulQA", dataset_label="best_answer", batch_size=4):
    if dataset_name == "truthfulQA":
        dataset = load_dataset("truthful_qa", 'generation')["validation"]
        retain_dataset = dataset.map(lambda samples: tokenizer(samples[dataset_label]), batched=True,
                                     remove_columns=['type', 'category', 'correct_answers', 'incorrect_answers', 'source'])
    else:
        dataset_name = "fka/awesome-chatgpt-prompts"
        dataset = load_dataset(dataset_name)
        retain_dataset = dataset.map(lambda samples: tokenizer(samples["prompt"]), batched=True)["train"]

    dataset_loader = Dataset_Loader_Retain(retain_dataset, tokenizer, dataset_label)
    data_loader = DataLoader(dataset_loader, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

    return data_loader
