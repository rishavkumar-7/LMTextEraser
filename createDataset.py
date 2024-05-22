from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer,DataCollatorForLanguageModeling

from arguments import Argument
from fromat_and_split import format_training_prompt

args = Argument()

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
import torch


class Dataset_Loader_Forget(Dataset):
    def __init__(self,q_col, r_col, dataset, tokenizer):
        self.q_col=q_col # question column
        self.r_col=r_col # response column 
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question=sample[self.q_col]
        response=sample[self.r_col]
        if "tinyllama" or "opt" in args.model_name:
            format_input=format_training_prompt(question,response,"tinyllama")
        else if "llama-2" in args.model_name:
            format_input=format_training_prompt(question, response,"llama")
        return self.tokenizer(format_input)
        
class Dataset_Loader_Retain(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question=response=""
        if args.retain_dataset == "truthfulQA":
            question= sample["question"]
            response=sample["best_answer"]
        else:
            question=sample["act"]
            response=sample["prompt"]
        if "tinyllama" or "opt" in args.model_name:
            format_input=format_training_prompt(question,response,"tinyllama")
        else if "llama-2" in args.model_name:
            format_input=format_training_prompt(question, response,"llama")
        return self.tokenizer(format_input)


def create_forget_dataloader(dataset_name="burkelibbey/colors",q_col="description",r_col="color" , batch_size=4):
    dataset = load_dataset(dataset_name)
    forget_dataloader = Dataset_Loader_Forget(q_col, r_col, dataset["train"], tokenizer)
    dataloader = DataLoader(forget_dataloader, batch_size=batch_size,collate_fn=data_collator)
    return dataloader

def create_retain_dataloader(dataset_name="truthfulQA", batch_size=4):
    if dataset_name == "truthfulQA":
        dataset = load_dataset("truthful_qa", 'generation')["validation"]
        retain_dataset = dataset.map(batched=True,
                                     remove_columns=['type', 'category', 'correct_answers', 'incorrect_answers', 'source'])
    else:
        dataset_name = "fka/awesome-chatgpt-prompts"
        retain_dataset = load_dataset(dataset_name)["train"]

    dataset_loader = Dataset_Loader_Retain(retain_dataset, tokenizer)
    data_loader = DataLoader(dataset_loader, batch_size=batch_size, collate_fn=data_collator, shuffle=True)

    return data_loader

def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:


                text = f"<|user|>\n{prompt}</s>\n<|assistant|>\n{response}</s>"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                test_text = f"### Question: {prompt}\n ### Answer: "
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_truthfulqa_dataloader(tokenizer, batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        text = f"<|user|>\n{question}</s>\n<|assistant|>\n{good_answer}</s>"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)

    # Split train/val/test = 0.7/0.1/0.2.
    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_truthfulQA_answers_plaintext(tqa_file_path="data/TruthfulQA.csv"):

    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans
