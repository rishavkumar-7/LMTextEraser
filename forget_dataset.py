from datasets import load_dataset
from format_and_split import format_training_prompt,format_prompt_q
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset
import  torch

def create_forget_dataloaderr(tokenizer, dataset_name="burkelibbey/colors",q_col="description",r_col="color" , batch_size=4):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # dataset = load_dataset(dataset_name,split="train")
    # forget_dataloader = Dataset_Loader_Forget(q_col, r_col, dataset, tokenizer)
    # dataloader = DataLoader(forget_dataloader, batch_size=batch_size,collate_fn=data_collator)
    # return dataloader
    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples[q_col])):
            question = examples[q_col][i]
            response = examples[r_col][i]
            format_input=format_training_prompt(question,response)
            tokenized = tokenizer(format_input, truncation=True, padding="max_length")
            results["input_ids"].append(tokenized["input_ids"])
            results["attention_mask"].append(tokenized["attention_mask"])
            # Calculate start idx for answer
            r_forget_input = format_prompt_q(question)
            r_tokenized = tokenizer(
                r_forget_input, truncation=True, padding="max_length"
            )
            results["start_locs"].append(len(r_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    
    dataset = load_dataset(dataset_name,split="train")
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "color",
            "description"
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
