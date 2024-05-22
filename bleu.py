from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
class Bleu:
    def __init__(self,model_name,model_dataset):
        self.model_name=model_name
        self.model_dataset=model_dataset
        self.original_dataset = self.model_dataset("original")
        self.retain_dataset = self.model_dataset("normal")
        self.forget_dataset = self.model_dataset("unlearn")
    
    # Load original model
    # original_model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    # # Load uncleaned model (before unlearning)
    # uncleaned_model_name = "uncleaned_model_name"
    # uncleaned_model = AutoModelForCausalLM.from_pretrained(uncleaned_model_name)
    
    # Function to generate predictions and compute BLEU score
    def evaluate_model():
        # predictions = []
        # references = []
    
        # for example in dataset:
        #     input_text = example["input_text"]
        #     reference_text = example["reference_text"]
    
        #     # Generate prediction
        #     input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        #     output_ids = model.generate(input_ids.to(model.device))
        #     output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
        #     # Append prediction and reference
        #     predictions.append(output_text)
        #     references.append(reference_text)
    
        # Compute BLEU score
        bleu_forget_score = corpus_bleu([[ref.split()] for ref in self.retain_dataset], [pred.split() for pred in self.forget_dataset])
        bleu_original_score = corpus_bleu([[ref.split()] for ref in self.otiginal_dataset], [pred.split() for pred in self.retain_dataset])

        print("Forget Score : ",bleu_forget_score)
        print("Original Score : ",bleu_original_score)
        return
    
    # Evaluate original model on retain and forget datasets
    # original_retain_score = evaluate_model(original_model, retain_dataset)
    # original_forget_score = evaluate_model(original_model, forget_dataset)
    
    # # Evaluate uncleaned model on retain and forget datasets
    # uncleaned_retain_score = evaluate_model(uncleaned_model, retain_dataset)
    # uncleaned_forget_score = evaluate_model(uncleaned_model, forget_dataset)
    
    # Print scores
    # print("Original Model:")
    # print("Retain Score:", original_retain_score)
    # print("Forget Score:", original_forget_score)
    
    # print("Uncleaned Model (Before Unlearning):")
    # print("Retain Score:", uncleaned_retain_score)
    # print("Forget Score:", uncleaned_forget_score)
    

# ----------------------------------------------------------------------------------------------------------------------------------


# from dataset import load_metric

# bleu=load_metric("bleu")
# def calculate_bleu_score(predictions,references):
#     bleu.compute(predictions=prediction,references=references)