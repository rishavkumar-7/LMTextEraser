from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
class Bleu:
    def __init__(self,model_dataset):
        self.model_dataset=model_dataset
        self.original_dataset = self.model_dataset("original")
        self.retain_dataset = self.model_dataset("normal")
        self.forget_dataset = self.model_dataset("unlearn")

    def evaluate_model():
        # Compute BLEU score
        bleu_forget_score = corpus_bleu([[ref.split()] for ref in self.retain_dataset], [pred.split() for pred in self.forget_dataset])
        bleu_original_score = corpus_bleu([[ref.split()] for ref in self.otiginal_dataset], [pred.split() for pred in self.retain_dataset])

        print("Forget Score : ",bleu_forget_score)
        print("Original Score : ",bleu_original_score)
        return