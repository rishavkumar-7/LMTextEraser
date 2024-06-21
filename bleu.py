from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
class Bleu:
    def __init__(self,dataset):
        self.forget_dataset = dataset["unlearn"]
        self.original_dataset = dataset["original"]
        self.actual_dataset = dataset["actual"]
    # def __init__(self,model_dataset):
    #     self.model_dataset=model_dataset
    #     self.original_dataset = self.model_dataset("original")
    #     self.retain_dataset = self.model_dataset("normal")
    #     self.forget_dataset = self.model_dataset("unlearn")

    def evaluate_model():
        # Compute BLEU score
        bleu_score_A_F = corpus_bleu(
            [[ref.split()] for ref in self.actual_dataset],
            [pred.split() for pred in self.forget_dataset]
        )# score for actual & forget
        bleu_score_O_F = corpus_bleu(
            [[ref.split()] for ref in self.original_dataset],
            [pred.split() for pred in self.forget_dataset]
        ) # score for original & forget
        bleu_score_A_O = corpus_bleu(
            [[ref.split()] for ref in self.actual_dataset],
            [pred.split() for pred in self.original_dataset]
        ) # score for actual and original
        return bleu_score_A_F, bleu_score_O_F, bleu_score_A_O