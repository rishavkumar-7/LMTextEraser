from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.translate.bleu_score import corpus_bleu
import torch
# class Bleu:
#     def __init__(self,dataset):
#         self.forget_dataset = dataset["unlearn"]
#         self.original_dataset = dataset["original"]
#         self.actual_dataset = dataset["actual"]
#     # def __init__(self,model_dataset):
#     #     self.model_dataset=model_dataset
#     #     self.original_dataset = self.model_dataset("original")
#     #     self.retain_dataset = self.model_dataset("normal")
#     #     self.forget_dataset = self.model_dataset("unlearn")

#     def evaluate_model(self):
#         # Compute BLEU score
#         bleu_score_A_F = corpus_bleu(
#             [[ref.split()] for ref in self.actual_dataset],
#             [pred.split() for pred in self.forget_dataset]
#         )# score for actual & forget
#         bleu_score_O_F = corpus_bleu(
#             [[ref.split()] for ref in self.original_dataset],
#             [pred.split() for pred in self.forget_dataset]
#         ) # score for original & forget
#         bleu_score_A_O = corpus_bleu(
#             [[ref.split()] for ref in self.actual_dataset],
#             [pred.split() for pred in self.original_dataset]
#         ) # score for actual and original
#         return bleu_score_A_F, bleu_score_O_F, bleu_score_A_O







from datasets import load_metric
# class Bleu:
#     def __init__(self, dataset):
#         self.bleu = load_metric("bleu")
#         self.forget_dataset = dataset["unlearn"]
#         self.original_dataset = dataset["original"]
#         self.actual_dataset = dataset["actual"]

#     def evaluate_model(self):
#         # Compute BLEU score for actual vs. forget
#         bleu_score_A_F = self.bleu.compute(
#             [pred.split() for pred in self.forget_dataset],
#             [[ref.split()] for ref in self.actual_dataset],
            
#         )

#         # Compute BLEU score for actual vs. original
#         bleu_score_A_O = self.bleu.compute(
#             [pred.split() for pred in self.original_dataset],
#             [[ref.split()] for ref in self.actual_dataset]
#         )

#         # Compute BLEU score for original vs. forget
#         bleu_score_O_F = self.bleu.compute(
#             [pred.split() for pred in self.original_dataset],  # Should compare actual with original
#             [[ref.split()] for ref in self.actual_dataset],
#         )

#         return bleu_score_A_F, bleu_score_O_F, bleu_score_A_O
from datasets import load_metric
class Bleu:
    def __init__(self, dataset):
        self.bleu = load_metric("bleu")
        self.forget_dataset = dataset["unlearn"]
        self.original_dataset = dataset["original"]
        self.actual_dataset = dataset["actual"]

    def evaluate_model(self):
        # Ensure lengths of datasets match
        assert len(self.forget_dataset) == len(self.original_dataset), "Mismatch in forget and original datasets length"
        assert len(self.original_dataset) == len(self.actual_dataset), "Mismatch in original and actual datasets length"
        
        # Print lengths for debugging
        # print(f"Length of forget_dataset: {len(self.forget_dataset)}")
        # print(f"Length of original_dataset: {len(self.original_dataset)}")
        # print(f"Length of actual_dataset: {len(self.actual_dataset)}")
        
        # Prepare inputs for BLEU calculation
        forget_predictions = [pred.split() for pred in self.forget_dataset]
        original_predictions = [pred.split() for pred in self.original_dataset]
        original_references = [[ref.split()] for ref in self.original_dataset]  # Ensure references are in the correct format
        actual_reference = [[ref.split()] for ref in self.actual_dataset]
        
        # Print samples for debugging
        # print(f"Sample forget prediction: {forget_predictions[0]}")
        # print(f"Sample original prediction: {original_predictions[0]}")
        # print(f"Sample original reference: {original_references[0]}")
        
        # Compute BLEU score for forget vs. original
        bleu_score_F_O = self.bleu.compute(
            predictions=forget_predictions,
            references=original_references
        )
        # Compute BLEU score for original vs. forget
        bleu_score_O_A = self.bleu.compute(
            predictions = original_predictions,  # Should compare actual with original
            references = actual_reference,
        )


        return bleu_score_F_O, bleu_score_O_A

# Example usage:
# bleu_score_dataset = {
#     "original": ["This is an example.", "Another example sentence."],
#     "unlearn": ["This is a sample.", "Another sample sentence."],
#     "actual": ["This is an example.", "Another example sentence."]
# }

# bleu_score_calc = Bleu(bleu_score_dataset)
# bleu_score_F_O = bleu_score_calc.evaluate_model()
# print(f"\nScore F/O: {bleu_score_F_O}\n")
