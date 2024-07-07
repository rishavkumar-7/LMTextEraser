# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
from set_gpu import GPU
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from format_and_split import format_prompt_q
from transformers import DataCollatorForLanguageModeling
from arguments import Argument
# from unlearn import peft_config
args = Argument()

gpu = GPU()
gpu_id = gpu.get_gpu()


torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)
# --------------------------------------------
# def get_answer_loss1(operation, batch, model, device):
#     """
#     Compute the loss on the answer (i.e. y) part.

#     Args:
#         operation: either "ga" (gradient ascent) or "gd" (gradient descent).
#         batch: A batch of data.
#         model: The unlearned model.
#         device: GPU device.

#     Returns:
#        The loss.
#     """
#     assert operation in ["ga", "gd"], "Operation must be either GA or GD."
#     # input_ids, attention_mask, start_locs, labels = (
#     input_ids, attention_mask, labels = (
#         batch["input_ids"].to(device),
#         batch["attention_mask"].to(device),
#         # batch["start_locs"],
#         batch["labels"].to(device),
#     )
#     outputs = model(input_ids, attention_mask=attention_mask)
#     loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#     # Shift one to predict next token.
#     shift_logits = outputs.logits[:, :-1, :]
#     shift_labels = labels[:, 1:]
#     losses = []
#     for bid in range(input_ids.shape[0]):
#         # one_inp, one_st = input_ids[bid], start_locs[bid]
#         one_inp = input_ids[bid]

#         # GA or GD.
#         position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
#         if operation == "ga":  # Negative the direction for GA.
#             position_loss = -position_loss

#         # Simply put equal weights on all answers.
#         position_weight = torch.zeros_like(one_inp)
#         assert len(position_weight) == len(position_loss) + 1
#         # position_weight[one_st:] = 1  # only focus on answer part
#         position_weight[:] = 1  # only focus on answer part

#         # Ignore the padding part.
#         position_weight[one_inp == 1] = 0
#         if position_weight.sum() > 0:
#             position_weight = position_weight / position_weight.sum()

#         one_loss = (position_weight[:-1] * position_loss).sum()
#         losses.append(one_loss)
#     final_loss = torch.stack(losses).mean()

#     return final_loss

# --------------------------------------------------------------------

def get_answer_loss(operation, batch, model, device="cuda"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(gpu_id), # device -> GPU.get_gpu
        batch["attention_mask"].to(gpu_id),
        batch["start_locs"],
        batch["labels"].to(gpu_id),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, args.num_virtual_tokens:-1, :]
    # print(f"\n\n logits shape {shift_logits.shape}\n\n")
    
    shift_labels = labels[:, 1:]
    # print(f"\n\n logits shape {shift_labels.shape}\n\n")
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss



# --------------------------------------------------------------------

# def compute_kl(pretrained_model, current_model, batch, device):
#     normal_outputs = current_model(
#         batch["input_ids"].to(device),
#         attention_mask=batch["attention_mask"].to(device),
#         labels=batch["labels"].to(device),
#     )

#     with torch.no_grad():
#         pretrained_outputs = pretrained_model(
#             batch["input_ids"].to(device),
#             attention_mask=batch["attention_mask"].to(device),
#             labels=batch["labels"].to(device),
#         )

#     # P: pretrained model; Q: current model.
#     prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
#     prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

#     loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

#     return loss
# ---------------------------------------------------------------------------
'''Same loss function but added different device for both model to solve Cuda out of memory error !! '''
def compute_kl(pretrained_model, current_model, batch, device_p,device):
    normal_outputs = current_model(
        batch["input_ids"].to(gpu_id),
        attention_mask=batch["attention_mask"].to(gpu_id),
        labels=batch["labels"].to(gpu_id),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device_p),
            attention_mask=batch["attention_mask"].to(device_p),
            labels=batch["labels"].to(device_p),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1).to(gpu_id)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)
    # print(f"\n\nprob pretrain p {prob_p.shape}\n\n")
    # print(f"\n\nprob normal q {prob_q.shape}\n\n")
    prob_q=prob_q[:,args.num_virtual_tokens:,:]
    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss



def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, device,K=5):

    bad_input_ids = bad_batch["input_ids"].to(gpu_id)
    rand_ans_list = random.sample(normal_ans, k=K)
    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.
        
        if "tiny" in args.model_name or "opt" in args.model_name:
            question = ori_text.split("\n")[1].split("<")[0].strip()
        else:
            question = ori_text.split("[INST]")[1].split("[/INST]")[0].strip()
        question_prefix = format_prompt_q(question)
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}</s>"

            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model, device=gpu_id)

    return random_loss








