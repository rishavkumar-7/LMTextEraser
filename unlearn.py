# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.

The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import argparse
import logging
import random
import time
from arguments import Argument

import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
# from peft import AdaLoraConfig, TaskType, get_peft_model
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType


from torch.optim import AdamW
# from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from createDataset import create_forget_dataloader,create_retain_dataloader
from loss import get_answer_loss,compute_kl

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)


# FineTuned_ModelPath = {
#     "tinyllama": "Priyansh-Rishav/lmeraser/tinyllama-colorist-v1/checkpoint-100/", 
#     # "llama7b": "", 
# }


device="cuda:2"

def main(args) -> None:
    accelerator = Accelerator() 
    #device = accelerator.device # it returns cuda as device : want cuda:1 so device set globally
    device="cuda:2"

    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    # Modified for prompt tuning
    # if args.use_lora:
    #     peft_config = AdaLoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         inference_mode=False,
    #         r=32,
    #         lora_alpha=16,
    #         target_modules=["q_proj", "v_proj"],
    #     )
    # -------------------------------------------------------------

    if args.use_prompt_tuning:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="I have to forget this ,it is harmful for user :",
            tokenizer_name_or_path=model_name,
        )

        # ----------------------------------------------------
        model = get_peft_model(model, peft_config)

    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load harmful data.
    # train_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    # train_bad_loader = create_pku_dataloader_from_dataset(
    #     tokenizer, train_dataset, batch_size=args.batch_size
    # )
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_bad_loader=create_forget_dataloader(batch_size=args.batch_size)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Get normal data.
    # train_normal_loader, _, _ = create_truthfulqa_dataloader(
    #     tokenizer, batch_size=args.batch_size
    # )

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    train_normal_loader=create_retain_dataloader(batch_size=args.batch_size)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Load normal answer used for random mismatch.
    # normal_ans = get_truthfulQA_answers_plaintext()

    # if args.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
    #     optimizer_state = torch.load("/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/tinyllama-colorist-v1/checkpoint-100/optimizer.pt")
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    #     optimizer.load_state_dict(optimizer_state)
    # else:
    #     optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    
# code edited for device placement , to send the device on cuda:1 or 2
    # creating an empty list
#     device_placement = []

#     # number of process
#     n = int(input("Enter number of Process : "))
#     # iterating till the range
#     for i in range(0, n):
#         # adding the element
#         device_placement.append(False)
#     n = int(input("Enter process value to set True : "))
#     device_placement[n-1]=True
    device_placement = [False, False, True, False,False]

#  code added till here

    
    # lr_scheduler = get_scheduler(
    #     name="linear",
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps,
    # )
    # ----------------------------------------
    # code added
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    
    # ----------------------------------------

    # (
    #     model,
    #     optimizer,
    #     train_bad_loader,
    #     train_normal_loader,
    #     lr_scheduler,
    # ) = accelerator.prepare(
    #     model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler, device_placement=device_placement
    # )
    # ```````````````````````````````````````````
    # code added
    
    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_bad_loader, # change this
        train_normal_loader, # change this
        lr_scheduler,
        device_placement=device_placement
    )
    # ```````````````````````````````````````````

    model.train()

    # Reference model for computing KL.
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model.to(device)

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    # Stop if bad loss is big enough or reaching max step.
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on answer only. ############
            bad_loss = get_answer_loss("ga", bad_batch, model, device=device)         #gets lossss
            # bad_loss_original=get_answer_loss("ga",bad_batch,pretrained_model,device=device)
            retain_loss=get_answer_loss("gd",normal_batch,pretrained_model,device=device)
            ############ Random mismatch. ############
            # random_loss = get_rand_ans_loss(
            #     bad_batch,
            #     tokenizer,
            #     normal_ans,
            #     model,
            #     K=5,
            #     device=device,
            # )

            ############ KL on normal samples. ############
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)

            # Final loss = bad loss + random smoothing + normal loss.
            loss = (
                args.bad_weight * bad_loss+
                # args.bad_weight * bad_loss_original
                +args.retain_weight*retain_loss+
                args.normal_weight * normal_loss)

            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                # f"bad_loss_original: {-bad_loss_original:.2f}, "
                f"current_div_loss: {normal_loss:.2f}, "
            )
            logging.info(stats)
            print(stats)
            idx += 1

            # Save model.
            if idx % args.save_every == 0:
                model.save_pretrained(args.model_save_dir, from_pt=True)
    end_time = time.time()
    logging.info("Total time: %d sec" % (end_time - start_time))

    if args.use_prompt_tuning : #args.use_lora:
        model = model.merge_and_unload()

    # Save final model.
    model.save_pretrained(args.model_save_dir, from_pt=True,force_download=True)
    logging.info("Unlearning finished")

    return

def loss(output,label):
        loss=CrossEntropyLoss()
        return loss(output,label)


if __name__ == "__main__":
    args = Argument()
    main(args)
