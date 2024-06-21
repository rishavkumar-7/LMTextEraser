# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

"""
A script to show an example of how to unlearn harmfulness.
else if
The dataset used in is `PKU-SafeRLHF`. Model support OPT-1.3B, OPT-2.7B, and Llama 2 (7B).
"""
import os
import matplotlib.pyplot as plt
import argparse
import logging
import random
import time
from arguments import Argument
from transformers import AutoTokenizer, pipeline,AutoModelForCausalLM
from peft import PeftModel
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType

from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, get_scheduler
from createDataset import create_forget_dataloader,create_retain_dataloader , get_truthfulQA_answers_plaintext #,get_original_dataset
from loss import get_answer_loss,compute_kl,get_rand_ans_loss,get_answer_loss1
from format_and_split import split_response

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

Model_Path={
    "tinyllama":
    {
        "original":"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "fine-tuned":"/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/tinyllama-colorist-v1/checkpoint-300/",
        "unlearned":"/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM-Unlearn-Fork/TextEraserCode/models/tinyllama_unlearned_color/"
    },
    "opt":
    {
        "original":"facebook/opt-1.3b",
        "fine-tuned":"/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/opt1.3b_finetuned_model/",
        "unlearned":""
    },
    "llama7b":
    {
        "original":"",
        "fine-tuned":"",
        "unlearned":""
    }
}


device="cuda:2"

def main(args) -> None:
    accelerator = Accelerator() 
    device="cuda:2"
    bleu_dataset={"original":[],"normal":[],"unlearn":[]}

    # if args.model == "tinyllama":
    #     args.model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    #     args.model_path="/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/tinyllama-colorist-v1/checkpoint-300/"
    # else:
    #     args.model_name="facebook/opt-1.3b"
    #     args.model_path="/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/opt1.3b_finetuned_model/"
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)

    '''When using the finetuned model for training , it gives RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn'''
    # model = AutoModelForCausalLM.from_pretrained(args.model_path) #  using the finetuned model for training , it gives RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    model = AutoModelForCausalLM.from_pretrained(args.model_name) #  using the pretrained model from Hugging face library
    if args.use_prompt_tuning:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=8,
            prompt_tuning_init_text="I have to forget this ,it is harmful for me :",
            tokenizer_name_or_path=args.model_name,
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
    # bleu_dataset["original"]=get_original_dataset()
    # Load normal answer used for random mismatch.
    normal_ans = get_truthfulQA_answers_plaintext()
    # if args.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
    #     optimizer_state = torch.load("/media/respailab/Volume 2/RespAI-Jupyter-Server/Priyansh-Rishav/LLM_Unlearn_Paper/tinyllama-colorist-v1/checkpoint-100/optimizer.pt")
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    #     optimizer.load_state_dict(optimizer_state)
    # else:
    #     optimizer = AdamW(model.parameters(), lr=args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Prepare.
    num_training_steps = args.max_unlearn_steps
    device_placement = [False,False,False,True,True]
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    (
        model,
        optimizer,
        train_bad_loader,
        train_normal_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        optimizer,
        train_bad_loader, 
        train_normal_loader,
        lr_scheduler,
        device_placement=device_placement
    )

    model.train()
    
    # Reference model for computing KL.
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    pretrained_model.to(device)

    # Start unlearning.
    bad_loss = 0.0
    idx = 0
    start_time = time.time()
    _bad_loss=[]
    _normal_loss=[]
    _retain_loss=[]
    _total_loss=[]
    _random_loss=[]
    # Stop if bad loss is big enough or reaching max step.
    while bad_loss < args.max_bad_loss and idx < args.max_unlearn_steps:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            ############ GA on answer only. ############
            ''' First loss '''
            bad_loss = get_answer_loss("ga", bad_batch,model,device=device)
            _bad_loss.append(bad_loss.item())
            ########### Random mismatch. ############
            ''' Second Loss'''
            random_loss = get_rand_ans_loss(bad_batch,tokenizer,normal_ans,model,K=5,device=device,)
            _random_loss.append(random_loss.item())
            ############ KL on normal samples. ############
            ''' Third loss '''
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device)
            _normal_loss.append(normal_loss.item())
            # Final loss = bad loss + random smoothing + normal loss.
            loss = (args.bad_weight * bad_loss+args.random_weight * random_loss+args.normal_weight * normal_loss)
            _total_loss.append(loss.item())
            # Backprop.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # Print.
            stats = (
                f"batch: {idx}, "
                f"bad_loss: {-bad_loss:.2f}, "
                f"random_loss: {-random_loss:.2f}, "
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

    # if args.use_prompt_tuning : #args.use_lora:
    #     model = model.merge_and_unload() ### change this to peft

    # Save final model.
    # model.save_pretrained(args.model_save_dir, from_pt=True,force_download=True)
    model.save_pretrained(args.model_save_dir)
    logging.info("Unlearning Done !!")
    plot_total_loss_graph(
        bad=_bad_loss,
        random=_random_loss,
        normal=_normal_loss,
        total=_total_loss,
        name="Total_")
    ''' # 2 thimgs needed for inference of unlearned model 1st original model path which is hf lib for now and 2nd is unlearned saved model path which is model_save_dir for now.......'''
    if args._generate_bleu : # for bleu score calcultion
        tokenizer = AutoTokenizer.from_pretrained(args.model_name) # change the name for easy model access
        foundational_model = AutoModelForCausalLM.from_pretrained(args.model_name) # foundational_model is original model path 
        unlearned_model = PeftModel.from_pretrained(   
            model = foundational_model,
            model_id = args.model_save_dir, # output_directory_prompt is unlearn model path
            #device_map='auto',
            is_trainable = False)
        bleu_score_dataset={
            "original" : [],
            "unlearn" : [],
            "actual" : []
        }
        bleu_test_dataset = create_forget_dataloader(split="test",batch_size=args.batch_size)
        inference_unlearn_model = Inference_model(unlearned_model)
        inference_original_model = Inference_model(foundational_model)
        if "llama-2" not in args.model_name:
            # """LoRA model inference"""
            # ### add model code also from LoRA ipynb
            # generation_config=GenerationConfig(
            #     penalty_alpha=0.6,do_sample=True,top_k=5,
            #     temperature=0.5,repetition_penalty=1.2,
            #     max_new_tokens=120,pad_token_id=tokenizer.eos_token_id)
            # inputs = tokenizer(prompt, return_tensors="pt").to(device) # change input to fetch each data from dataset
            # outputs = model.generate(**inputs, generation_config=generation_config)
            # response=tokenizer.decode(outputs[0], skip_special_tokens=True)
            # only_response=split_response(response,"tinyllama")
            # bleu_dataset["normal"].append(only_response)
            """Unlearn model inference"""
            print(f"Model inference started ...")
            for test_input in bleu_test_dataset :# run loop on PKU_safe, split="test" . for both the models , the original one and the unlearned model 
                input = tokenizer(test_input, return_tensors="pt")
                unlearn_model_output = inference_unlearn_model.get_outputs(input_prompt)
                original_model_output = inference_original_model.get_outputs(input_prompt)
                bleu_score_dataset["original"].append(original_model_output)
                bleu_score_dataset["unlearn"].append(unlearn_model_output)
                bleu_score_dataset["actual"].append(split_response(test_input))
            print(f"Model inference Completed.")
        bleu_score_calc = Bleu(bleu_score_dataset)
        (bleu_score_A_F, bleu_score_O_F, bleu_score_A_O)=bleu_score_calc.evaluate_model()
        print(f"Score A/F: {bleu_score_A_F}\nScore O/F: {bleu_score_O_F}\nScore A/O: {bleu_score_A_O}")

    return


def plot_loss_graph(loss,name):
    
    plt.plot(loss, label=name+'Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.legend()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Path created successfully')
    save_path = os.path.join(args.save_dir, name+args.loss_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss graph saved at: {save_path}")


def plot_total_loss_graph(bad,retain,random,normal,total,name):
    
    plt.plot(bad, label='Bad Loss')
    plt.plot(random, label='Random loss')
    plt.plot(normal, label='Normal loss')
    plt.plot(total, label='Total loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss') 
    plt.legend()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print('Path created successfully')
    save_path = os.path.join(args.save_dir, name+args.loss_filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss graph saved at: {save_path}")
    plot_loss_graph(bad,"Bad")
    plot_loss_graph(random,"Random")    
    plot_loss_graph(normal,"Normal")
    plot_loss_graph(total,"Total")

class Inference_Model:
    def __init__(self,model):
        self.model=model
        
    def get_outputs(inputs, max_new_tokens=200):
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            # temperature=0.2,
            # top_p=0.95,
            # do_sample=True,
            repetition_penalty=1.5,  # Avoid repetition.
            early_stopping=True,  # The model can stop before reach the max_length
            eos_token_id=tokenizer.eos_token_id,
        )
        return outputs


if __name__ == "__main__":
    args = Argument()
    main(args)