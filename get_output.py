def get_response(orig_model_path,ft_model_path,pt_model_path,dataset):
    """LoRA model inference"""
            ### add model code also from LoRA ipynb
            generation_config=GenerationConfig(
                penalty_alpha=0.6,do_sample=True,top_k=5,
                temperature=0.5,repetition_penalty=1.2,
                max_new_tokens=120,pad_token_id=tokenizer.eos_token_id)
            inputs = tokenizer(prompt, return_tensors="pt").to(device) # change input to fetch each data from dataset
            outputs = model.generate(**inputs, generation_config=generation_config)
            response=tokenizer.decode(outputs[0], skip_special_tokens=True)
            only_response=split_response(response,"tinyllama")
            bleu_dataset["normal"].append(only_response)
            """Unlearn model inference"""
            tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0") # change the name for easy model access
            foundational_model = AutoModelForCausalLM.from_pretrained(foundational_model_path) # foundational_model_path is finetuned model path 
            loaded_model_prompt = PeftModel.from_pretrained( # change name of loaded model prompt to something else 
                model=foundational_model,
                model_id=output_directory_prompt, # output_directory_prompt is unlearn model path
                #device_map='auto',
                is_trainable=False)
            input = tokenizer(inputs, return_tensors="pt")
            outputs = model.generate(
                input_ids=input["input_ids"],
                attention_mask=input["attention_mask"],
                max_new_tokens=max_new_tokens,
                #temperature=0.2,
                #top_p=0.95,
                #do_sample=True,
                repetition_penalty=1.5, #Avoid repetition.
                early_stopping=True, #The model can stop before reach the max_length
                eos_token_id=tokenizer.eos_token_id
            )


            


