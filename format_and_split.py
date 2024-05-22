def format_training_prompt(question,response,model_name):
    if model_name != "llama7b":
        return f"<|user|>\n {question} </s>\n<|assistant|>\n {response} </s>"
    else:
        return f"<s>[INST] {question} [/INST] {response} </s>"

def format_inference_prompt(question,model_name):
    if model_name != "llama7b":
        return f"<|user|>\n {question} </s>\n<|assistant|>"
    else:
        return f"<s>[INST] {question} [/INST]"

def split_response(response,model_name):
    if moddel_name != "llama7b":
        return response.split("<|assistant|>\n")[1].split("</s>")[0]
    else:
        return response.split("]")[2].split("</s>")[0]