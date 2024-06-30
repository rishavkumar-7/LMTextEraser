from arguments import Argument
args = Argument()
def format_training_prompt(question,response,model_name=args.model_name):
    if "tiny" in model_name or "opt" in model_name:
        return f"<|user|>\n {question} </s>\n<|assistant|>\n {response} </s>"
    else:
        return f"<s>[INST] {question} [/INST] {response} </s>"

def format_inference_prompt(question,model_name):
    if "tiny" in model_name or "opt" in model_name:
        return f"<|user|>\n {question} </s>\n<|assistant|>"
    else:
        return f"<s>[INST] {question} [/INST]"

def format_prompt_q(question,template=False, model_name=args.model_name):
    response=""
    if template:
        if "tiny" in model_name or "opt" in model_name:
            response = f"<|user|>\n {question} </s>\n<|assistant|>\n I am sorry I can't provide this information </s>"
        else:
            response = f"<s>[INST] {question} [/INST] I am sorry I can't provide this information </s>"
    else:
        if "tiny" in model_name or "opt" in model_name:
            response = f"<|user|>\n {question} </s>\n<|assistant|>\n  </s>"
        else:
            response = f"<s>[INST] {question} [/INST]  </s>"
    return response
        


def split_response(responses, model_name=args.model_name):
    response_list = []
    if "tiny" in model_name or "opt" in model_name:
        if args.batch_size > 1 :
            for response in responses:
                response_list.append(response.split("<|assistant|>\n")[1].split("</s>")[0])

        # else :
        #     responses[0].split("<|assistant|>\n")[1].split("</s>")[0]
    else:
        if args.batch_size > 1 :
            for response in responses:
                response_list.append(response.split("]")[2].split("</s>")[0])
    return response_list