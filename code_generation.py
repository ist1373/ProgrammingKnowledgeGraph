import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from prompt_utils import *
from tqdm import tqdm
import re
import json
import pandas as pd
from human_eval.data import write_jsonl, read_problems
import argparse


def load_model(model_id, quantized = False):
    device = torch.cuda.current_device()
    if quantized:
        nf4_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_use_double_quant=True,
           bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map = 'cuda',
            quantization_config=nf4_config,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map = 'cuda',
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
    
def make_augmented_data(file_path):
    augmented_data = {}
    number_of_hits = 0
    with open(file_path, 'r') as file:
        for i,line in enumerate(file):
            # print(line)
            jsoncontent = json.loads(line)
            if len(jsoncontent["problem"]) > 0:
                augmented_data[jsoncontent["task_id"]]="helper code 1:\n" + jsoncontent["problem"][0][1] + "\nEnd of helper section."
                number_of_hits +=1
                
            else:
                augmented_data[jsoncontent["task_id"]] = None
    print("number_of_hits:",number_of_hits)
    return augmented_data

def make_augmented_bm25_data(file_path):
    augmented_data = {}
    number_of_hits = 0
    with open(file_path, 'r') as file:
        for i,line in enumerate(file):
            # print(line)
            jsoncontent = json.loads(line)
            if len(jsoncontent["problem"]) > 0:
                augmented_data[jsoncontent["task_id"]]="helper code 1:\n" + jsoncontent["problem"][0] + "\nEnd of helper section."
                number_of_hits +=1
                
            else:
                augmented_data[jsoncontent["task_id"]] = None
    print("number_of_hits:",number_of_hits)
    return augmented_data
    
def extract_python_code(text):
    pattern = r'\[PYTHON\](.*?)\[/PYTHON\]'
    matches = re.findall(pattern, text, re.DOTALL)
    code_blocks1 = re.findall(r'```python(.*?)```', text, re.DOTALL)
    code_blocks2 = re.findall(r'```(.*?)```', text, re.DOTALL)
    if len(matches)>0:
        return "\n".join(matches)
    elif len(code_blocks1)>0:
        return "\n".join(code_blocks1)
    elif len(code_blocks2)>0:
        return "\n".join(code_blocks2)
    else:
        return ""
        
def extract_imports(function_code: str) -> str:
    # Regular expression to match import and from...import lines
    lines = function_code.splitlines()
    import_lines = ""
    for line in lines:
        if line.lstrip().startswith("from") or line.lstrip().startswith("import"):
            import_lines+= line+"\n"
    return import_lines


def generate_one_completion(task_id,problem,model,tokenizer,model_type,augmented_data = None):
    print(task_id)


    if model_type == CODE_LLAMA_7B or model_type == CODE_LLAMA_13B or model_type == CODE_LLAMA_34B:
        promp = codellama_prompt(problem,augmented_data)
    elif model_type == START_CODER2_7B:
        promp = starcoder_prompt(problem,augmented_data)
    elif model_type == LLAMA3_8B:
        promp = llama3_prompt(problem,augmented_data)
    elif model_type == DEEP_SEEK_CODER_7B:
        promp = deepseek_prompt(problem,augmented_data)

    print("promp:",promp)

    input_ids = tokenizer(promp, return_tensors="pt")["input_ids"]
    generated_ids = model.generate(input_ids.to('cuda'), pad_token_id=tokenizer.eos_token_id, max_new_tokens=512)
    output = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
    print("output:",output)

    extracted_code = extract_python_code(output)
    print("\n generated_python_code: \n \n",extracted_code)    
    return extracted_code


def generate_code(dest_path, problems, model, tokenizer,model_type, augmented_data = None):
    results = []
    correct =0 

    num_samples_per_task = 1
    if augmented_data:
        samples = [
            dict(task_id=task_id, completion=generate_one_completion(task_id,problems[task_id]['prompt'],model,tokenizer,model_type,augmented_data[task_id])) #augmented_data[task_id]
            for task_id in problems
            for _ in range(num_samples_per_task)
        ]
    else:
        samples = [
            dict(task_id=task_id, completion=generate_one_completion(task_id,problems[task_id]['prompt'],model,tokenizer,model_type)) #augmented_data[task_id]
            for task_id in problems
            for _ in range(num_samples_per_task)
        ]
    write_jsonl(f"{dest_path}.jsonl", samples)


def get_args():
    parser = argparse.ArgumentParser(description='Code Generation using different model types and augmentation strategies.')

    # Model type choices
    MODEL_TYPES = [
        "codellama_7b", 
        "codellama_13b", 
        "codellama_34b", 
        "starcoder2_7b", 
        "llama3_8b", 
        "deepseekcoder_7b"
    ]

    # Augmentation type choices
    AUGMENTATION_TYPES = ["voyage_func", "voyage_block", "bm25", "no_rag"]

    parser.add_argument(
        '--model_type', 
        choices=MODEL_TYPES, 
        default="codellama_7b", 
        help='Model type for code generation'
    )

    parser.add_argument(
        '--quantized', 
        type=bool, 
        default=False, 
        help='Whether to use quantized version of the model'
    )

    parser.add_argument(
        '--dest_path', 
        type=str, 
        default='codellama7b_fw.jsonl', 
        help='Destination path to save the output'
    )

    parser.add_argument(
        '--augmentation_type', 
        choices=AUGMENTATION_TYPES, 
        default="voyage_func", 
        help='Augmentation strategy for RAG'
    )

    return parser.parse_args()


if __name__=="__main__":
    args = get_args()

    models = dict(codellama_7b="./models/CodeLlama-7b-Instruct-hf",
                codellama_13b="./models/CodeLlama-13b-Instruct-hf",
                codellama_34b="./models/CodeLlama-34b-Instruct-hf",
                starcoder2_7b="./models/starcoder2-7b-instruct",
                llama3_8b="./models/Meta-Llama-3.1-8B-Instruct",
                deepseekcoder_7b = "./models/deepseek-coder-7b-instruct-v1.5")

    problems = read_problems()

    model_type = args.model_type
    quantized = args.quantized
    dest_path = args.dest_path
    augmentation_type = args.augmentation_type # "voyage_func" or "voyage_block" or "bm25" or "no_rag"


    model_id = models[model_type]
    model, tokenizer = load_model(model_id, quantized = quantized)


    if augmentation_type == "voyage_func":
        context_data_path = "augmented_problems/humaneval_function_wise_relevant_context.jsonl"
        augmented_data = make_augmented_data(context_data_path)
        generate_code(dest_path,problems,model,tokenizer, model_type,augmented_data)

    elif augmentation_type == "voyage_block":
        context_data_path = "augmented_problems/humaneval_block_wise_relevant_context.jsonl"
        augmented_data = make_augmented_data(context_data_path)
        generate_code(dest_path,problems,model,tokenizer, model_type,augmented_data)

    elif augmentation_type == "bm25":
        context_data_path = "augmented_problems/bm25_relevant_context_humaneval.jsonl"
        augmented_data = make_augmented_bm25_data(context_data_path)
        generate_code(dest_path,problems,model,tokenizer, model_type,augmented_data)
    elif augmentation_type == "no_rag":
        generate_code(dest_path,problems,model,tokenizer, model_type)
    