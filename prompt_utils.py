CODE_LLAMA_7B="codellama_7b"
CODE_LLAMA_13B="codellama_13b"
CODE_LLAMA_34B="codellama_34b"
START_CODER2_7B="starcoder2_7b"
LLAMA3_8B="llama3_8b"
DEEP_SEEK_CODER_7B="deepseekcoder_7b"

def codellama_prompt(problem,augmented_data=None):
    if augmented_data:
        prompt = f"""[INST] You are a python programmer. Solve the following problem:\n{problem} \n\nThe following code might be helpful:\n{augmented_data}\nIf helper section is useful, integrate their logic directly into the body of the main function, otherwise just ignore them. You MUST write your solution between [PYTHON] and [/PYTHON]. Your solution MUST be executable.[/INST]"""
        return prompt
    else:
        prompt = f"""[INST] You are a python programmer. Solve the following problem:\n{problem} \n\nPlease write the python solution inside [PYTHON] and [/PYTHON] tags.\n[/INST]"
        """
        return prompt
        
def starcoder_prompt(problem,augmented_data=None):
    #[INST] You are a python expert programmer. {problem} \n\n The following code might be helpful:\n{augmented_data}\n. Make sure that the code SHOULD be Executable, self-contain. do NOT call helper codes, if you want to use them, use them in the body function. The code MUST pass all the following test cases:\n{test_string}\n[/INST]
    if augmented_data:
        prompt = f"""### Instruction
        You are a python programmer. Solve the following problem:\n{problem} \n\n The following code might be helpful:\n{augmented_data}\n. If they are useful, integrate their logic directly into the body of the main function, otherwise just ignore them.\n
        ### Response
        """
        return prompt
    else:
        prompt = f"""### Instruction
        You are a python programmer. Solve the following problem:\n{problem} \n\n
        ### Response
        """
        return prompt

def deepseek_prompt(problem,augmented_data=None):
    if augmented_data:
        prompt = f"""[INST] You are a python programmer. Solve the following problem:\n{problem} \n\n The following code might be helpful:\n{augmented_data}\n.If they are useful, integrate their logic directly into the body of the main function, otherwise just ignore them.\n[/INST]"""
        return prompt
    else:
        prompt = f"""[INST] You are a python programmer.  Solve the following problem: \n {problem} \n\n[/INST]"""
        return prompt

def llama3_prompt(problem, augmented_data):
    if augmented_data:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a python programmer.<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Implement a python function to solve the following problem: \n {problem} \n\n The following helper code might be helpful:\n{augmented_data}.\nIf they are useful, integrate their logic directly into the body of the main function, otherwise just ignore them.\nThe solution should be executable. The code should pass all the following test cases. Please write the python solution inside [PYTHON] and [/PYTHON] tags\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        return prompt
    else:
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    
    You are a python programmer.<|eot_id|><|start_header_id|>user<|end_header_id|>
    
    Implement a python function to solve the following problem: \n {problem} \n\n. Please write the python solution inside [PYTHON] and [/PYTHON] tags.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """ 
        return prompt