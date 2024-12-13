from function_analyzer import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import BitsAndBytesConfig


class FunctionEnhancer():
  def __init__(self,model,tokenizer, analyzer:FunctionAnalyzer):
    self.tokenizer = tokenizer
    self.analyzer = analyzer
    model_path = "PRETRAINED_MODEL PATH"
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = AutoModelForCausalLM.from_pretrained(model_path)

  def _generate(self, fim_code):
    prompt = f'''
    <|fim_prefix|>
    {fim_code}
    <|fim_middle|>'''
    
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    outputs = self.model.generate(**inputs, max_new_tokens=256,pad_token_id=self.tokenizer.eos_token_id)
    string_output = self.tokenizer.decode(outputs[0][prompt_len:])
    end_token = "<|file_separator|><eos>"
    if end_token in string_output:
      string_output = string_output.replace(end_token, '')
    return string_output

  def generate_docstring(self,code):
    try:
        func_line = self.analyzer.get_function_def_line(code)
        if func_line == -1:
          return None,None
        if not self.analyzer.get_docstring(code):
          lines = code.splitlines()
          
          # line = lines[func_line]
          # line = line.lstrip()
            
          line = lines[func_line+1]
          lspace = len(line) - len(line.lstrip())
          lines.insert(func_line+1," "*lspace + '"""<|fim_suffix|>"""')
          fim_code = '\n'.join(lines)
          docstring = self._generate(fim_code)
          if docstring:  
              fill_code = fim_code.replace('<|fim_suffix|>',docstring)
          else:
              fill_code = fim_code.replace('<|fim_suffix|>','')
        else:
            return code,self.analyzer.get_docstring(code)
        return fill_code,docstring
    except:
        return code, ""

  def comment_formatter(self,code,commenting_lines):
      comments = []
      for line_num in commenting_lines:
        lines = code.splitlines()
        line = lines[line_num]
        lspace = len(line) - len(line.lstrip())
        lines.insert(line_num," "*lspace + '#<|fim_suffix|>.')
        temp_code = '\n'.join(lines)
        comment = self._generate(temp_code)
        comments.append(comment.replace("\n",""))
      
      fixed_margin_index = []
      for i,k in enumerate(commenting_lines):
        fixed_margin_index.append(i+k)
    
      lines = code.splitlines()
      for i,line_num in enumerate(commenting_lines):
        line = lines[fixed_margin_index[i]]
        lspace = len(line) - len(line.lstrip())
        lines.insert(fixed_margin_index[i]," "*lspace + '#' + comments[i])
      
      code = '\n'.join(lines)
      return code,comments

  def assertion_generation(self,code):
      
      temp_code = code + "\n#assertions:\nassert <|fim_suffix|>"
      assertions = "assert "+self._generate(temp_code)
      return assertions

  def comment_formatter_v1(self,code,commenting_lines):
      comments = []
      for line_start,line_end in commenting_lines:
        lines = code.splitlines()
        blocks = lines[line_start:line_end+1]
        line = lines[line_start]
        lspace = len(line) - len(line.lstrip())
        blocks.insert(0," "*lspace + '#<|fim_suffix|>')
        temp_code = '\n'.join(blocks)
        comment = self._generate(temp_code)
        comments.append(comment.replace("\n",""))
      return comments

