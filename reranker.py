import json 
import numpy as np
import pickle
import pandas as pd
import ast
import astor
import signal
from human_eval.data import write_jsonl, read_problems
import voyageai
import argparse


class FirstFunctionExtractor(ast.NodeTransformer):
    def __init__(self):
        self.first_function_found = False  # To keep track of whether we've found the first function
    
    def visit_FunctionDef(self, node):
        if not self.first_function_found:
            # Keep the first function and mark that we found it
            self.first_function_found = True
            # Visit the body of the function and remove its docstrings
            node.body = [stmt for stmt in node.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Str))]
            self.generic_visit(node)
            return node
        # Ignore any further functions
        return None
    
    def visit_Module(self, node):
        # Keep only the first function in the module
        node.body = [stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef) and not self.first_function_found]
        self.generic_visit(node)
        return node
    
# Define a custom exception for timeout
class TimeoutException(Exception):
    pass

# Define a handler that raises TimeoutException when the time limit is reached
def timeout_handler(signum, frame):
    raise TimeoutException

def read_from_jsonl(file_path):
    # Read the .jsonl file
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse each line as a JSON object and append to the list
            data.append(json.loads(line.strip()))
    
    # Now, `data` contains all the JSON objects from the file
    return data

def write_to_jsonl(data,file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def save_to_file(object,file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(object, file)

def load_from_file(file_name):
    with open(file_name, 'rb') as file:
        loaded_data = pickle.load(file)
        return loaded_data
    
def cosine_similarity(embedding1, embedding2):
    # Ensure the embeddings are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Compute the dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # Compute the magnitudes (norms) of the embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Calculate cosine similarity
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Avoid division by zero if one of the embeddings is all zeros
    cosine_sim = dot_product / (norm1 * norm2)
    
    return cosine_sim    



def remove_comments_and_docstrings(code: str) -> str:
    # Parse the code into an AST
    signal.signal(signal.SIGALRM, timeout_handler)

    try:
        signal.alarm(5)
        exec(code)
        tree = ast.parse(code)
        
        # Create a transformer instance and apply it to the AST
        extractor = FirstFunctionExtractor()
        transformed_tree = extractor.visit(tree)
        
        # Convert the transformed AST back to code
        if transformed_tree:
            cleaned_code = astor.to_source(transformed_tree)
            
            return cleaned_code.strip()
    except:        
        return "Wrong code!"
    finally:
        signal.alarm(0)


def count_correct_answers(norag,bm25rag,bwrag,fwrag):
    norag_count = 0
    bm25_count = 0
    bwrag_count = 0
    fwrag_count = 0
    ideal = 0
    for i in range(len(norag)):
        if norag[i]["passed"] or bm25rag[i]["passed"] or bwrag[i]["passed"] or fwrag[i]["passed"]:
            ideal+=1
        if norag[i]["passed"]:
            norag_count+=1
        if bm25rag[i]["passed"]:
            bm25_count+=1
        if bwrag[i]["passed"]:
            bwrag_count+=1
        if fwrag[i]["passed"]:
            fwrag_count+=1 
    return {'norag_count':norag_count,'bm25_count':bm25_count,'bwrag_count':bwrag_count,'fwrag_count':fwrag_count,'ideal':ideal}

def init_voyageai_embedder():
    voyageai.api_key = "pa-lpHvggxAmQX_QTSipVOhn7qC5Ue_9XRxT_RGwCEFRmE"
    vo = voyageai.Client()
    return vo


def rerank_one_solution(query, list_of_imp,vo):
    
    query_embedding = vo.embed([query], model="voyage-code-2").embeddings[0]
    codes = []
    for sol in list_of_imp:
        codes.append(remove_comments_and_docstrings(sol['completion']))
       
    solution_embeddings = vo.embed(codes, model="voyage-code-2").embeddings

    similarities = np.array([cosine_similarity(query_embedding,emb) for emb in solution_embeddings])
    # print(f"solution:\n{list_of_imp[similarities.argmax()]}")
    return list_of_imp[similarities.argmax()]


def calculate_reranked_correct_answers(reranked_solutions):
    corrected_answers = 0
    for sol in reranked_solutions:
        if sol['passed']:
            corrected_answers+=1
    return corrected_answers
    

def main(output_path, evaluation, norag_path, bwrag_path, fwrag_path, bm25rag_path):
    embedder_model = init_voyageai_embedder()

    norag = read_from_jsonl(norag_path)
    bwrag = read_from_jsonl(bwrag_path)
    fwrag = read_from_jsonl(fwrag_path)
    bm25rag = read_from_jsonl(bm25rag_path)

    print(count_correct_answers(norag,bwrag,fwrag,bm25rag))

    if evaluation == "human_eval":
        mbpp_problems = pd.read_csv("mbpp.csv")
        reranked_solutions = []
        for i,problem in enumerate(mbpp_problems["text"].values):
            print(f"problem {i} is processing.")
            list_of_imp = [norag[i],bwrag[i],fwrag[i],bm25rag[i]]
            
            query = problem
            reranked_solutions.append(rerank_one_solution(query,list_of_imp,embedder_model))

    elif evaluation == "mbpp":
        problems = read_problems()
        reranked_solutions = []
        for i,problem in enumerate(problems):
            print(f"problem {problem} is processing.")
            list_of_imp = [norag[i],bwrag[i],fwrag[i],bm25rag[i]]
            query = problems[problem]["prompt"]
            reranked_solutions.append(rerank_one_solution(query,list_of_imp,embedder_model))

    correct_answers = calculate_reranked_correct_answers(reranked_solutions)
    print(f"correct answers:{correct_answers}")
    write_to_jsonl(reranked_solutions,output_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process code generation evaluations using different RAG models.')

    parser.add_argument('--output_path', type=str, default="starcoder-7b-reranked.jsonl", help='Path to save the output file.')
    parser.add_argument('--evaluation', type=str, choices=['human_eval', 'mbpp'], default="human_eval", help='Evaluation type (human_eval or mbpp).')
    parser.add_argument('--norag_path', type=str, required=True, help='Path to the NoRAG JSONL file.')
    parser.add_argument('--bwrag_path', type=str, required=True, help='Path to the Block-level RAG JSONL file.')
    parser.add_argument('--fwrag_path', type=str, required=True, help='Path to the Function-level RAG JSONL file.')
    parser.add_argument('--bm25rag_path', type=str, required=True, help='Path to the BM25 RAG JSONL file.')

    args = parser.parse_args()

    main(args.output_path, args.evaluation, args.norag_path, args.bwrag_path, args.fwrag_path, args.bm25rag_path)