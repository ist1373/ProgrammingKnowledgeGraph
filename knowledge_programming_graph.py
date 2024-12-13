from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import numpy as np
import pandas as pd
import uuid     
from function_analyzer import * 
from function_enhancer import *    


def get_embeddings_from_dict(data,embedder_tokenizer,embedder_model):
    strings = []

    # Extract all strings from the dictionary
    for key, value in data.items():
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, list):
            strings.extend([item for item in value if isinstance(item, str)])

    embeddings = get_embeddings(strings,embedder_tokenizer,embedder_model)

    augmented_data = {}
    index = 0
    for key, value in data.items():
        if isinstance(value, str):
            augmented_data[key] = {"value":value,"embedding":embeddings[index]}
            index += 1
        elif isinstance(value, list):
            augmented_data[key] = {"value":value,"embedding":[embeddings[index + i] for i, item in enumerate(value) if isinstance(item, str)]}
            index += len([item for item in value if isinstance(item, str)])
    return augmented_data



def get_embeddings(strings,tokenizer,model):
   
    inputs = tokenizer(strings, padding=True, truncation=True, return_tensors="pt").to("cuda")
    embedding = model(**inputs)
    return embedding.cpu().detach()
    

def cosine_similarity(embeddings):
    """
    Calculate the cosine similarity between each pair of embeddings in a list.

    :param embeddings: List of embeddings (each embedding is a numpy array or list)
    :return: A 2D numpy array containing cosine similarity scores
    """
    embeddings = np.array(embeddings)
    similarity_matrix = np.zeros((embeddings.shape[0], embeddings.shape[0]))

    for i in range(len(embeddings)):
        for j in range(len(embeddings)):
            dot_product = np.dot(embeddings[i], embeddings[j])
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            similarity_matrix[i][j] = dot_product / (norm_i * norm_j)
    
    return similarity_matrix

import uuid        
class Node:
    def __init__(self,uuid,node_type,content,embedding):
        self.uuid = uuid
        self.node_type = node_type
        self.content = content
        self.embedding = embedding
        
    def to_dict(self):
        return {
            'uuid': self.uuid,
            'node_type': self.node_type,
            'content': self.content,
            'embedding':self.embedding.tolist()
        }
    
    def __str__(self):
        return f"uuid:{self.uuid}, node_type:{self.node_type}, content:{self.content}"

    @staticmethod
    def get_nodes_by_type(nodes,node_type):
        specific_nodes = []
        for node in nodes:
            if node.node_type == node_type:
                specific_nodes.append(node)
        return specific_nodes
        
class Relation:
    def __init__(self, uuid_from, uuid_to, relation_type):
        self.uuid_from = uuid_from
        self.uuid_to = uuid_to
        self.relation_type = relation_type


    def to_dict(self):
        return {
            'uuid_from': self.uuid_from,
            'uuid_to': self.uuid_to,
            'relation_type': self.relation_type,
        }
    def __str__(self):
        return f"uuid_from:{self.uuid_from}, uuid_to:{self.uuid_to}, relation_type:{self.relation_type}"

        
class GraphMacker:
    def __init__(self):
        pass

    def generate_uuid(self):
        return str(uuid.uuid4())

    def generate_nodes(self,func_info):
        node_list = []
        for key, value in func_info.items():
            if key == "func_name":
                node_list.append(Node(uuid =self.generate_uuid(), node_type = "func_name", content = value["value"] ,embedding = value["embedding"]))
            elif key == "implementation":
                node_list.append(Node(uuid =self.generate_uuid(), node_type = "implementation", content = value["value"] ,embedding = value["embedding"]))     
            elif key == "code_blocks":
                for i, code_block in enumerate(value["value"]):
                    node_list.append(Node(uuid =self.generate_uuid(), node_type = "code_block", content = value["value"][i] ,embedding = value["embedding"][i]))                   
        return node_list
        
    def generate_relations(self,nodes,siblings,parents):
        relation_list = []
        func_name_node = Node.get_nodes_by_type(nodes,"func_name")[0]
        implementation_node = Node.get_nodes_by_type(nodes,"implementation")[0]
        code_block_nodes = Node.get_nodes_by_type(nodes,"code_block")

        if func_name_node and implementation_node:
            relation_list.append(Relation(uuid_from = func_name_node.uuid, uuid_to = implementation_node.uuid, relation_type = "implementation"))
        if implementation_node and code_block_nodes:
            relation_list.append(Relation(uuid_from = implementation_node.uuid, uuid_to = code_block_nodes[0].uuid, relation_type = "child"))
        
        for key,value in parents.items():
            relation_list.append(Relation(uuid_from = code_block_nodes[value].uuid, uuid_to = code_block_nodes[int(key)].uuid, relation_type = "child"))        
        
        return relation_list
    
    def create_semantic_relations(self,nodes,node_type):
        subset_nodes = Node.get_nodes_by_type(nodes,node_type)
        subset_embeddings = []
        for node in subset_nodes:
            subset_embeddings.append(node.embedding)
            
        similarity = cosine_similarity(subset_embeddings)
        # Define the threshold
        threshold = 0.8
        
        # Find the indices where the similarity is greater than the threshold
        indices = np.argwhere(similarity > threshold)
        
        # Filter out self-similarities (diagonal elements)
        filtered_indices = [(i, j) for i, j in indices if i != j]
        
        # filter repetitive elements
        unique_relations = []
        for i,j in filtered_indices:
            if (i,j) not in unique_relations and (j,i) not in unique_relations:
                unique_relations.append((i,j))
        
        relation_list = []
        for i,j in unique_relations:
            relation_list.append(Relation(uuid_from = subset_nodes[i].uuid, uuid_to = subset_nodes[j].uuid, relation_type = similarity[i,j]))
    
        return relation_list
    
    @staticmethod
    def save_nodes(nodes,file_path):
        node_list_dict = [node.to_dict() for node in nodes]
        
        with open(file_path, 'w') as json_file:
            json.dump(node_list_dict, json_file, indent=4)
        
    @staticmethod
    def save_relatios(relations,file_path):
        relation_list_dict = [relation.to_dict() for relation in relations]
        
        with open(file_path, 'w') as json_file:
            json.dump(relation_list_dict, json_file, indent=4)


    



def main():
    model_path = "PRETRAINED_MODEL PATH"
    df = pd.read_csv("./datasets/python_alpaca.csv")

    embedder_tokenizer = AutoTokenizer.from_pretrained(model_path)
    embedder_model = AutoModelForCausalLM.from_pretrained(model_path)
    analyzer = FunctionAnalyzer()
    python_codes = df["output"].apply(analyzer.extract_python_code).apply(analyzer.get_function_blocks)
    python_codes = np.concatenate(python_codes)

    analyzer = FunctionAnalyzer()
    enhancer = FunctionEnhancer(embedder_model,embedder_tokenizer,analyzer)
    graph_macker = GraphMacker()
    for j in range(0,len(python_codes)-1000,1000):
        all_nodes = []
        all_relations = []
        for i,code in enumerate(python_codes[j:j+155]):
            func_name = analyzer.get_function_name(code)
            pure_code = analyzer.remove_docstring_from_function(code)
            code_blocks,block_info = analyzer.get_code_blocks(pure_code) 
            docstr_formatted_code,doc_string = enhancer.generate_docstring(code)

            if docstr_formatted_code:
                # enhanced_code,comments = enhancer.comment_formatter(docstr_formatted_code,[item[0] for item in block_info])
                siblings,parents = analyzer.extract_relations(block_info)
                
                data = {
                    'func_name': func_name,
                    'implementation': docstr_formatted_code,
                    'code_blocks':code_blocks,
                }
                # 'comments':comments
                func_info = get_embeddings_from_dict(data,embedder_tokenizer,embedder_model)
                
                nodes = graph_macker.generate_nodes(func_info)
                relations = graph_macker.generate_relations(nodes,siblings,parents)
            
                all_nodes.extend(nodes)
                all_relations.extend(relations)
                print(f"code {i+1} has been processed.")
            else:
                print(f"ignore item:{i+1}:\n" + code)
        
        GraphMacker.save_nodes(all_nodes,f"canon_nodes_{j}.json")
        GraphMacker.save_relatios(all_relations, f"canon_relations_{j}.json")

if __name__ == "__main__":
    main()