# Context-Augmented Code Generation Using Programming Knowledge Graphs

This repository accompanies the research paper ["Context-Augmented Code Generation Using Programming Knowledge Graphs"](https://arxiv.org/pdf/2410.18251) and provides the implementation for a novel framework leveraging a Programming Knowledge Graph (PKG) to enhance code generation using Retrieval-Augmented Generation (RAG).

Overview
Large Language Models (LLMs) and Code-LLMs (CLLMs) excel in code generation tasks but struggle with complex problems due to irrelevant or incomplete contexts. This project introduces a PKG-based framework to address these challenges by:

Fine-grained Code Retrieval: Using tree-pruning techniques to focus on the most relevant code segments while minimizing irrelevant context.
Re-ranking Mechanism: Reducing hallucinations by selectively integrating retrieved context with non-RAG solutions.
Fill-in-the-Middle (FIM) Enhancement: Automatically augmenting code with comments and docstrings for better generation accuracy.
Key Contributions
PKG-Based Retrieval: Semantic representation and retrieval of code at fine granularity levels (block-wise and function-wise).
Tree Pruning: Improved retrieval precision by pruning irrelevant branches in the knowledge graph.
Re-ranking Solutions: Selecting robust solutions to reduce hallucination in generated code.
FIM Enhancer Module: Automatic augmentation of code with relevant metadata.
Our method achieves:

Up to 20% improvement in pass@1 accuracy.
Outperformance of state-of-the-art models by 34% on MBPP.
