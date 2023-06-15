# EnCoSum: Enhanced Semantic Features for Multi-scale Multi-modal Source Code Summarization
The source code and datasets for the EnCoSum.
# Datasets
EnCoSum includes Java and Python datasets. If you want to train the model, you must download the datasets.
## Java dataset
You can get it on: https://github.com/xing-hu/TL-CodeSum
## Python dataset
You can get it on: https://github.com/EdinburghNLP/code-docstring-corpus
## CodeXGLUE dataset
You can get it on: https://github.com/microsoft/CodeXGLUE/tree/main
# Data process
EnCoSum uses the [JDK](http://www.eclipse.org/jdt/) compiler to parse java methods as ASTs, and the [Treelib](https://treelib.readthedocs.io/en/latest/) toolkit to prase Python functions as ASTs.  
## Get AST
In datapre file, the `get_java_ast.py` generates ASTs for the Java dataset and `get_python_ast.py` generates ASTs for Python datasets. You can run the following command：
```
python3 source.code ast.json
```
### Get E-AST
The file `get_adj.py` gets ASTs with added control flow and data flow.

### BERT embedding
You can use BERT method to embed AST nodes, detail in [here](https://github.com/hanxiao/bert-as-service)  
```
pip install bert-serving-server  
pip install bert-serving-client
```
# Train-Test
In Model file, the `main.py` enables to train the model. 
Train and test model:  
```
Python3 main.py
```
The nlg-eval can be set up in the following way, detail in [here](https://github.com/Maluuba/nlg-eval).  
Install Java 1.8.0 (or higher).  
Install the Python dependencies, run:
```
pip install git+https://github.com/Maluuba/nlg-eval.git@master
```
# Requirements
If you want to run the model, you will need to install the following packages.  
```
pytorch 1.7.1  
bert-serving-client 1.10.0  
bert-serving-server 1.10.0  
javalang 0.13.0  
nltk 3.5  
networkx 2.5  
scipy 1.1.0  
treelib 1.6.1
```
