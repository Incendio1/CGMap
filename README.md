# Advancing interpretable cancer gene identification via width scaling graph representation learning

>  **The implementation of CGMap & data accompanying the manuscript submitted to _Nature Machine Intelligence_.**   

---

## Abstract
This project is the open code for CGMap, where each gene is represented as a node. It identifies cancer genes through a width-oriented parallel propagation and a path diffusion mechanism. CGMap demonstrates outstanding performance, stability, and resistance to over-smoothing, while maintaining generalization capabilities under data bias, noise, and structural incompleteness. Its inherent interpretability supports model decisions by elucidating regulatory pathways and revealing topologically distant gene dependencies.

<img width="2117" height="1599" alt="image" src="https://github.com/user-attachments/assets/b91dacd9-44f4-414c-ae57-80401ac5bb2b" />



## Requirements and Installation  

### Requirements
```bash
Python Version: 3.8
PyTorch Version: 1.12.1
CUDA Version: 11.3
Numpy Version: 1.24.3
Pandas Version: 2.0.2
Networkx Version: 3.1
Pytroch Geometric Version: 2.3.1
scikit-learn Version: 1.2.2

```

### Installation
The following PyG versions are recommended:
```bash
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-sparse -f https://pytorch-f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.12.1+cu113.html
pip install pytorch-geometric==2.3.1
```
## Additional datasets
Network of Cancer Genes (NCG 7.2):
http://network-cancer-genes.org/
