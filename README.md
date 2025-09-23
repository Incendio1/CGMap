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

## Running the Project
Since CGMap employs a width-oriented parallel propagation algorithm, it can locate gene associations at any distance scale in a single step, and this process can be preprocessed. Therefore, we directly provide preprocessed OPP results at a maximum distance scale of 10, available for download at https://zenodo.org/records/17178713. After downloading, simply copy the contents into the OPP_info directory. If you prefer not to download, CGMap can also automatically process all gene associations less than or equal to the value specified by the hyperparameter `OPP_layer`. Our dataset can be viewed in the file "data". Beyond AUC and AUPR metrics, the file "screening" presents the screening and ranking results of baselines specifically designed for cancer genes.

Execute the project by running the following command and configuration:
```bash
python run_CGMap.py --model "CGMap" --device 0 --dataset PPNet --agg sum --theta 0.9 --alpha 0.45 --gamma 6.0
```
```bash
python run_CGMap.py --model "CGMap" --device 0 --dataset GGNet --agg sum --i_w 0.51 0.5 0.1 1.1 --lr 0.00046 --dropout 0.49 --epoch 2500 --hidden 101 --w_decay 3.7e-06 
```
```bash
python run_CGMap.py --model "CGMap" --device 0 --dataset PathNet --agg sum --i_w 0.2 0.1 0.006 2.5 --lr 0.00072 --dropout 0.61 --alpha 0.37 --gamma 5 --epoch 1900 --w_decay 2.7e-07
```

## Additional datasets
Network of Cancer Genes (NCG 7.2):
http://network-cancer-genes.org/

Pytorch Geometirc benchmarks:
https://github.com/pyg-team/pytorch_geometric

Optuna v4/v5 framework:
https://github.com/optuna/optuna

an example of building a cancer-specific dataset using `customize_dataset.py`

## Overall Results on PPI, PathNet and GGNet in 5-CV Test

**Bold** indicates the best performance and <u>underlining</u> indicates 2<sup>nd</sup> best

### PPI (H.R=0.899)

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 80.10 | 72.08 | 0.0140 |
| GAT | 77.77 | 69.29 | 0.0192 |
| ... | ... | ... | ... |
| **CGMap** | **87.07** | **83.95** | <u>0.0065</u> |

### PathNet (H.R=0.876)

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 79.64 | 76.83 | 0.0059 |
| GAT | 74.95 | 71.71 | 0.0113 |
| ... | ... | ... | ... |
| **CGMap** | **86.01** | **85.23** | <u>0.0053</u> |

### GGNet (H.R=0.943)

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 61.17 | 50.46 | 0.0238 |
| GAT | 60.20 | 47.57 | 0.0342 |
| ... | ... | ... | ... |
| **CGMap** | **85.68** | **81.15** | <u>0.0057</u> |

*Record the training time (s) for each epoch.*
