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

### PPI (H.R=0.899)
| Method | AUC | AUPR | TIME* | Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|:-------|:---:|:---:|:---:|
| GCN | 80.10 | 72.08 | 0.0140 | ARMAGNN | 79.42 | <u>78.98</u> | 0.0103 |
| GAT | 77.77 | 69.29 | 0.0192 | TAGCN | 82.58 | 78.50 | 0.0108 |
| GATv2 | 80.64 | 73.17 | 0.0495 | PMLP | 65.80 | 47.86 | **0.0023** |
| ChebNet | 81.14 | 75.62 | 0.0145 | AGNN | 80.61 | 72.56 | 0.0161 |
| JKNet | 81.22 | 75.15 | 0.0195 | EMOGI | 81.92 | 75.72 | 0.0150 |
| MTGCN | <u>82.88</u> | 78.75 | 0.9344 | CGMega | 80.49 | 76.16 | 0.0214 |
| **CGMap** | **87.07** | **83.95** | <u>0.0065</u> | | | | |

### PathNet (H.R=0.876)
| Method | AUC | AUPR | TIME* | Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|:-------|:---:|:---:|:---:|
| GCN | 79.64 | 76.83 | 0.0059 | ARMAGNN | 79.98 | 82.12 | 0.0072 |
| GAT | 74.95 | 71.71 | 0.0113 | TAGCN | 84.39 | 82.68 | 0.0079 |
| GATv2 | 78.86 | 73.42 | 0.0197 | PMLP | 55.86 | 49.78 | **0.0023** |
| ChebNet | 82.37 | 81.48 | 0.0098 | AGNN | 78.50 | 70.21 | 0.0082 |
| JKNet | 80.04 | 75.24 | 0.0108 | EMOGI | 82.48 | 81.64 | 0.0980 |
| MTGCN | <u>84.43</u> | <u>82.86</u> | 0.2980 | CGMega | 80.23 | 78.51 | 0.0157 |
| **CGMap** | **86.01** | **85.23** | <u>0.0053</u> | | | | |

### GGNet (H.R=0.943)
| Method | AUC | AUPR | TIME* | Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|:-------|:---:|:---:|:---:|
| GCN | 61.17 | 50.46 | 0.0238 | ARMAGNN | 75.01 | 73.79 | 0.0172 |
| GAT | 60.20 | 47.57 | 0.0342 | TAGCN | <u>81.34</u> | <u>75.65</u> | 0.0206 |
| GATv2 | 68.37 | 57.16 | 0.0994 | PMLP | 57.44 | 45.97 | **0.0023** |
| ChebNet | 78.28 | 71.50 | 0.0255 | AGNN | 70.35 | 58.82 | 0.0300 |
| JKNet | 64.84 | 55.85 | 0.0357 | EMOGI | 78.73 | 72.69 | 0.0258 |
| MTGCN | 81.18 | 73.70 | 2.0766 | CGMega | 78.29 | 71.63 | 0.0205 |
| **CGMap** | **85.68** | **81.15** | <u>0.0057</u> | | | | |



## Overall Results on PPI, PathNet and GGNet in 5-CV Test

**Bold** indicates the best performance and <u>underlining</u> indicates 2<sup>nd</sup> best

<details>
<summary>📊 PPI Results (H.R=0.899) - Click to expand</summary>

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 80.10 | 72.08 | 0.0140 |
| ... | ... | ... | ... |
| **CGMap** | **87.07** | **83.95** | <u>0.0065</u> |

</details>

<details>
<summary>📊 PathNet Results (H.R=0.876) - Click to expand</summary>

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 79.64 | 76.83 | 0.0059 |
| ... | ... | ... | ... |
| **CGMap** | **86.01** | **85.23** | <u>0.0053</u> |

</details>

<details>
<summary>📊 GGNet Results (H.R=0.943) - Click to expand</summary>

| Method | AUC | AUPR | TIME* |
|:-------|:---:|:---:|:---:|
| GCN | 61.17 | 50.46 | 0.0238 |
| ... | ... | ... | ... |
| **CGMap** | **85.68** | **81.15** | <u>0.0057</u> |

</details>

*Record the training time (s) for each epoch.*
