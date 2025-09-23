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
python run_CGMap.py --model "CGMap" --device 0 --dataset PPNet --agg sum --theta 0.9 --alpha 0.45 --gamma 6.0
```
```bash
python run_CGMap.py --model "CGMap" --device 0 --dataset PPNet --agg sum --theta 0.9 --alpha 0.45 --gamma 6.0
```
```bash
python run_CGMap.py --model "CGMap" --device 0 --dataset PPNet --agg sum --theta 0.9 --alpha 0.45 --gamma 6.0
```

## Running the Project
Since CGMap employs a width-oriented parallel propagation algorithm, it can locate gene associations at any distance scale in a single step, and this process can be preprocessed. Therefore, we directly provide preprocessed OPP results at a maximum distance scale of 10, available for download at https://zenodo.org/records/17178713. After downloading, simply copy the contents into the OPP_info directory. If you prefer not to download, CGMap can also automatically process all gene associations less than or equal to the value specified by the hyperparameter OPP_layer. Our dataset can be viewed in the file "data"

Then, execute the project by running the following command and configuration:
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

## Additional datasets
Network of Cancer Genes (NCG 7.2):
http://network-cancer-genes.org/
