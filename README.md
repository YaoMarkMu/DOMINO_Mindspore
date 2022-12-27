# Mindspore Code for DOMINO
## Decomposed Mutual Information Optimization for Generalized Context in Meta-Reinforcement Learning (Neurips 2022)

The whole framework is shown as follow:
![DOMINO Framework](pngs/framework.png)

## Method

This paper addresses the multi-confounded challenge by decomposed mutual information optimization for context learning, which explicitly learns a disentangled context to maximize the mutual information between the context and historical trajectories while minimizing the state transition prediction error. 

- [Project webpage](https://sites.google.com/view/dominorl/)

## Instructions

Introduction to Mindspore


MindSpore is a deep learning framework in all scenarios, aiming to achieve easy development, efficient execution, and all-scenario coverage.
Easy development features user-friendly APIs and low debugging difficulty. Efficient execution is reflected in computing, data preprocessing, and distributed training. All-scenario coverage means that the framework supports cloud, edge, and device scenarios.


Install required packages with below commands:

```
conda create -c conda-forge -n domino_mindspore python=3.7.5 -y
conda activate domino_mindspore
conda install mindspore-gpu=1.8.1 cudatoolkit=11.1 -c mindspore -c conda-forge
pip install -r requirements.txt
```

Train the agents:

```
python -m run_scripts.run_domino --dataset [hopper,ant,halfcheetah,cripple_ant,cripple_halfcheetah] --normalize_flag
```


Test the agents:

```
python domino_ms_infer.py
```