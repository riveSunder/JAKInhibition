# Estimating inhibition metrics for JAK kinases

![ inference example ](assets/inference_example.png)

## Summary

The objective of this project is to predict pKi (log dissocation constant [Ki](https://en.wikipedia.org/wiki/Enzyme_inhibitor#Quantitative_description) of small molecules on Janus-associated kinases JAK1, JAK2, JAK3, and TYK2. 

I used a two part approach: unsupervised training with a transformer on SMILES sequences, then training a bootstrap ensemble of multilayer perceptrons on the encoded features extracted by the pre-trained transformer. My notes on each part of the process can be found in the notebooks linked below. If you'd like to try out a deployment prototype first, check out the [notebook mockup](https://mybinder.org/v2/gh/riveSunder/JAKInhibition/HEAD?labpath=notebooks%2Fmockup.ipynb)

0. [Index](notebooks/index.ipynb)
1. [Exploratory Data Analysis](notebooks/eda.ipynb)
2. [Data Processing](notebook/data_processing.ipynb)
3. [Model Development](notebooks/model_development.ipynb)
4. [Model Evaluation](notebooks/model_evaluation.ipynb)
5. [Model Deployment](notebooks/model_deployment.ipynb) -> prototype on [mybinder](https://mybinder.org/v2/gh/riveSunder/JAKInhibition/HEAD?labpath=notebooks%2Fmockup.ipynb) or [local notebook](notebooks/mock.ipynb)
6. [Future Ideas](notebooks/future_ideas.ipynb)

