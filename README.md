# CS-433 ML4Science project 2: Personalized Federated Image classification using Weight Erosion
Yuan Vincent, Gengler Damien, Beaussart Martin

## Introduction

This  project  took  place  in  Fall  2020  for  the  course  of machine  learning  (CS-433).  It  is  one  of  the  several  interdis-ciplinary ML for Scienceprojects.

During  4  weeks,  we  worked  with  the  intelligent  GlobalHealth  (iGH)  group,  a  project  dedicated  to  machine  learningapplications for healthcare launched by the machine learning and Optimization Laboratory (MLO) at EPFL.

In this project, we adapted the Weight Erosion scheme develpoed in the iGH from a logistic regression model to a neural network ones. Then, we benchmarked this new model on different distributions of the MNIST dataset, and plotted the results.

Special thanks to Felix Grimberg and Mary-Anne Hartley for their help and support throughout the project ! 

## How to get started

### Recommended: Anaconda GUI and Google Collab

We recommand to use [Anaconda](https://docs.anaconda.com/anaconda/install/) with the GUI to import the libraries. Otherwise, you can use [Google Collab](https://colab.research.google.com) if you don't want to bother with the libraries import, and not let the notebook run for hours in your local machine.

The list of dependencies is as follow: 
```
numpy == 1.19.1
pytorch = 1.4.0
matplotlib == 3.3.1
tqdm == 4.54.1
torchvision == 0.5.0
```

After installing the dependencies, you can just let the notebook run. The MNIST dataset will be automatically downloaded

### Using miniconda and command line

This is not recommanded, as the libraries take a lot of places and it's easier to run on a collab. However, if you want want to run on your local pc, here are the steps:

Creating the conda environment

```
conda create --name weight_erosion python=3.8
conda activate weight_erosion
```

Importing libraries (numpy, matplotlib, tqdm)

```
conda install --file requirements.txt
```

The command line for installing PyTorch will be different depending on your OS. To install it, you will have to go to [Pytorch website](https://pytorch.org/) to get the correct command line.

Then, after installing PyTorch, you can finally install torchvision

```
conda install -c pytorch torchvision
```

Installing jupyter lab to open the .ipynb file:

```
conda install -c conda-forge jupyterlab
```

Launching the jupyter lab
```
jupyter lab
```

Then, you have to go to original_framework.ipynb

### Important notes on PyTorch

We created this notebook such that it needs to run on a GPU. Thus, you'll need to have a machine that has a GPU and that is compatible with conda. On a more personal note, team members used more google collab than local running, as the computations were heavy. 

## Files

File tree:

```
├── Weight_Erosion_Project.ipynb    : Main script
├── README.md                       : The README guideline and explanation for our project.
├── original_framework.ipynb        : Framework implementing the Federated Average scheme, which was our initial template
├── project_report.pdf              : Report of the project
├── requirement.txt                 : Dependencies
├── report_annex_all_plots.pdf      : plots of our different benchmarking
|
├── data/MNIST                      : MNIST data 
```

### original_framework.ipynb

This python notebook contains the framework on which we started our work. This framework implement the federated average scheme using pytorch. Even though we don't use this file for our project, we decided to include it in our git repo for information purposes

### project_report.pdf 

This is our project report, where we explain more in depth our reasoning, the different steps we went through, and our final results.

### report_annex_all_plots.pdf

This PDF file contains the plots of all of our benchmarkings. It contains the plots for all distributions (A to G), and for the 4 numbers of agents (10 20 50 100).

The plots contain the evolution over the 30 rounds of the test accuracy, as well as the mean of the weights in the WE scheme.

### weight_erosion.ipynb

This python notebook contains our work. Each cell has a different purpose, explained below:

1. **Imports**: the first cell contains all of the necessary imports
2. **Model**: the second cell contains the neural network model, the gradient stocker abstraction, and the utility function
3. **Weight Erosion**: the third cell implements Weight Erosion scheme
4. **Federated average**: the fourth cell implements the federated average scheme
5. **Local training**: the fifth cell implements the local training baseline
6. **Distributions**: the sixth cell implements all of our distributions that we used during this project. Even though we only use the function _get_non_iid_loader_distribution_ in our final version of the notebook, we found interesting to put all of the different distributions functions we used during this project.
7. **Benchmarking**: this cell benchmarks these 3 baselines on MNIST dataset using the neural network model, and stores the data in pickle files in the folder generated/pickles.
8. **Plot**: This cell plots the distribution we obtained on the cell 7
9. **Results**: This cell prints all of the results we obtained in our .pickle files

We decided to benchmark 7 distributions in our final report, and final code. However, as the computation runtime is extremely long (~10h on google collabs) and the resulting cell is not readable, with too much print, we decided to run for only the distribution B in our cell 7. 

Note that the notebook is pre-ran. The cells output can be quite long. The reader is invited to look at the code on a Jupyter instane as it allows to fold these outputs and get a better understanding of the code.

## Sustainability

In a sustainability optic, we used the python module 'Cumulator' to keep track of our Benchmark's carbon footprint. A run of the Benchmark for our 7 distribution yields a carbon footprint of 707 gCO2eq. You can find out more about Cumulator here : 