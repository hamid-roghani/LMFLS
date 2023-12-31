# LMFLS
Local Multi-Factor Node Scoring and Label Selection-based Algorithm for Community Detection in Social Networks 

This code implements LMFLS algorithm, a fast and non-overlapping community detection algorithm in Python 3.10.

We have used neighboring list structure as the input file for algorithm. The execution of the LMFLS is so simple.it is only needed to write the name of the dataset to run the algorithm.

Everything is built to run the algorithm (such as pathes to folders and files) and it is only needed to put the extracted folders of: "datasets","groundtruth" in one folder with the source code.

To execute algorithm jupyter notebook or any other platform which runs python can be used. The main configurations of LMFLS are as follows:

# ---------------------- Configurations -------------------
dataset_name = "karate"# name of dataset
path = "datasets/" + dataset_name + ".txt" # path to dataset
iteration1 = 2        # number of iterations for label selection step
iteration2 = 2        # number of iterations for final label selection step
threshold = 0.9
merge_flag = 1        # merge_flag=0 -> do not merge ////  merge_flag=1 -> do merge
modularity_flag = 1   # 1 means calculate modularity. 0 means do not calculate modularity
NMI_flag = 1          # 1 means calculate NMI. 0 means do not calculate NMI
# ---------------------------------------------------------

Names of datasets are as follows and are available in "datasets" folder.
Datasets:

karate

Dolphins

Polbooks

Football

Netscience

power

ca_grqc

collaboration

ca_hepth

PGP

condmat_2003

condmat_2005

DBLP

Amazon

Youtube
