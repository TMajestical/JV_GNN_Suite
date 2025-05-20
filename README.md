This repo contains code of GNN Suite (GraphSAGE, GCN and GIN) in C++ : completely written from scratch - without any dependencies. Fully based on SOLID principles, the code can be extended to support

The Datasets are expected to be present in a directory Datasets/, now depending on where the main function is accordingly set the relative path to the Datasets directory to read datasets correctly. 

The datasets are obtained from the python code JV_Dataset2CSR.py to which -ds "Dataset Name" must be passed while running it from the terminal. The other commandline flags are specified in the code itself.

These (data) files are generated : "features.txt", "labels.txt", "train_indices.txt", "val_indices.txt" and

	a. "CSR.txt" and "index.txt" these files are the CSR representation of the dataset where index.txt contains the offsets. These files are expected to be present in the same directory as the code.
	
	b. "edgelist.txt" is the input COO format used by StarPlat's graph representation generation module.

For the GNN suite to work correctly, aforementioned files are required to be present at /Dataset/<DatasetName>/

To test the genreated code, the GNN_Suite_code, the generated training code, the main function and the corresponding header files like graph.hpp, graph_ompv2.hpp, atomicUtil.h along with the Datasets directory could be placed in a separate directory and the main could be compiled and executed as follows:

g++ -O3 -ffast-math -march=native -fopenmp -std=c++17 -o gnn_binary main_file_name.cpp

./gnn_binary <dataset_name> //dataset name as per the directory name in the Datasets dir needs to be passed as a command line arg.