# PLA-GNN

> PLA-GNN: Computational inference of protein subcellular location alterations under drug treatments with deep graph neural networks

In this work, we used PLA-GNN (**P**rotein **L**ocalization **A**lterations by **G**raph **N**eural **N**etwork) to identify alterations of  protein localizations in the drug perturbation state. We took the case of studies on three drugs, **tacrolimus**, **bortezomib** and **trichostain** **A (TSA)** as instances for this study.

---

## 1 Requirements<a id='1'></a>

### 1.1 R requirements

| Requirements       | Release |
| :----------------- | :------ |
| R                  | 3.6     |
| reshape2           | 1.4.4   |
| illuminaHumanv4.db | 1.26.0  |
| dplyr              | 1.0.9   |

### 1.2 Python requirements

| Requirements | Release      |
| :----------- | :----------- |
| CUDA         | 11.6         |
| Python       | 3.8.10       |
| torch        | 1.10.0+cu113 |
| dgl_cu113    | 0.8.2.post1  |
| numpy        | 1.21.4       |
| pandas       | 1.4.3        |
| scikit-learn | 1.1.1        |
| spicy        | 1.8.1        |
| matplotlib   | 3.5.0        |
| tqdm         | 4.61.2       |

---

## 2 Project Catalog Structure

### code<a id='2-1'></a>

> This folder stores the code files. 

+ data_reader.R<a id='2-1-data_reader'></a>

    > This file convert gene expression data to CSV file from GEO downloads. Running the code will generate the corresponding CSV expression files in the `data/support_materials` folder.
    >
    > | Generate file        | Descriptrion             |
    > | -------------------- | ------------------------ |
    > | GSE74572_exprSet.csv | GSE74572 expression data |
    > | GSE30931_exprSet.csv | GSE30931 expression data |
    > | GSE27182_exprSet.csv | GSE27182 expression data |

+ data_preprocess.py<a id='2-1-data_pre'></a>

    > This file generate multi-conditional PPI, ECC, PCC sparse matrix files and some related ancillary files.  The generated files will be stored in the `data/generate_materials` folder.
    >
    > | Generate file                   | Description                                                  |
    > | ------------------------------- | ------------------------------------------------------------ |
    > | protein_ppi.json                | List of proteins in the protein-protein interaction network. |
    > | PPI_normal.npz                  | Adjacency matrix of the protein-protein interaction network in the **control state**. |
    > | ECC_normal.npz                  | ECC matrix in the **control state**.                         |
    > | ECC_normal_pca.npy              | ECC matrix after PCA processing in the **control state**.    |
    > | GCN_normal.npz                  | Protein co-expression matrix in the **control state**.       |
    > | GCN_normal_pca.npy              | Protein co-expression matrix after PCA processing in the **control state**. |
    > | expr_normal.npy                 | Protein expression matrix in the **control state**.          |
    > | GSE74572_data/PPI_inter.npz     | Adjacency matrix of the protein-protein interaction network in the **TSA perturbation state**. |
    > | GSE74572_data/ECC_inter.npz     | ECC matrix in the **TSA perturbation state.**                |
    > | GSE74572_data/ECC_inter_pca.npy | ECC matrix after PCA processing in the **TSA perturbation state**. |
    > | GSE74572_data/GCN_inter.npz     | Protein co-expression matrix in the **TSA perturbation state**. |
    > | GSE74572_data/GCN_inter_pca.npy | Protein co-expression matrix after PCA processing in the **TSA perturbation state**. |
    > | GSE74572_data/expr_inter.npy    | Protein expression matrix in the **TSA perturbation state**. |
    > | GSE30931_data/PPI_inter.npz     | Adjacency matrix of the protein-protein interaction network in the **bortezomib perturbation state**. |
    > | GSE30931_data/ECC_inter.npz     | ECC matrix in the **bortezomib perturbation state**.         |
    > | GSE30931_data/ECC_inter_pca.npy | ECC matrix after PCA processing in the **bortezomib perturbation state**. |
    > | GSE30931_data/GCN_inter.npz     | Protein co-expression matrix in the **bortezomib perturbation state**. |
    > | GSE30931_data/GCN_inter_pca.npy | Protein co-expression matrix after PCA processing in the **bortezomib perturbation state**. |
    > | GSE30931_data/expr_inter.npy    | Protein expression matrix in the **bortezomib perturbation state**. |
    > | GSE27182_data/PPI_inter.npz     | Adjacency matrix of the protein-protein interaction network in the **tacrolimus perturbation state**. |
    > | GSE27182_data/ECC_inter.npz     | ECC matrix in the **tacrolimus perturbation state**.         |
    > | GSE27182_data/ECC_inter_pca.npy | ECC matrix after PCA processing in the **tacrolimus perturbation state**. |
    > | GSE27182_data/GCN_inter.npz     | Protein co-expression matrix in the **tacrolimus perturbation state**. |
    > | GSE27182_data/GCN_inter_pca.npy | Protein co-expression matrix after PCA processing in the **tacrolimus perturbation state**. |
    > | GSE27182_data/expr_inter        | Protein expression matrix in the **tacrolimus perturbation state**. |
    > | loc_matrix.npz                  | Protein localization matrix.                                 |
    > | label_list.json                 | List of proteins and their subcellular localization in protein-protein interaction networks. |
    > | label_with_loc_list.json        | The index list in the` label_list.json` file for the proteins with positioning are reported in `uniprot_sprot_human.dat.gz` in the PPI network. |

+ utils.py

    > This file include building graph, data normalization and other code.

+ model.py

    > This file include graph neural network model code.

+ train.py

    > This file include model training and data storage related code. During the running of the code, log files are generated and stored in the `data/log` folder.

+ main_normal.py

    > Prediction code for protein location score in control state.

+ main_inter.py

    > Prediction code for protein localization score in drug perturbation state.

+ main.py

    > Discovery of potentially mis-localized proteins in drug perturbation. Code run completion will generate protein mis-localizaion score tables.

+ performance.py

    > Performance in the control state and randomized trial.

+ statistics.py

    > Generate statistical information about the topology adjustment process.

### data

#### support_materials

> This folder stores project-dependent data files. The files in this folder need to be downloaded and placed in the correct path by yourself before the program running. A more detailed description of the file download will be presented in [Section 3](#3).
>
> | File                                            | Description                                                  |
> | ----------------------------------------------- | ------------------------------------------------------------ |
> | BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt | Protein-protein interaction data downloaded from [BioGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/). |
> | GSE30931_series_matrix.txt                      | Proteasome inhibition blocks estrogen-dependent gene transcription by decreasing histone H2B monoubiquitination in human breast cancer cells. |
> | GSE27182_series_matrix.txt                      | Expression data in HK-2 cells following exposure to Fk.      |
> | GSE74572_series_matrix.txt                      | Effect of APE1 and its acetylation on gene expression profile of lung cancer cells. |
> | uniprot_sprot_human.dat.gz                      | Protein localization data downloaded from [Uniprot](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/) |
> | cellular_component.txt                          | Customized files with multiple subcellular localizations. Can be modified by yourself, visit [GENEONTOLOGY](http://geneontology.org) for inquiry. |

#### generate_materials

> This folder stores the files generated during program execution for subsequent use. The files in this folder will be generated continuously during the program runtime. Please see [section code](#2-1-data_pre) in Section 2 for detailed documentation

#### log

> This folder stores program execution results (Includes images, run records in txt format, result matrix, etc).

#### res

> This folder holds the prediction results for each dataset.

---

## 3.Data Preparation<a id='3'></a>

> All of the following files are stored in the `data/support_materials` folder

+ BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt (Download from [BioGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/))

    > Please check the appropriate version when downloading.

+ uniprot_sprot_human.dat.gz (Download from [Uniprot](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/))

    > Please check the appropriate version when downloading.

+ GSE30931_series_matrix.txt (Download from [NCBI GEO GSE30931](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30931))

    > Click on 'Series Matrix File(s)' under 'Download family' on the web page to download the file in TXT format.

+ GSE27182_series_matrix.txt (Download from [NCBI GEO GSE27182](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE27182))

    > Click on 'Series Matrix File(s)' under 'Download family' on the web page to download the file in TXT format.

+ GSE74572_series_matrix.txt (Download from [NCBI GEO GSE74572](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74572))

    > Click on 'Series Matrix File(s)' under 'Download family' on the web page to download the file in TXT format.

+ cellular_component.txt 

    > Customized files with multiple subcellular localizations. Can be modified by yourself, visit [GENEONTOLOGY](http://geneontology.org) for inquiry.

---

## 4.Running programs

> Before running the program, you must ensure that all required files are downloaded and placed in the correct path, and create the appropriate folder. Finally, switch the runtime environment to the `code` folder. The result will be saved in the `data/log` and `data/res` folder.
>
> For the environment necessary to run the code, please refer to [Section 1](#1). For the files necessary to run the code, please refer to [Section 3](#3).

1. Run the R script `data_reader.R` to generate the expression files (CSV format).

    > When the code finishes running, it generates the expression data in CSV format corresponding to the three datasets GSE74572, GSE30931 and GSE27182. Please refer to the [section code](#2-1-data_reader) in Section 2 for the specific files to be generated.
2. Run `python data_preprocess.py` in the terminal to generate the sparse matrix files for subsequent use. 

    > This step may consume a lot of memory, if memory is not enough, choose to execute it by steps. It generates matrix files and json files for various conditions after the code finishes running. Please refer to the [section code](#2-1-data_pre) in Section 2 for the specific files to be generated.
3. Run `python main_normal.py -data GSEXXXXX` in the terminal. 

    > This step generates the prediction matrix for the control state. The `XXXXX` in the command line represents the ID of the dataset. This command contains various optional parameters, please refer to [Section 5](#5) for detailed description.
4. Run `python main_inter.py -data GSEXXXXX` in the terminal. 

    > This step generates the prediction matrix for the drug perturbation state. The `XXXXX` in the command line represents the ID of the dataset. This command contains various optional parameters, please refer to [Section 5](#5) for detailed description.
5. Run `python main.py` in the terminal to get the protein mis-location scores files.

    > This step generates protein mis-localization tables(CSV format). 

---

## 5.Hyper-parameters description<a id='5'></a>

| Hyper-parameter | Type    | Default    | Description                                                  |
| :-------------- | :------ | :--------- | :----------------------------------------------------------- |
| `-data`         | `str`   | No default | **Required parameter**, selected from [`GSE30931`, `GSE74572`, `GSE27182`]. Select the current running dataset. |
| `-lr`           | `float` | 0.00005    | **Learning rate**, the default value is recommended.         |
| `-f`            | `int`   | 10         | **Cross-validation fold number**, 10-fold cross-validation was used in the original paper. |
| `-e`            | `int`   | 200        | **Epoch**, the default value is recommended.                 |
| `-a`            | `list`  | [0.1]      | **α**, set as 0.1 can be closest to the original data distribution. Support input list for comparison between different α (eg. `-lr 0.1 0.2 0.3`). |
| `-d`            | `str`   | 'cuda'     | Support **CPU** and **GPU** to run programs, the default value is recommended. |

> The command `python main_normal.py -h` or `python main_inter.py -h` can be used to display a brief description of the parameters.

___

