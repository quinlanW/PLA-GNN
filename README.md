# PLA-GNN

> PLA-GNN: Systematic identification of protein localization alterations in breast cancer cells under Bortezomib perturbation with deep graph neural networks

In this work, we used PLA-GNN (**P**rotein **L**ocalization **A**lterations by **G**raph **N**eural **N**etwork) to identify alterations of  protein localizations in the drug perturbation state. We took the **Bortezomib** and **breast cancer** cell as instances for this study.

---

## 1 Requirements

### 1.1 R

| Requirements       | Release |
| :----------------- | :------ |
| R                  | 3.6     |
| reshape2           | 1.4.4   |
| illuminaHumanv4.db | 1.26.0  |
| dplyr              | 1.0.9   |

### 1.2 Python

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

> code
>
> > This folder stores the code files. 
> >
> > | File               | Description                                                  |
> > | :----------------- | :----------------------------------------------------------- |
> > | data_reader.R      | Convert gene expression data to CSV file from GEO downloads. |
> >| data_preprocess.py | Generate multi-conditional PPI, ECC, PCC sparse matrix files and some related ancillary files. |
> > | utils.py           | Include building graph, data normalization and other code.   |
> >| model.py           | Include graph neural network model code.                     |
> > | train.py           | Model training and data storage related code.                |
> > | main_normal.py     | Prediction code for protein location score in control state. |
> > | main_inter.py      | Prediction code for protein localization score in Bortezomib perturbation state. |
> >| main.py            | Discovery of potentially mis-localized proteins in Bortezomib perturbation. |
> > | performance.py     | Performance in the control state and randomized trial.       |
> >| figure.py          | The breakdown of the dataset for different localization multiplicity figure and the number of proteins in each subcellular location figure. |
> > | statistics.py      | Generate statistical information about the topology adjustment process. |
> 
> data 
> 
> >generate_materials (need to create on your own)
> >
> >> This folder stores the files generated during program execution for subsequent use. The files in this folder will be generated continuously during the program runtime.
> >
> >support_materials
> >
> >> This folder stores project-dependent data files. The files in this folder need to be downloaded and placed in the correct path by yourself before the program running. A more detailed description of the file download will be presented in Section 3.
> >
> >log (need to create on your own)
> >
> >> This folder stores program execution results (Includes images, run records in txt format, result matrix, etc.).

---

## 3.Data Preparation

> All of the following files are stored in the `data/support_materials` folder

+ BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt (Download from [BioGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/))
+ GSE30931_series_matrix.txt (Download from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30931))
+ uniprot_sprot_human.dat.gz (Download from [Uniprot](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/))
+ cellular_component.txt (Customized files with multiple subcellular localizations)

---

## 4.Running programs

> Before running the program, you must ensure that all required files are downloaded and placed in the correct path, and create the appropriate folder. Finally, switch the runtime environment to the `code` folder. The result will be saved in the `log` folder.

1. Run the R script `data_reader.R` to generate the expression file (csv).
2. Run `python data_preprocess.py` in the terminal to generate the sparse matrix files for subsequent use.
3. Run `utils.py` in the terminal.
4. Run `python main_normal.py` in the terminal.
5. Run `python main_inter.py` in the terminal.
6. Run `python main.py` in the terminal to get the protein mis-location scores files.

---

## 5.Hyper-parameters description

|   Hyper-parameter   | Type  | Default |                         Description                          |
| :-----------------: | :---: | :-----: | :----------------------------------------------------------: |
| learning rate [-lr] | float | 0.00005 |       Learning rate, the default value is recommended.       |
|      fold [-f]      |  int  |   10    | Cross-validation fold number, 10-fold cross-validation was used in the original paper. |
|     epoch [-e]      |  int  |   200   |           Epoch, the default value is recommended.           |
|       α [-a]        | list  |  [0.1]  | α, set as 0.1 can get the best performance of AIM and mlACC. Support input list for comparison between different α (eg. [-lr 0.1 0.2 0.3]). |
|     device [-d]     |  str  | 'cuda'  | Support CPU or GPU to run programs, the default value is recommended. |

> The command `python main_normal.py -h` or `python main_inter.py -h` can be used to display a brief description of the parameters.