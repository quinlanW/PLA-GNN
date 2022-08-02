# PLA-GNN

PLA-GNN: Systematic identification of protein localization alterations in breast cancer cells under Bortezomib perturbation with deep graph neural networks

## 1.Requirements

+ CUDA == 11.0.2
+ python == 3.8
+ pytorch == 1.7.0
+ torchvision == 0.8.0
+ torchaudio == 0.7.0
+ cudatoolkit ==11.0
+ dgl_cu110 == 0.6.0

## 2.Project Catalog Structure

> code
>
> > ```
> > Store project code
> > ```
>
> data (need to create on your own)
>
> >generate_materials
> >
> >> ```
> >> Store generated large files
> >> ```
> >
> >support_materials
> >
> >> ```
> >> Store project-dependent data files
> >> ```
> >
> >log
> >
> >> ```
> >> Store generated results
> >> ```

## 3.Data Preparation

> All of the following files are stored in the `data/support_materials` folder

+ BIOGRID-ORGANISM-Homo_sapiens-4.4.203.mitab.txt (Download from [BioGRID](https://downloads.thebiogrid.org/BioGRID/Release-Archive/))

+ GSE30931_series_matrix.txt (Download from [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30931))

+ uniprot_sprot_human.dat.gz (Download from [Uniprot](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions/))
+ cellular_component.txt (Customized files with multiple subcellular localizations)

## 4.Running programs

> Before running the program, you must ensure that all required files are downloaded and placed in the correct path, and create the appropriate folder. Finally, switch the runtime environment to the `code` folder

1. Run the R script `data_reader.R`
2. Run `python data_preprocess.py` in the terminal
3. Run `utils.py` in the terminal
4. Run `python main_normal.py` in the terminal
5. Run `python main_inter.py` in the terminal
6. Run `python main.py` in the terminal