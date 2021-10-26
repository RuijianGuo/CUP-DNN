# CUP-DNN
CUP-DNN is a deep neural network model used to predict tissues of origin for cancers of unknown of primary. 
The model wad trained on the expression data of 2387 genes from TCGA RNAseq data of 32 tumor types. We would like to use this pretrained model for transfer learning with our clinical data. With the tranfer of pretrained model, we expect that our new model with clinical could predict more tumor types, and partially alleviate the scarcity of clinical data often encountered in clinical studies.   

# Directory structure
The root contains two subdirectories, inputs where data used for model training and prediction is stored, and outputs where the model, accuracy and loss plots are stored.
The three python files starting with 'CUP' are the core scripts of the project. 

# Library dependencies
matplotlib v3.4.3
numpy v1.21.2
pandas v1.3.3
scikit-learn v1.0
tqdm v4.62.3
torch v0.2.2

# Run model training without learning rate scheduler and early stopping
python CUP_traning -i [input_file] 

# Run model training with learning rate scheduler
python CUP_traning -i [input_file] --learning-rate

# Run model training with early stopping
python CUP_traning -i [input_file] --early-stopping

# Run model training with both(strongly recommended)
python CUP_traning -i [input_file] --learning-rate --early-stopping

