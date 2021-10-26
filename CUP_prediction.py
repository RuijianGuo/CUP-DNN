
import torch
import pandas as pd
import numpy as np
from CUP_training import CUP

def data_transform(data):
    tpm = 1000000*data/np.sum(data)
    log_tpm = np.log2(tpm+0.001)
    return log_tpm 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CUP model prediction')
    parser.add_argument('--data_training','-dt',dest = 'data_train', help ='mRNA dataset used for cup model training')
    parser.add_argument('--data_test', '-dp', dest = 'data_test', help ='RNAseq data with TPM values used tp predict tissue of origin')
    parser.add_argument('--model_path', '-m', dest = 'model_path', help = 'the path in which the trained cup model was stored')
    args = parser.parse_args() 
    
    print('Model prediction starts!')
    #load pretained cup model
    model = CUP().double()
    pretrained_model = torch.load(args.model_path,map_location='cpu')
    model.load_state_dict(pretrained_model['model'])
    scaler = pretrained_model['scaler']
    label = pretrained_model['label']
    # 
    data_training = pd.read_csv(args.data_train, index_col = 0)
    data_training = data_training.drop('TCGA_codes', axis = 1)
    print(data_training.columns)
    data_test = pd.read_csv(args.data_test) 
    data_test = data_test.loc[~data_test.duplicated('ensemble_id'),:]
    data_test = data_test.set_index('ensemble_id')
    columns = data_test.columns
    data_master = data_test.reindex(data_training.columns).T
    data_master = data_master.apply(data_transform, axis = 1)
    data_std = scaler.transform(data_master)
    data_master = torch.from_numpy(data_std)

    model.eval()
    out = model(data_master)
    _, preds = torch.max(out.data,1)
    res = pd.DataFrame(preds, index = columns,columns =['label'])
    res['sample'] = res['label'].map(label)
    res.to_csv('outputs/test_results.csv')   
    print('Results prediction finished')
