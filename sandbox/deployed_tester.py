#!/usr/bin/env python3
# test a pretrained model
import phyloencode as ph
from phyloencode.PhyloAutoencoder import PhyloAutoencoder
import torch
import joblib
import h5py
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# load trained model and normalizers and create PhyloAutoencoder object
ae_model       = torch.load("ae_trained.pt")
phy_normalizer = joblib.load("ae_test.phy_normalizer.pkl")
aux_normalizer = joblib.load("ae_test.aux_normalizer.pkl")

tree_autoencoder = PhyloAutoencoder(model     = ae_model, 
                                    optimizer = torch.optim.Adam(ae_model.parameters()), 
                                    loss_func = torch.nn.MSELoss(), 
                                    phy_loss_weight = 1.0)


# import test data
with h5py.File("test_data/peak_time.train.hdf5", "r") as f:
    test_phy_data = torch.tensor(f['phy_data'][45000:45100,...], dtype = torch.float32)
    test_aux_data = torch.tensor(f['aux_data'][45000:45100,...], dtype = torch.float32)


# make predictions with trained model
phydat = phy_normalizer.transform(test_phy_data)
auxdat = aux_normalizer.transform(test_aux_data)
phydat = phydat.reshape((phydat.shape[0], ae_model.num_structured_input_channel, 
                         int(phydat.shape[1]/ae_model.num_structured_input_channel)), 
                         order = "F")
phydat = torch.Tensor(phydat)
auxdat = torch.Tensor(auxdat)
phy_pred, auxpred = tree_autoencoder.predict(phydat, auxdat)
phy_pred = phy_normalizer.inverse_transform(phy_pred.reshape((phy_pred.shape[0], -1), order = "F"))

for i in range(50,53):
    print(test_phy_data.numpy()[i,18:24])
    print(np.array(phy_pred[i,18:24]))
    print("    ")

    
# tree latent space check on test trees
encoded_test_trees = tree_autoencoder.tree_encode(phydat, auxdat)

# for PCA analysis of a sample of training trees
# make encoded tree file
latent_dat = tree_autoencoder.tree_encode(phydat, auxdat)

latent_dat_df = pd.DataFrame(latent_dat.detach().to('cpu').numpy(), columns = None, index = None)
latent_dat_df.to_csv("testdat_latent_for_pca.csv", header = False, index = False)