import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem

df = pd.read_csv('data/odorants.csv').set_index('CID') # delete solubility/melting and boiling???? and delete the 35 missing??
labels = pd.read_csv('data/labels.csv').set_index('CID')

bitlength = 1024
nmolecules = len(df)
fp = np.zeros([nmolecules, bitlength]) #1024 values bits 0,1 - 1024-bit ECFP4 fingerprint
count = 0

for smiles in df['IsomericSMILES']:
    mol = Chem.MolFromSmiles(smiles) # from SMILES to mol
    bits = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)) #from mol we obtain the bit vector for each molecules
    fp[count] = np.array(bits) #save each vector as an array
    count = count+1

output = np.array(labels.drop(["IsomericSMILES"], axis=1))
input = fp

