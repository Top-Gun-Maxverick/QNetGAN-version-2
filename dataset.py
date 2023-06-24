import tensorflow as tf
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

class DataHelper():
    def __init__(self):
        csv_path = tf.keras.utils.get_file("qm9.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv")
        self.data = list()
        with open(csv_path, "r")  as fin:
            for line in fin.readlines()[1:]:
                self.data.append(line.split(",")[1])
    
    def __getitem__(self, idx: int):
        smiles = self.data[idx]
        print(f"SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        print(f"Number of Heavy Atoms: {mol.GetNumHeavyAtoms()}")
        return mol