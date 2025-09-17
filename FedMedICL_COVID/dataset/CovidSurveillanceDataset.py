# dataset/CovidSurveillanceDataset.py
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os

class CovidSurveillanceDataset(Dataset):
    """
    Dataset class for COVID-19 Surveillance tabular data.

    Expects a CSV file with columns for features and a label. If a client_id is provided,
    the dataset will filter rows according to a column named "HospitalID" (if it exists)
    or "simulated_client_id" (if "HospitalID" is absent).
    """
    def __init__(self, csv_file, client_id=None, cat_features=None, num_features=None, label_column="death_yn"):
        """
        Args:
          csv_file (str): Path to the CSV file.
          client_id (str or int, optional): Client identifier.
          cat_features (list): List of column names to use as categorical features.
          num_features (list): List of column names to use as numerical features.
          label_column (str): The label column name.
        """
        # Read CSV with low_memory disabled to avoid dtype warnings
        self.data = pd.read_csv(csv_file, low_memory=False, dtype=str)
        
        # If a client_id is provided, filter data by "HospitalID" if present; otherwise by "simulated_client_id"
        if client_id is not None:
            if "HospitalID" in self.data.columns:
                self.data = self.data[self.data["HospitalID"] == str(client_id)]
            elif "simulated_client_id" in self.data.columns:
                self.data = self.data[self.data["simulated_client_id"] == str(client_id)]
            else:
                print("Warning: No client ID column found; not filtering by client.")
        
        self.cat_features = cat_features or []
        self.num_features = num_features or []
        self.label_column = label_column
        
        # Process categorical features: strip, replace empty strings, fill missing values, and encode as integers.
        for col in self.cat_features:
            self.data[col] = self.data[col].str.strip().replace(r'^\s*$', 'Unknown', regex=True)
            self.data[col] = self.data[col].fillna('Unknown')
            self.data[col] = self.data[col].astype('category').cat.codes
        
        # Process numerical features: convert values to numbers and apply min-max normalization.
        for col in self.num_features:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col] = self.data[col].fillna(self.data[col].median())
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            if col_max > col_min:
                self.data[col] = (self.data[col] - col_min) / (col_max - col_min)
            else:
                self.data[col] = 0.0
        
        # Process the label column: map "No" to 0 and "Yes" to 1.
        self.data[self.label_column] = self.data[self.label_column].map({"No": 0, "Yes": 1})
        # Drop rows where label mapping resulted in NaN (i.e. unexpected or missing values)
        self.data = self.data.dropna(subset=[self.label_column])
        self.data[self.label_column] = self.data[self.label_column].astype(int)
        
        self.X_cat = self.data[self.cat_features].values.astype(np.int64)
        self.X_num = self.data[self.num_features].values.astype(np.float32)
        self.y = self.data[self.label_column].values.astype(np.int64)
        
        self.length = len(self.y)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        import torch
        cat_feats = torch.tensor(self.X_cat[idx], dtype=torch.long)
        num_feats = torch.tensor(self.X_num[idx], dtype=torch.float)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return (cat_feats, num_feats), label

def create_federated_datasets(csv_file, client_ids, cat_features, num_features, label_column):
    """
    Creates a dictionary mapping each client_id to a CovidSurveillanceDataset object.
    If the CSV does not contain a "HospitalID" column, a simulated client ID is assigned.
    
    Returns:
        dict: {client_id: CovidSurveillanceDataset}
    """
    df = pd.read_csv(csv_file, low_memory=False, dtype=str)
    # If "HospitalID" is missing, simulate client IDs.
    if "HospitalID" not in df.columns:
        print("No 'HospitalID' column found; simulating client splitting.")
        num_clients = len(client_ids)
        df["simulated_client_id"] = np.random.randint(0, num_clients, size=len(df)).astype(str)
        # Ensure the directory "data" exists before writing.
        os.makedirs("data", exist_ok=True)
        temp_csv = "data/simulated_clients.csv"
        df.to_csv(temp_csv, index=False)
        csv_file = temp_csv
    datasets = {}
    for cid in client_ids:
        datasets[cid] = CovidSurveillanceDataset(csv_file, client_id=cid, 
                                                 cat_features=cat_features,
                                                 num_features=num_features,
                                                 label_column=label_column)
    return datasets
