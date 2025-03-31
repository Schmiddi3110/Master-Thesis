import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

sys.path.append(BASE_DIR)
import json
import random
import joblib
import optuna
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from aeon.datasets import load_classification
from misc.utils import train_models
from aeon.classification.convolution_based import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from art.estimators.classification import PyTorchClassifier
from art.attacks import ExtractionAttack
from art.attacks.extraction import CopycatCNN, KnockoffNets
from misc.utils import normalize_data, pad_time_series, create_folders, get_percentage_per_class
from models.CNN_Models import *
from models.RNN_Models import BaseRNN1D
from Code.models.LSTM_Models import BaseLSTM1D
from models.AeonClassifier import AeonClassifier
from attack.CustomKnockoffNets import CustomKnockoffNets
from Code.Experiments.base_attack.basic_attack import run_basic_attack
from misc.model_utils import load_trained_model, load_untrained_model, get_conv_architectures_catalogue
from models.VAE import VAE
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser(description="Model Extraction Attacks Against Timeseries Models")
parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset")
args = parser.parse_args()
dataset = args.dataset

random.seed(420)
np.random.seed(420)
torch.manual_seed(420)
torch.cuda.manual_seed_all(420)

num_epochs = int(os.getenv("num_epochs"))
learning_rate = float(os.getenv("learning_rate"))
batch_size = int(os.getenv("batch_size"))
patience = int(os.getenv("patience"))
hidden_size = int(os.getenv("hidden_size"))
num_layers = int(os.getenv("num_layers"))
latent_dim = int(os.getenv("latent_dim"))
    
device = os.getenv("device")
torch.cuda.set_device(device)


X, y = load_classification(dataset)
le = LabelEncoder()
y = le.fit_transform(y)

X_train_val, X_test0, y_train_val, y_test0 = train_test_split(X, y, test_size=0.15, random_state=42)


X_train0, X_val0, y_train0, y_val0 = train_test_split(X_train_val, y_train_val, test_size=0.1765, random_state=42) 

if X.shape[2] <= 9:
    X_train0 = pad_time_series(X_train0)
    X_test0 = pad_time_series(X_test0)
    X_val0 = pad_time_series(X_val0)

X_train0 = normalize_data(X_train0)
X_test0 = normalize_data(X_test0)
X_val0 = normalize_data(X_val0)

num_classes = len(np.unique(y))

input_size = X_train0.shape[2]     
num_channels = X_train0.shape[1]

create_folders(f"TimeSeriesModels/{dataset}")

train_data = TensorDataset(torch.FloatTensor(X_train0).to(device), torch.LongTensor(y_train0).to(device))
val_data = TensorDataset(torch.FloatTensor(X_val0).to(device), torch.LongTensor(y_val0).to(device))
test_data = TensorDataset(torch.FloatTensor(X_test0).to(device), torch.LongTensor(y_test0).to(device))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

model_dir = f'TimeSeriesModels/{dataset}/Models'

train_models(model_dir, num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, train_loader, val_loader, device, X_train0, y_train0, dataset)

if not os.path.exists(f'TimeSeriesModels/{dataset}/Models/test_accuracies.json'):
    from misc.utils import test_models
    test_models(num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, X_test0, y_test0, dataset, device)

conv_architectures_catalogue = get_conv_architectures_catalogue( dataset, X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, device)

basic_attack_results = []
N = 1  
basic_attack_results.extend(run_basic_attack(N, conv_architectures_catalogue, X_test0, y_test0, batch_size, num_epochs, learning_rate))
df = pd.DataFrame(basic_attack_results, columns=('Iteration', 'Model Pairing', 'Method Name', 'Stealing Dataset Size', 'Accuracy', 'Original Model Accuracy', 'Parameter Count', 'Attack duration'))

df.to_csv(f"TimeSeriesModels/{dataset}/Results/BaseAttacks/Base_Attack.csv", index=False)
