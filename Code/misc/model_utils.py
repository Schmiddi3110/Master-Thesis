# model_utils.py

import os
import json
import joblib
import torch
from models.CNN_Models import Attention1DCNN, Dilated1DCNN, Residual1DCNN, Deep1DCNN
from models.RNN_Models import BaseRNN1D
from Code.models.LSTM_Models import BaseLSTM1D
from misc.utils import pad_time_series  

def load_trained_model(dataset, original_model_path, X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device):
    with open(f'TimeSeriesModels/{dataset}/Models/test_accuracies.json', 'r') as f:
        loaded_accuracies = json.load(f)
    
    if 'multiRocket' in original_model_path:        
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['multiRocket']
    elif 'miniRocket' in original_model_path:
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['miniRocket']
    elif 'rocket' in original_model_path:
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['rocket']
    elif 'multiRocketHydra' in original_model_path:
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['multiRocketHydra']
    elif 'hydra' in original_model_path:
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['hydra']
    elif 'arsenal' in original_model_path:
        model = joblib.load(original_model_path)
        original_model_accuracy = loaded_accuracies['arsenal']
    elif 'Attention1DCNN' in original_model_path:
        model = Attention1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
        _ = model(dummy_input)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['Attention1DCNN']
    elif 'Dilated1DCNN' in original_model_path:
        model = Dilated1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
        _ = model(dummy_input)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['Dilated1DCNN']
    elif 'Residual1DCNN' in original_model_path:
        model = Residual1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
        _ = model(dummy_input)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['Residual1DCNN']
    elif 'Deep1DCNN' in original_model_path:
        model = Deep1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
        _ = model(dummy_input)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['Deep1DCNN']
    elif 'BaseRNN1D' in original_model_path:
        model = BaseRNN1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=X_test0.shape[2], hidden_size=128, num_layers=2).to(device)
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['BaseRNN1D']
    elif 'BaseLSTM1D' in original_model_path:
        model = BaseLSTM1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=X_test0.shape[2], hidden_size=128, num_layers=2).to(device)        
        model.load_state_dict(torch.load(original_model_path, map_location=device))
        original_model_accuracy = loaded_accuracies['BaseLSTM1D']


    else:
        raise ValueError(f"Model type not recognized from path: {original_model_path}")
    
    return model, original_model_accuracy

def load_untrained_model(model, num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device):
    if '1_Conv_CNN' in model:
        from models.CNN_Models import Simple1DCNN
        substitute_model = Simple1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
        substitute_model.apply(substitute_model.weights_init)
    elif '2_Conv_CNN' in model:
        from models.CNN_Models import BaseCNN1D
        substitute_model = BaseCNN1D(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
        substitute_model.apply(substitute_model.weights_init)
    elif '3_Conv_CNN' in model:
        substitute_model = Deep1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
        substitute_model.apply(substitute_model.weights_init)
    elif 'LSTM' in model:
        substitute_model = BaseLSTM1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    elif 'RNN' in model:
        substitute_model = BaseRNN1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
    else:
        raise ValueError(f"Model type not recognized from path: {model}")
    return substitute_model

def get_small_conv_architectures_catalogue(dataset, X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, device):
    catalogue = {        
    'CNN-CNN': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'RNN-CNN': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseRNN1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'LSTM-CNN': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseLSTM1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'CNN-RNN': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'RNN-RNN': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseRNN1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'LSTM-RNN':[
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseLSTM1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'CNN-LSTM': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'RNN-LSTM': [
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseRNN1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    'LSTM-LSTM':[
        load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/BaseLSTM1D.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
        load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],

    }
    return catalogue

def get_conv_architectures_catalogue(dataset, X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, device):
    catalogue = {
        # LSTM variants
        'MultiRocket-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MiniRocket-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Rocket-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/rocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MultiRocketHydra-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Hydra-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/hydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Arsenal-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/arsenal.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Attention_CNN-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'DilatedCNN-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'ResidualCNN-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'BaseCNN1D_Deep-LSTM': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("LSTM", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        # RNN variants
        'MultiRocket-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MiniRocket-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Rocket-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/rocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MultiRocketHydra-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Hydra-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/hydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Arsenal-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/arsenal.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Attention_CNN-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'DilatedCNN-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'ResidualCNN-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'BaseCNN1D_Deep-RNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("RNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        # 1_Conv_CNN variants
        'MultiRocket-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MiniRocket-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Rocket-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/rocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MultiRocketHydra-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Hydra-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/hydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Arsenal-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/arsenal.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Attention_CNN-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'DilatedCNN-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'ResidualCNN-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'BaseCNN1D_Deep-1_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("1_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        # 2_Conv_CNN variants
        'MultiRocket-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MiniRocket-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Rocket-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/rocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MultiRocketHydra-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Hydra-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/hydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Arsenal-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/arsenal.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Attention_CNN-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'DilatedCNN-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'ResidualCNN-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'BaseCNN1D_Deep-2_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("2_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        # 3_Conv_CNN variants
        'MultiRocket-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MiniRocket-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Rocket-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/rocket.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'MultiRocketHydra-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Hydra-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/hydra.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Arsenal-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/arsenal.pkl", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "aeon"],
        'Attention_CNN-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'DilatedCNN-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'ResidualCNN-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
        'BaseCNN1D_Deep-3_Conv_CNN': [
            load_trained_model(dataset, f"TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth", X_test0, num_classes, num_channels, num_epochs, learning_rate, patience, device),
            load_untrained_model("3_Conv_CNN", num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, device), "torch"],
    }
    return catalogue
