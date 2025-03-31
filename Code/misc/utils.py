import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib
import json
from aeon.classification.convolution_based import *
from models.CNN_Models import *
from models.RNN_Models import BaseRNN1D
from Code.models.LSTM_Models import BaseLSTM1D

def model_exists(model_name, dataset):
    return os.path.exists(os.path.join(f'TimeSeriesModels/{dataset}/Models', model_name))

def train_models(model_dir, num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, train_loader, val_loader, device, X_train0, y_train0, dataset):
    if not model_exists("Simple1DCNN.pth", dataset):
        model = Simple1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("Deep1DCNN.pth", dataset):
        model = Deep1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("Dilated1DCNN.pth", dataset):
        model = Dilated1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("Attention1DCNN.pth", dataset):
        model = Attention1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("Residual1DCNN.pth", dataset):
        model = Residual1DCNN(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("BaseLSTM1D.pth", dataset):
        model = BaseLSTM1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("BaseRNN1D.pth", dataset):
        model = BaseRNN1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    if not model_exists("BaseCNN1D.pth", dataset):
        model = BaseCNN1D(num_classes=num_classes, in_channels=num_channels, num_epochs=num_epochs, learning_rate=learning_rate, patience=patience).to(device)
        model.apply(model.weights_init)
        model.train_model(train_loader, val_loader, model_dir)
        del model

    # Train and save scikit-learn-based classifiers if they don’t exist
    if not model_exists("arsenal.pkl", dataset):
        arsenal = Arsenal(n_jobs=-1)
        print("Training Arsenal Classifier...")
        arsenal.fit(X_train0, y_train0)
        joblib.dump(arsenal, os.path.join(model_dir, "arsenal.pkl"))
        del arsenal

    if not model_exists("hydra.pkl", dataset):
        hydra = HydraClassifier()
        print("Training Hydra Classifier...")
        hydra.fit(X_train0, y_train0)
        joblib.dump(hydra, os.path.join(model_dir, "hydra.pkl"))
        del hydra

    if not model_exists("multiRocketHydra.pkl", dataset):
        multiRocketHydra = MultiRocketHydraClassifier()
        print("Training MultiRocketHydra Classifier...")
        multiRocketHydra.fit(X_train0, y_train0)
        joblib.dump(multiRocketHydra, os.path.join(model_dir, "multiRocketHydra.pkl"))
        del multiRocketHydra

    if not model_exists("rocket.pkl", dataset):
        rocket = RocketClassifier()
        print("Training Rocket Classifier...")
        rocket.fit(X_train0, y_train0)
        joblib.dump(rocket, os.path.join(model_dir, "rocket.pkl"))
        del rocket

    if not model_exists("miniRocket.pkl", dataset):
        miniRocket = MiniRocketClassifier()
        print("Training MiniRocket Classifier...")
        miniRocket.fit(X_train0, y_train0)
        joblib.dump(miniRocket, os.path.join(model_dir, "miniRocket.pkl"))
        del miniRocket

    if not model_exists("multiRocket.pkl", dataset):
        multiRocket = MultiRocketClassifier()
        print("Training MultiRocket Classifier...")
        multiRocket.fit(X_train0, y_train0)
        joblib.dump(multiRocket, os.path.join(model_dir, "multiRocket.pkl"))
        del multiRocket


def pad_time_series(X, min_timepoints=10):
    X_padded = []
    
    for sample in X:
        n_channels, n_timepoints = sample.shape

        if n_timepoints < min_timepoints:
            pad_width = min_timepoints - n_timepoints
            padded_sample = np.pad(sample, ((0, 0), (0, pad_width)), mode='constant')
        else:
            padded_sample = sample

        X_padded.append(padded_sample)

    return np.array(X_padded)



def create_folders(parent_folder, subfolders=["Results", "Models", "Hyperparameter", "Gen_Models"]):
    try:
        os.makedirs(parent_folder, exist_ok=True)
        print(f"Parent folder '{parent_folder}' created successfully.")
        
        for subfolder in subfolders:
            os.makedirs(os.path.join(parent_folder, subfolder), exist_ok=True)
            print(f"Subfolder '{subfolder}' created inside '{parent_folder}'.")
            
            if subfolder == "Results":
                additional_subfolders = ["BaseAttacks", "Defense", "Noise", "MixedVAE", "PureVAE", "Noise_VAE_Defense"]
                for extra_subfolder in additional_subfolders:
                    os.makedirs(os.path.join(parent_folder, subfolder, extra_subfolder), exist_ok=True)
                    print(f"Subfolder '{extra_subfolder}' created inside '{subfolder}'.")
        
    except Exception as e:
        print(f"An error occurred: {e}")



def normalize_data(X):
    scaler = MinMaxScaler()
    X_shape = X.shape
    X = X.reshape(X_shape[0], -1)
    X = scaler.fit_transform(X)
    return X.reshape(X_shape)



def test_models(num_classes, num_channels, learning_rate, num_epochs, patience, input_size, hidden_size, num_layers, X_test0, y_test0, dataset, device):
    from sklearn.metrics import accuracy_score
    from models.CNN_Models import Deep1DCNN, Residual1DCNN, Attention1DCNN, Dilated1DCNN, BaseCNN1D, Simple1DCNN
    from models.RNN_Models import BaseRNN1D
    from models.LSTM_Models import BaseLSTM1D
    import torch


    accuracy_results = {}
    model = Deep1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)

    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device) 
    _ = model(dummy_input)

    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/Deep1DCNN.pth', map_location=device, weights_only=True))

    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        deep_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {deep_cnn_accuracy:.4f}')
        accuracy_results['Deep1DCNN'] = deep_cnn_accuracy

    del model
    
    model = Residual1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)  
    _ = model(dummy_input)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/Residual1DCNN.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        residual_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {residual_cnn_accuracy:.4f}')
        accuracy_results['Residual1DCNN'] = residual_cnn_accuracy
    del model


    model = Attention1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)  
    _ = model(dummy_input)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/Attention1DCNN.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        attention_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {attention_cnn_accuracy:.4f}')
        accuracy_results['Attention1DCNN'] = attention_cnn_accuracy
    del model


    model = Dilated1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device) 
    _ = model(dummy_input)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/Dilated1DCNN.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        dilated_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {dilated_cnn_accuracy:.4f}')
        accuracy_results['Dilated1DCNN'] = dilated_cnn_accuracy
    del model


    model = BaseLSTM1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/BaseLSTM1D.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0)
        test_labels = torch.LongTensor(y_test0)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)    
        lstm_accuracy = (predicted == test_labels).sum().item()/ test_labels.size(0)
        print(f'Test Accuracy: {lstm_accuracy:.4f}')
        accuracy_results['BaseLSTM1D'] = lstm_accuracy
    del model


    model = BaseRNN1D(num_classes=num_classes, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience, input_size=input_size, hidden_size=hidden_size,num_layers=num_layers)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/BaseRNN1D.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0)
        test_labels = torch.LongTensor(y_test0)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)    
        rnn_accuracy = (predicted == test_labels).sum().item()/ test_labels.size(0)
        print(f'Test Accuracy: {rnn_accuracy:.4f}')
        accuracy_results['BaseRNN1D'] = rnn_accuracy
    del model


    model = BaseCNN1D(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
    _ = model(dummy_input)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/BaseCNN1D.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        base_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {base_cnn_accuracy:.4f}')
        accuracy_results['BaseCNN1D'] = base_cnn_accuracy
    del model


    model = Simple1DCNN(num_classes=num_classes, in_channels=num_channels, learning_rate=learning_rate, num_epochs=num_epochs, patience=patience).to(device)
    dummy_input = torch.randn(1, X_test0.shape[1], X_test0.shape[2]).to(device)
    _ = model(dummy_input)
    model.load_state_dict(torch.load(f'TimeSeriesModels/{dataset}/Models/Simple1DCNN.pth', map_location=device, weights_only=True))
    with torch.no_grad():
        model.eval()
        test_data = torch.FloatTensor(X_test0).to(device)
        test_labels = torch.LongTensor(y_test0).to(device)
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        simple_cnn_accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)
        print(f'Test Accuracy: {simple_cnn_accuracy:.4f}')
        accuracy_results['Simple1DCNN'] = simple_cnn_accuracy
    del model

    arsenal = joblib.load(f"TimeSeriesModels/{dataset}/Models/arsenal.pkl")
    pred = arsenal.predict(X_test0)
    arsenal_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {arsenal_accuracy:.4f}")
    accuracy_results['arsenal'] = arsenal_accuracy
    del arsenal


    hydra = joblib.load(f"TimeSeriesModels/{dataset}/Models/hydra.pkl")
    pred = hydra.predict(X_test0)
    hydra_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {hydra_accuracy:.4f}")
    accuracy_results['hydra'] = hydra_accuracy
    del hydra


    multiRocketHydra = joblib.load(f"TimeSeriesModels/{dataset}/Models/multiRocketHydra.pkl")
    pred = multiRocketHydra.predict(X_test0)
    multi_rocket_hydra_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {multi_rocket_hydra_accuracy:.4f}")
    accuracy_results['multiRocketHydra'] = multi_rocket_hydra_accuracy
    del multiRocketHydra


    rocket = joblib.load(f"TimeSeriesModels/{dataset}/Models/rocket.pkl")
    pred = rocket.predict(X_test0)
    rocket_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {rocket_accuracy:.4f}")
    accuracy_results['rocket'] = rocket_accuracy
    del rocket


    miniRocket = joblib.load(f"TimeSeriesModels/{dataset}/Models/miniRocket.pkl")
    pred = miniRocket.predict(X_test0)
    mini_rocket_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {mini_rocket_accuracy:.4f}")
    accuracy_results['miniRocket'] = mini_rocket_accuracy
    del miniRocket


    multiRocket = joblib.load(f"TimeSeriesModels/{dataset}/Models/multiRocket.pkl")
    pred = multiRocket.predict(X_test0)
    multi_rocket_accuracy = accuracy_score(y_test0, pred)
    print(f"Test Accuracy: {multi_rocket_accuracy:.4f}")
    accuracy_results['multiRocket'] = multi_rocket_accuracy
    del multiRocket
    
    with open(f'TimeSeriesModels/{dataset}/Models/test_accuracies.json', 'w') as f:
        json.dump(accuracy_results, f)


def get_percentage_per_class(X, y, percentage):
    selected_samples = []
    selected_labels = []

    for cls in np.unique(y):
        class_indices = np.where(y == cls)[0]
        selected_indices = np.random.choice(class_indices, max(1, int(percentage * len(class_indices))), replace=False)        
        
        selected_samples.append(X[selected_indices])
        selected_labels.append(y[selected_indices])

    X_gan_train = np.vstack(selected_samples)
    y_gan_train = np.concatenate(selected_labels)
    return X_gan_train, y_gan_train


