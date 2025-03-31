import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.AeonClassifier import AeonClassifier
from art.attacks.extraction import CopycatCNN, KnockoffNets
from art.estimators.classification import PyTorchClassifier
from attack.CustomKnockoffNets import CustomKnockoffNets


import time

def run_basic_attack(N, model_catalogue, X_test0, y_test0, batch_size, num_epochs, learning_rate):        
    conv_model_results = []
    num_classes = len(np.unique(y_test0))

    for iteration in range(N):
        print(f"Iteration {iteration + 1}/{N}")
        
        for model_pairing in model_catalogue:
            print(f"Running attacks for: {model_pairing}")
            
            model, orginal_model_accuracy = model_catalogue[model_pairing][0]
            criterion = nn.CrossEntropyLoss()
            if model_catalogue[model_pairing][2] == "aeon":
                classifier_original = AeonClassifier(model=model, nb_classes=num_classes, input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]))
            elif model_catalogue[model_pairing][2] == "torch":
                optimizer_original = optim.Adam(model.parameters(), lr=learning_rate)
                classifier_original = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer_original, input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]), nb_classes=num_classes)
            else:
                raise ValueError(f"Unrecognized Classifier: {model_catalogue[model_pairing][2]}")

            percentages = [0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
            for percentage in percentages:
                len_steal = max(int(len(X_test0) * percentage), num_classes)  
                x_steal, y_steal = [], []
                selected_indices = []  
                remaining_indices = list(range(len(X_test0)))

                for class_label in range(num_classes):
                    class_indices = np.where(y_test0 == class_label)[0]
                    if len(class_indices) > 0:
                        selected_idx = np.random.choice(class_indices, 1, replace=False)
                        x_steal.append(X_test0[selected_idx])
                        y_steal.append(y_test0[selected_idx])
                        selected_indices.extend(selected_idx)
                        remaining_indices.remove(selected_idx[0])

                additional_needed = len_steal - len(selected_indices)
                additional_indices = []

                if additional_needed > 0:
                    additional_indices = np.random.choice(remaining_indices, additional_needed, replace=False).tolist()
                    x_steal.append(X_test0[additional_indices])
                    y_steal.append(y_test0[additional_indices])
                    selected_indices.extend(additional_indices)

                x_steal = np.vstack(x_steal)  
                y_steal = np.hstack(y_steal)

                x_steal = torch.tensor(x_steal, dtype=torch.float32).numpy()
                y_steal = torch.tensor(y_steal, dtype=torch.float32).numpy()

                x_test = np.delete(X_test0, selected_indices, axis=0)
                y_test = np.delete(y_test0, selected_indices, axis=0)

                extraction_attacks = {
                    "Probabilistic CopycatCNN": CopycatCNN(
                        classifier=classifier_original,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=int(len_steal),
                        use_probability=True),
                    "Argmax CopycatCNN": CopycatCNN(
                        classifier=classifier_original,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=int(len_steal),
                        use_probability=False),
                    "Adaptive KnockoffNets": CustomKnockoffNets(
                        classifier=classifier_original,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=len_steal,
                        verbose=False,
                        sampling_strategy='adaptive',
                        use_probability=False),
                }

                for name, attack in extraction_attacks.items():
                    start_time = time.time()
                    model_stolen = model_catalogue[model_pairing][1]
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model_stolen.parameters(), lr=learning_rate)
                    classifier_stolen = PyTorchClassifier(
                        model=model_stolen, 
                        loss=criterion, 
                        optimizer=optimizer, 
                        input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]), 
                        nb_classes=num_classes)

                    classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)

                    pred = classifier_stolen.predict(torch.FloatTensor(x_test))
                    preds = np.argmax(pred, axis=1)
                    acc = np.mean(preds == y_test)
                    
                    num_params = sum(p.numel() for p in classifier_stolen.model.parameters())
                    duration = time.time() - start_time  
                    
                    conv_model_results.append((
                        iteration + 1, model_pairing, name, len_steal, acc, 
                        orginal_model_accuracy, num_params, duration))
    return conv_model_results
