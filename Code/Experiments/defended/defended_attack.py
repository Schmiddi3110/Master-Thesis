import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from art.estimators.classification import PyTorchClassifier
from models.AeonClassifier import AeonClassifier
from attack.CustomKnockoffNets import CustomKnockoffNets
from art.defences.postprocessor import ReverseSigmoid, ClassLabels, GaussianNoise, HighConfidence, Rounded
from art.defences.preprocessor import GaussianAugmentation, FeatureSqueezing, TotalVarMin
from misc.model_utils import load_trained_model, load_untrained_model, get_small_conv_architectures_catalogue
import os
from art.attacks.extraction import CopycatCNN
import optuna


def run_defended_attack(dataset, defence, model_catalogue, N,  X_test0, y_test0, num_classes, num_epochs, learning_rate, batch_size, device):
    results_protected = []
    if defence not in ["ReverseSigmoid", "ClassLabels", "GaussianNoise", "HighConfidence", "Rounded", "GaussianAugmentation", "FeatureSqueezing", "TotalVarMin"]:
        raise ValueError("Unknown Defense")

    if defence in ["GaussianAugmentation", "FeatureSqueezing", "TotalVarMin"]:
        storage_path = f"TimeSeriesModels/{dataset}/Hyperparameter/PreProcessing.db"
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Storage path {storage_path} does not exist. Run Script with --search_hyperparameter True and --num_trials > 1") 
        
        study = optuna.load_study(
            study_name=defence, 
            storage=f"sqlite:///{storage_path}"
        )
    else:
        storage_path = f"TimeSeriesModels/{dataset}/Hyperparameter/PostProcessing.db"
        if not os.path.exists(storage_path):
            raise FileNotFoundError(f"Storage path {storage_path} does not exist. Run Script with --search_hyperparameter True and --num_trials > 1")
        
        study = optuna.load_study(
            study_name=defence, 
            storage=f"sqlite:///{storage_path}"
        )
    

    pareto_trials = []
    for trial in study.best_trials:
        mean_protected_acc_diff = trial.values[1]  
        mean_acc_stolen = trial.values[0] 

        params = {key: trial.params.get(key, "N/A") for key in trial.params}

        pareto_trials.append({
            "trial_number": trial.number,
            "params": params,
            "mean_protected_acc_diff": mean_protected_acc_diff,
            "mean_acc_stolen": mean_acc_stolen,
        })

    pareto_trials = sorted(
        pareto_trials, 
        key=lambda x: (x["mean_protected_acc_diff"], x["mean_acc_stolen"])
    )
    


    if defence == "ReverseSigmoid":
        beta = pareto_trials[0]['params']['beta'] *1.2
        gamma = pareto_trials[0]['params']['gamma'] *1.2
        print(f"ReverseSigmoid - beta: {beta}, gamma: {gamma}")
        postprocessor = ReverseSigmoid(beta=beta, gamma=gamma)
    elif defence == "ClassLabels":
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"ClassLabels - apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        postprocessor = ClassLabels(apply_fit, apply_predict)
    elif defence == "GaussianNoise":
        scale = pareto_trials[0]['params']['scale'] 
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"GaussianNoise - scale: {scale}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        postprocessor = GaussianNoise(scale, apply_fit, apply_predict)
    elif defence == "HighConfidence":
        cutoff = pareto_trials[0]['params']['cutoff'] 
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"HighConfidence - cutoff: {cutoff}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        postprocessor = HighConfidence(cutoff, apply_fit, apply_predict)
    
    elif defence == "Rounded":
        decimals = pareto_trials[0]['params']['decimals']
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        postprocessor = Rounded(decimals, apply_fit, apply_predict)
    elif defence == "GaussianAugmentation":
        sigma = pareto_trials[0]['params']['sigma']
        augmentation = pareto_trials[0]['params']['augmentation']
        ratio = pareto_trials[0]['params']['ratio']
        clip_min = max(pareto_trials[0]['params']['clip_min'], 0)
        clip_max = pareto_trials[0]['params']['clip_max']
        if augmentation:
            apply_fit = True
        else:
            apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"GaussianAugmentation - sigma: {sigma}, ratio: {ratio}, clip_values: {clip_min}, {clip_max}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        preprocessor = GaussianAugmentation(sigma, augmentation, ratio, (clip_min, clip_max), apply_fit, apply_predict)
    elif defence == "FeatureSqueezing":
        clip_min = max(pareto_trials[0]['params']['clip_min'], 0)
        clip_max = pareto_trials[0]['params']['clip_max']
        bit_depth = pareto_trials[0]['params']['bit_depth'] 
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"FeatureSqueezing - bit_depth: {bit_depth}, clip_values: {clip_min}, {clip_max}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        preprocessor = FeatureSqueezing((clip_min, clip_max), bit_depth, apply_fit, apply_predict)

    
    elif defence == "TotalVarMin":
        clip_min = max(pareto_trials[0]['params']['clip_min'], 0)
        clip_max = pareto_trials[0]['params']['clip_max']
        norm = pareto_trials[0]['params']['norm']
        solver = pareto_trials[0]['params']['solver']
        lamb = pareto_trials[0]['params']['lamb']
        max_iter = pareto_trials[0]['params']['max_iter']
        prob = pareto_trials[0]['params']['prob']
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        preprocessor = TotalVarMin(prob, norm, lamb, solver, max_iter, (clip_min, clip_max), apply_fit, apply_predict)       
               
    else:
        raise ValueError("Unknown Defense")

    for iteration in range(N):
        print(f"Iteration {iteration + 1}/{N}")
        for model_pairing in model_catalogue:
            print(model_pairing)
            model, orginal_model_accuracy = model_catalogue[model_pairing][0]
            criterion = nn.CrossEntropyLoss()
            if defence in ["GaussianAugmentation", "FeatureSqueezing", "TotalVarMin"]:  
                if model_catalogue[model_pairing][2] == "aeon":
                    classifier_protected = AeonClassifier(
                        model=model,
                        nb_classes=num_classes,
                        input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]),   
                        preprocessing_defences=preprocessor
                    )              
                else:
                    optimizer_original = optim.Adam(model.parameters(), lr=learning_rate)
                    classifier_protected = PyTorchClassifier(
                    model=model, 
                    loss=criterion, 
                    optimizer=optimizer_original, 
                    input_shape=(X_test0.shape[1], X_test0.shape[2], batch_size), 
                    nb_classes=num_classes, 
                    preprocessing_defences=preprocessor, 
                    device_type=device
                )
            else:
                if model_catalogue[model_pairing][2] == "aeon":
                    classifier_protected = AeonClassifier(
                        model=model,
                        nb_classes=num_classes,
                        input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]),   
                        postprocessing_defences=postprocessor
                    )
                else:
                    optimizer_original = optim.Adam(model.parameters(), lr=learning_rate)
                    classifier_protected = PyTorchClassifier(
                        model=model, 
                        loss=criterion, 
                        optimizer=optimizer_original, 
                        input_shape=(X_test0.shape[1], X_test0.shape[2], batch_size), 
                        nb_classes=num_classes, 
                        postprocessing_defences=postprocessor, 
                        device_type=device
                    )
            
            pred = classifier_protected.predict(torch.FloatTensor(X_test0))

            preds = np.argmax(pred, axis=1)
            protected_acc = np.mean(preds == y_test0)            

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
                    "Probabilistic CopycatCNN (vs. Protected)": CopycatCNN(
                        classifier=classifier_protected,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=int(len_steal),
                        use_probability=True
                    ),
                    "Argmax CopycatCNN (vs. Protected)": CopycatCNN(
                        classifier=classifier_protected,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=int(len_steal),
                        use_probability=False
                    ),
                    "KnockoffNets (vs. Protected)": CustomKnockoffNets(
                        classifier=classifier_protected,
                        batch_size_fit=batch_size,
                        batch_size_query=batch_size,
                        nb_epochs=num_epochs,
                        nb_stolen=int(len_steal),
                        reward='all',
                        verbose=False,
                        sampling_strategy="adaptive",
                        
                    )        
                }

                for name, attack in extraction_attacks.items():
                    model_stolen = model_catalogue[model_pairing][1]
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model_stolen.parameters(), lr=learning_rate)

                    classifier_stolen = PyTorchClassifier(
                        model=model_stolen, 
                        loss=criterion, 
                        optimizer=optimizer, 
                        input_shape=(X_test0.shape[1], X_test0.shape[2], batch_size), 
                        nb_classes=num_classes, 
                        device_type=device
                    )

                    classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)

                    pred = classifier_stolen.predict(torch.FloatTensor(x_test))
                    preds = np.argmax(pred, axis=1)
                    acc = np.mean(preds == y_test)

                    results_protected.append((iteration +1,defence, model_pairing, name, len_steal, acc, orginal_model_accuracy, protected_acc))

            del classifier_protected
    return results_protected

def objective(trial, defence, dataset, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, X_test0, y_test0, device, batch_size):    

    if defence == "ReverseSigmoid":
        beta = trial.suggest_float("beta", 0.8, 1.2, log=True)
        gamma = trial.suggest_float("gamma", 0.01, 0.3, log=True)
        postprocessor = ReverseSigmoid(beta=beta, gamma=gamma)
    elif defence == "ClassLabels":
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])
        postprocessor = ClassLabels(apply_fit, apply_predict)
    elif defence == "Rounded":
        decimals = trial.suggest_int("decimals", 1, 5)
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])
        postprocessor = Rounded(decimals, apply_fit, apply_predict)
    elif defence == "GaussianNoise":
        scale = trial.suggest_float("scale", 0, 0.4)
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])
        postprocessor = GaussianNoise(scale, apply_fit, apply_predict)
    elif defence == "HighConfidence":
        cutoff = trial.suggest_float("cutoff", 0, 0.5)
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])
        postprocessor = HighConfidence(cutoff, apply_fit, apply_predict)    
    elif defence == "FeatureSqueezing":
        clip_min = trial.suggest_float("clip_min", X_test0.min(), 0)
        clip_max = trial.suggest_float("clip_max", 0, X_test0.max())
        bit_depth = trial.suggest_int("bit_depth", 1, 64)
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])
        preprocessor = FeatureSqueezing((clip_min, clip_max), bit_depth, apply_fit, apply_predict)        
    elif defence == "GaussianAugmentation":
        sigma = trial.suggest_float("sigma", 0, 1)
        augmentation = trial.suggest_categorical("augmentation", [True, False])
        ratio = trial.suggest_float("ratio", 0, 1)
        clip_min = trial.suggest_float("clip_min", X_test0.min(), 0)
        clip_max = trial.suggest_float("clip_max", 0, X_test0.max())
        
        if augmentation:
            apply_fit = True 
        else:
            apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        
        apply_predict = trial.suggest_categorical("apply_predict", [False])
        
        preprocessor = GaussianAugmentation(sigma, augmentation, ratio, (clip_min, clip_max), apply_fit, apply_predict)
        

    elif defence == "TotalVarMin":
        clip_min = trial.suggest_float("clip_min", X_test0.min(), 0)
        clip_max = trial.suggest_float("clip_max", 0, X_test0.max())
        norm = trial.suggest_int("norm", 1, 5)
        solver = trial.suggest_categorical("solver", ["L-BFGS-B", "CG", "Newton-CG"])
        lamb = trial.suggest_float("lamb", 0.1, 0.7)
        max_iter = trial.suggest_int("max_iter", 1, 15)
        prob = trial.suggest_float("prob", 0.1, 0.5)
        apply_fit = trial.suggest_categorical("apply_fit", [True, False])
        apply_predict = trial.suggest_categorical("apply_predict", [True, False])   
        apply_predict = False
        preprocessor = TotalVarMin(prob, norm, lamb, solver, max_iter, (clip_min, clip_max), apply_fit, apply_predict)      

    
    else:
        raise ValueError("Unknown Defense")
        
    num_epochs=100
    results_protected = []
    model_catalogue = get_small_conv_architectures_catalogue(
        dataset, X_test0, num_classes, num_channels,
        num_epochs, learning_rate, patience,
        input_size, hidden_size, num_layers, device
    )
    for model_pairing in model_catalogue:
        
        model, orginal_model_accuracy = model_catalogue[model_pairing][0]
        criterion = nn.CrossEntropyLoss()
        optimizer_original = optim.Adam(model.parameters(), lr=learning_rate)
        if defence in ["GaussianAugmentation", "FeatureSqueezing", "TotalVarMin"]:                
            classifier_protected = PyTorchClassifier(
                model=model, 
                loss=criterion, 
                optimizer=optimizer_original, 
                input_shape=(X_test0.shape[1], X_test0.shape[2], batch_size), 
                nb_classes=num_classes, 
                preprocessing_defences=preprocessor, 
                device_type=device
            )
        else:
            classifier_protected = PyTorchClassifier(
                model=model, 
                loss=criterion, 
                optimizer=optimizer_original, 
                input_shape=(X_test0.shape[1], X_test0.shape[2], batch_size), 
                nb_classes=num_classes, 
                postprocessing_defences=postprocessor, 
                device_type=device
            )        
        pred = classifier_protected.predict(torch.FloatTensor(X_test0))

        preds = np.argmax(pred, axis=1)
        
        protected_acc = np.mean(preds == y_test0)
        model_protected_diff = orginal_model_accuracy - protected_acc
        

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


            extraction_attacks = {"Probabilistic CopycatCNN (vs. Protected)": CopycatCNN(classifier=classifier_protected,
                                                        batch_size_fit=batch_size,
                                                        batch_size_query=batch_size,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=int(len_steal),   
                                                        use_probability=True),
                                "Argmax CopycatCNN (vs. Protected)": CopycatCNN(classifier=classifier_protected,
                                                        batch_size_fit=batch_size,
                                                        batch_size_query=batch_size,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=int(len_steal),
                                                        use_probability=False),
                                "KnockoffNets (vs. Protected)": CustomKnockoffNets(
                                                        classifier=classifier_protected,
                                                        batch_size_fit=batch_size,
                                                        batch_size_query=batch_size,
                                                        nb_epochs=num_epochs,
                                                        nb_stolen=int(len_steal),
                                                        reward='all',
                                                        verbose=False,
                                                        sampling_strategy="adaptive",                    
                                                        )  
                                }
          
            for name, attack in extraction_attacks.items():
                model_stolen = model_catalogue[model_pairing][1]
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model_stolen.parameters(), lr=learning_rate)

                classifier_stolen = PyTorchClassifier(model=model_stolen, loss=criterion, optimizer=optimizer, input_shape=(batch_size,X_test0.shape[1], X_test0.shape[2]), nb_classes=num_classes, device_type=device)
                
                classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)

                pred = classifier_stolen.predict(torch.FloatTensor(x_test))
                
                preds = np.argmax(pred, axis=1)
                acc = np.mean(preds == y_test)
                
                results_protected.append((model_pairing, name, len_steal, acc, orginal_model_accuracy, model_protected_diff))
    mean_stolen_acc = np.mean([result[3] for result in results_protected]) 
    mean_protected_acc_diff = np.mean([result[5] for result in results_protected])
    return mean_stolen_acc, mean_protected_acc_diff

def run_hyperparameter_search(defence, dataset, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, X_test0, y_test0, device, num_trials, batch_size):    
    if defence in ["GaussianAugmentation", "FeatureSqueezing", "TotalVarMin"]:
        study = optuna.create_study(directions=["minimize", "minimize"], study_name=defence, storage=f"sqlite:///TimeSeriesModels/{dataset}/Hyperparameter/PreProcessing.db", load_if_exists=True)
    else:
        study = optuna.create_study(directions=["minimize", "minimize"], study_name=defence, storage=f"sqlite:///TimeSeriesModels/{dataset}/Hyperparameter/PostProcessing.db", load_if_exists=True)
        
    study.optimize(lambda trial: objective(trial, defence, dataset, num_classes, num_channels, num_epochs, learning_rate, patience, input_size, hidden_size, num_layers, X_test0, y_test0, device, batch_size), n_trials=num_trials)



