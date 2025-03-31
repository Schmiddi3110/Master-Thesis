import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.AeonClassifier import AeonClassifier
from art.attacks.extraction import CopycatCNN, KnockoffNets
from art.estimators.classification import PyTorchClassifier
from attack.CustomKnockoffNets import CustomKnockoffNets
import gc
import numpy as np
from models.VAE import VAE
import optuna
from torch.utils.data import TensorDataset
from art.defences.postprocessor import ReverseSigmoid, ClassLabels, GaussianNoise, HighConfidence, Rounded
from art.defences.preprocessor import GaussianAugmentation, FeatureSqueezing, TotalVarMin


def run_noise_generated_defended_attack(defence, noise_level, N, model_catalogue, X_test0, y_test0, batch_size, num_epochs, learning_rate, latent_dim, device, dataset):
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
        clip_min = pareto_trials[0]['params']['clip_min']
        clip_max = pareto_trials[0]['params']['clip_max']
        if augmentation:
            apply_fit = True
        else:
            apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"GaussianAugmentation - sigma: {sigma}, ratio: {ratio}, clip_values: {clip_min}, {clip_max}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        preprocessor = GaussianAugmentation(sigma, augmentation, ratio, (clip_min, clip_max), apply_fit, apply_predict)
    elif defence == "FeatureSqueezing":
        clip_min = pareto_trials[0]['params']['clip_min']
        clip_max = pareto_trials[0]['params']['clip_max']
        bit_depth = pareto_trials[0]['params']['bit_depth'] 
        apply_fit = pareto_trials[0]['params']['apply_fit']
        apply_predict = pareto_trials[0]['params']['apply_predict']
        print(f"FeatureSqueezing - bit_depth: {bit_depth}, clip_values: {clip_min}, {clip_max}, apply_fit: {apply_fit}, apply_predict: {apply_predict}")
        preprocessor = FeatureSqueezing((clip_min, clip_max), bit_depth, apply_fit, apply_predict)

    
    elif defence == "TotalVarMin":
        clip_min = pareto_trials[0]['params']['clip_min']
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
    

    gen_results = []
    num_classes = len(np.unique(y_test0))

    input_size = X_test0.shape[2]    
    num_channels = X_test0.shape[1]

    for iteration in range(N):
        print(f"Iteration {iteration + 1}/{N}")
        
        for model_pairing in model_catalogue:
            print(f"Running attacks for: {model_pairing}")
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
            
            vae = VAE(input_size=input_size, num_channels=num_channels, latent_dim=latent_dim, num_classes=num_classes).to(device)
            state_dict = torch.load(f"TimeSeriesModels/{dataset}/Gen_Models/VAE_0.3.pth", map_location=device)
            vae.load_state_dict(state_dict)
            vae.eval()
            
            for percentage in [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                len_steal = max(int(len(X_test0) * percentage), num_classes)
                sample_to_generate = int(len_steal * 0.33)
                samples_to_noise = int(len_steal * 0.33)
                real_samples = len_steal - sample_to_generate - samples_to_noise
                
                x_steal, y_steal = [], []
                selected_indices = []
                remaining_indices = list(range(len(X_test0)))
                
                for class_label in range(num_classes):
                    class_indices = np.where(y_test0 == class_label)[0]
                    if len(class_indices) > 0:
                        selected_idx = np.random.choice(class_indices, 1, replace=False)
                        x_steal.append(X_test0[selected_idx].squeeze())
                        y_steal.append(y_test0[selected_idx].squeeze())
                        selected_indices.extend(selected_idx)
                        remaining_indices.remove(selected_idx[0])
                
                additional_real_needed = real_samples - len(selected_indices)
                additional_real_indices = []
                if additional_real_needed > 0:
                    additional_real_indices = np.random.choice(remaining_indices, additional_real_needed, replace=False).tolist()
                    x_steal.extend(X_test0[additional_real_indices].squeeze())
                    y_steal.extend(y_test0[additional_real_indices].squeeze())
                    selected_indices.extend(additional_real_indices)
                    remaining_indices = [idx for idx in remaining_indices if idx not in additional_real_indices]
                
                noise_needed = samples_to_noise - len(x_steal) if samples_to_noise > len(x_steal) else samples_to_noise
                noise_indices = np.random.choice(selected_indices, noise_needed, replace=False).tolist()
                for idx in noise_indices:
                    noisy_sample = add_noise(X_test0[idx], noise_level=noise_level)
                    x_steal.append(noisy_sample.squeeze())
                    y_steal.append(y_test0[idx].squeeze())                
                
                generated_x, generated_y = generate_samples(vae, sample_to_generate, np.unique(y_test0).astype(int), latent_dim, device)
                x_steal.extend(generated_x.squeeze())
                y_steal.extend(generated_y.squeeze())
                
                x_steal = np.array(x_steal)
                y_steal = np.array(y_steal)
                
                x_steal = torch.tensor(x_steal, dtype=torch.float32).numpy()
                y_steal = torch.tensor(y_steal, dtype=torch.float32).numpy()
                
                x_test = np.delete(X_test0, selected_indices, axis=0)
                y_test = np.delete(y_test0, selected_indices, axis=0)           

                
                extraction_attacks = {
                            "Probabilistic CopycatCNN": CopycatCNN(classifier=classifier_protected,
                                                                batch_size_fit=batch_size,
                                                                batch_size_query=batch_size,
                                                                nb_epochs=num_epochs,
                                                                nb_stolen=int(len_steal),   
                                                                use_probability=True),
                            "Argmax CopycatCNN": CopycatCNN(classifier=classifier_protected,
                                                            batch_size_fit=batch_size,
                                                            batch_size_query=batch_size,
                                                            nb_epochs=num_epochs,
                                                            nb_stolen=int(len_steal),
                                                            use_probability=False),                    
                            "Adaptive KnockoffNets": CustomKnockoffNets(classifier=classifier_protected,
                                                                        batch_size_fit=batch_size,
                                                                        batch_size_query=batch_size,
                                                                        nb_epochs=num_epochs,
                                                                        nb_stolen=len_steal,
                                                                        verbose=False,
                                                                        sampling_strategy='adaptive',                                                            
                                                                        use_probability=False),               
                }   
                
                for name, attack in extraction_attacks.items():
                    model_stolen = model_catalogue[model_pairing][1]
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model_stolen.parameters(), lr=learning_rate)
                    
                    classifier_stolen = PyTorchClassifier(
                        model=model_stolen,
                        loss=criterion,
                        optimizer=optimizer,
                        input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]),
                        nb_classes=num_classes
                    )
                    
                    classifier_stolen = attack.extract(x_steal, y_steal, thieved_classifier=classifier_stolen)
                    
                    pred = classifier_stolen.predict(torch.FloatTensor(X_test0))
                    preds = np.argmax(pred, axis=1)
                    acc = np.mean(preds == y_test0)
                    
                    gen_results.append((defence, noise_level, iteration + 1, model_pairing, name, real_samples, sample_to_generate, samples_to_noise, acc, orginal_model_accuracy, protected_acc))
                    
    return gen_results



def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, np.std(data) * noise_level, data.shape) 
    return data + noise



def generate_samples(vae, sample_to_generate, unique_classes, latent_dim, device):
    vae.eval()
    generated_samples_list = []
    generated_labels_list = []   
    
    
    with torch.no_grad():
        for _ in range(sample_to_generate):
            cls = np.random.choice(unique_classes)            
            z = torch.randn(1, latent_dim).to(device)
            class_label = torch.tensor([cls], dtype=torch.long).to(device)
            generated_sample = vae.decode(z, class_label).cpu().numpy()  
            generated_samples_list.append(generated_sample[0])
            generated_labels_list.append(cls)
    
    X_generated = np.array(generated_samples_list)
    y_generated = np.array(generated_labels_list)
    
    
    return X_generated, y_generated


def subset_to_numpy(subset):
    x = subset.tensors[0].numpy()
    y = subset.tensors[1].numpy()
    return x, y

