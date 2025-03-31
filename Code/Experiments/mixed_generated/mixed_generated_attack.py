import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from art.attacks.extraction import CopycatCNN, KnockoffNets
from models.AeonClassifier import AeonClassifier
from models.VAE import VAE
from torch.utils.data import TensorDataset
from attack.CustomKnockoffNets import CustomKnockoffNets


def subset_to_numpy(subset):
    x = subset.tensors[0].numpy()
    y = subset.tensors[1].numpy()
    return x, y


def run_mixed_generated_data_experiment(real_percentage, N, conv_architectures_catalogue, X_test0, y_test0, batch_size, num_epochs, learning_rate, latent_dim, device, dataset):
    gen_results = []
    num_classes = len(np.unique(y_test0))

    input_size = X_test0.shape[2]     
    num_channels = X_test0.shape[1]

    for iteration in range(N):
        print(f"Iteration {iteration + 1}/{N}")
        
        for model_pairing in conv_architectures_catalogue:
            print(f"Running attacks for: {model_pairing}")
            model, orginal_model_accuracy = conv_architectures_catalogue[model_pairing][0]
            criterion = nn.CrossEntropyLoss()
            
            if conv_architectures_catalogue[model_pairing][2] == "aeon":
                classifier_original = AeonClassifier(model=model, nb_classes=num_classes, input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]))
            elif conv_architectures_catalogue[model_pairing][2] == "torch":
                optimizer_original = optim.Adam(model.parameters(), lr=learning_rate)
                classifier_original = PyTorchClassifier(model=model, loss=criterion, optimizer=optimizer_original, input_shape=(batch_size, X_test0.shape[1], X_test0.shape[2]), nb_classes=num_classes)
            else:
                raise ValueError(f"Unrecognized Classifier: {conv_architectures_catalogue[model_pairing][2]}")       
          

            vae = VAE(input_size=input_size, num_channels=num_channels, latent_dim=latent_dim, num_classes=num_classes).to(device)
            state_dict = torch.load(f"TimeSeriesModels/{dataset}/Gen_Models/VAE_{real_percentage}.pth", map_location=device)
            vae.load_state_dict(state_dict)
            vae.eval()
            
            for percentage in [0.001, 0.005, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
                len_steal = max(int(len(X_test0) * percentage), 1)
                real_samples = len_steal - int(len_steal * real_percentage)
                sample_to_generate = len_steal - real_samples
                
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


                additional_needed = len_steal - len(selected_indices) - sample_to_generate
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
                
                combined_dataset = create_combined_dataset(vae, sample_to_generate, x_steal, y_steal, latent_dim, device)
                x_steal, y_steal = subset_to_numpy(combined_dataset)
                
                extraction_attacks = {
                            "Probabilistic CopycatCNN": CopycatCNN(classifier=classifier_original,
                                                                batch_size_fit=batch_size,
                                                                batch_size_query=batch_size,
                                                                nb_epochs=num_epochs,
                                                                nb_stolen=int(len_steal),  
                                                                use_probability=True),
                            "Argmax CopycatCNN": CopycatCNN(classifier=classifier_original,
                                                            batch_size_fit=batch_size,
                                                            batch_size_query=batch_size,
                                                            nb_epochs=num_epochs,
                                                            nb_stolen=int(len_steal),
                                                            use_probability=False),                    
                            "Adaptive KnockoffNets": CustomKnockoffNets(classifier=classifier_original,
                                                                        batch_size_fit=batch_size,
                                                                        batch_size_query=batch_size,
                                                                        nb_epochs=num_epochs,
                                                                        nb_stolen=len_steal,
                                                                        verbose=False,
                                                                        sampling_strategy='adaptive',                                                            
                                                                        use_probability=False),               
                }   
                
                for name, attack in extraction_attacks.items():
                    model_stolen = conv_architectures_catalogue[model_pairing][1]
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
                    
                    gen_results.append((real_percentage, iteration + 1, model_pairing, name, real_samples, sample_to_generate, acc, orginal_model_accuracy))
                    
    return gen_results


def create_combined_dataset(vae, sample_to_generate, X_real, y_real, latent_dim, device):
    if sample_to_generate == 0:
        return TensorDataset(torch.tensor(X_real, dtype=torch.float32), torch.tensor(y_real, dtype=torch.long))
    
    vae.eval()
    generated_samples_list = []
    generated_labels_list = []
    
    unique_classes = np.unique(y_real).astype(int)
    
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
    
    X_combined = np.concatenate([X_real, X_generated], axis=0)
    y_combined = np.concatenate([y_real, y_generated], axis=0)
    
    combined_dataset = TensorDataset(torch.tensor(X_combined, dtype=torch.float32),
                                     torch.tensor(y_combined, dtype=torch.long))
    
    return combined_dataset