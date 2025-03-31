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

def run_pure_generated_data_experiment(real_percentage, N, conv_architectures_catalogue, X_test0, y_test0, batch_size, num_epochs, learning_rate, latent_dim, device, dataset):
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
                len_steal = max(int(len(X_test0) * percentage), num_classes)                

                steal_dataset = create_generated_dataset(vae, len_steal, num_classes, latent_dim, device)
                x_steal, y_steal = subset_to_numpy(steal_dataset)
                
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
                    gen_results.append((real_percentage, iteration + 1, model_pairing, name, len_steal, acc, orginal_model_accuracy))
                    
    return gen_results



def create_generated_dataset(vae, sample_to_generate, num_classes, latent_dim, device):    
    vae.eval()
    generated_samples_list = []
    generated_labels_list = []
    
    unique_classes = np.arange(num_classes)
    
    with torch.no_grad():
        for cls in unique_classes:
            z = torch.randn(1, latent_dim).to(device)
            class_label = torch.tensor([cls], dtype=torch.long).to(device)
            generated_sample = vae.decode(z, class_label).cpu().numpy()
            generated_samples_list.append(generated_sample[0])
            generated_labels_list.append(cls)
        
        for _ in range(sample_to_generate - num_classes):
            cls = np.random.choice(unique_classes)
            z = torch.randn(1, latent_dim).to(device)
            class_label = torch.tensor([cls], dtype=torch.long).to(device)
            generated_sample = vae.decode(z, class_label).cpu().numpy()
            generated_samples_list.append(generated_sample[0])
            generated_labels_list.append(cls)
    
    X_generated = np.array(generated_samples_list)
    y_generated = np.array(generated_labels_list)  
    
    combined_dataset = TensorDataset(torch.tensor(X_generated, dtype=torch.float32),
                                     torch.tensor(y_generated, dtype=torch.long))
    
    return combined_dataset