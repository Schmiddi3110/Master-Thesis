import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class BaseCNN(nn.Module):
    def __init__(self, num_classes, num_epochs=100, learning_rate=0.001, patience=10):
        super(BaseCNN, self).__init__()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def forward(self, x):
        raise NotImplementedError("Subclasses must implement this method")
    
    def weights_init(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.xavier_uniform_(m.in_proj_weight)
            nn.init.zeros_(m.in_proj_bias)
            nn.init.xavier_uniform_(m.out_proj.weight)
            if m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)

    def train_model(self, train_loader, val_loader, save_path):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.num_epochs):
            self.train()
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for inputs, labels in train_loader:                
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            avg_loss = running_loss / len(train_loader)
            accuracy = correct_predictions / total_samples * 100

            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            if (epoch + 1) % 5 == 0:
                self.eval()
                val_loss = 0.0
                val_correct_predictions = 0
                val_total_samples = 0

                with torch.no_grad():
                    for val_inputs, val_labels in val_loader:
                        val_inputs = val_inputs.to(self.device)
                        val_labels = val_labels.to(self.device)

                        val_outputs = self(val_inputs)
                        val_loss += criterion(val_outputs, val_labels).item()

                        _, val_predicted = torch.max(val_outputs.data, 1)
                        val_total_samples += val_labels.size(0)
                        val_correct_predictions += (val_predicted == val_labels).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = val_correct_predictions / val_total_samples * 100
                print(f'Validation at Epoch [{epoch + 1}/{self.num_epochs}]: Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0 
                torch.save(self.state_dict(), f'{save_path}/{self.__class__.__name__}.pth')
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print("Early stopping triggered. Training stopped.")
                break



class BaseCNN1D(BaseCNN):
    def __init__(self, num_classes,in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(BaseCNN1D, self).__init__(num_classes, num_epochs, learning_rate, patience)
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3).to(self.device)
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = None  

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))

        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
            self.fc1 = nn.Linear(flattened_size, self.num_classes).to(self.device)

        x = x.view(x.size(0), -1) 
        
        x = self.fc1(x)
        return F.softmax(x, dim=1)


class Deep1DCNN(BaseCNN):
    def __init__(self, num_classes, in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(Deep1DCNN, self).__init__(num_classes, num_epochs, learning_rate, patience)
        
        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, padding=1)  
        self.conv3 = nn.Conv1d(16, 32, kernel_size=3, padding=1) 
        
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.fc1 = None 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        if x.shape[-1] > 1:
            x = self.pool(F.relu(self.conv3(x)))  
        else:
            x = F.relu(self.conv3(x))  

        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
        
            self.fc1 = nn.Linear(flattened_size, self.num_classes).to(self.device)

        x = x.view(x.size(0), -1)  
        
        x =  self.fc1(x)
        return F.softmax(x, dim=1)



class Simple1DCNN(BaseCNN):
    def __init__(self, num_classes, in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(Simple1DCNN, self).__init__(num_classes, num_epochs, learning_rate, patience)
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=8, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = None


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
            self.fc1 = nn.Linear(flattened_size, self.num_classes).to(self.device)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return F.softmax(x, dim=1)




class Residual1DCNN(BaseCNN):
    def __init__(self, num_classes, in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(Residual1DCNN, self).__init__(num_classes, num_epochs, learning_rate, patience)
        
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3, padding=1)

        self.downsample = nn.Conv1d(in_channels, 16, kernel_size=1)  

        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        identity = self.downsample(x)  
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) + identity  
        x = self.pool(F.relu(self.conv3(x)))
        
        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
            self.fc1 = nn.Linear(flattened_size, 128).to(self.device)

        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        
        return F.softmax(self.fc2(x), dim=1)


class Attention1DCNN(BaseCNN):
    def __init__(self, num_classes, in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(Attention1DCNN, self).__init__(num_classes, num_epochs, learning_rate, patience)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=2)
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = x.permute(2, 0, 1)  
        x, _ = self.attn(x, x, x)
        x = x.permute(1, 2, 0) 

        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)
            self.fc1 = nn.Linear(flattened_size, 128).to(self.device)

        x = x.contiguous().view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class Dilated1DCNN(BaseCNN):
    def __init__(self, num_classes, in_channels, num_epochs=100, learning_rate=0.001, patience=10):
        super(Dilated1DCNN, self).__init__(num_classes, num_epochs, learning_rate, patience)
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, dilation=1, padding=1)  
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, dilation=2, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, dilation=4, padding=4)
        self.pool = nn.AdaptiveAvgPool1d(output_size=10)  
        self.fc1 = None
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))

        if self.fc1 is None:
            flattened_size = x.size(1) * x.size(2)            
            self.fc1 = nn.Linear(flattened_size, 128).to(self.device)

        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


