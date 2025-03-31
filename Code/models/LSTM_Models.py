import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BaseLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, learning_rate, patience, num_epochs):
        super(BaseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.patience = patience
        self.num_epochs = num_epochs         
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
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
    
    
    
class BaseLSTM1D(BaseLSTM):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, learning_rate, patience, num_epochs):
        super(BaseLSTM1D, self).__init__(input_size, hidden_size, num_layers, num_classes, learning_rate, patience, num_epochs)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        
        out, _ = self.lstm(x, (h0, c0))  
        out = out[:, -1, :]  
                
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out
