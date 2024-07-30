import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CustomCNN(nn.Module):
    def __init__(self, layers, img_height):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList()
        input_channels = 3  

        for output_channels in layers:
            self.layers.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            self.layers.append(nn.BatchNorm2d(output_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2))
            input_channels = output_channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels * (img_height // 2**len(layers))**2, 512),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),  
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)  
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        return x

class ModelTrainer:
    def __init__(self, dataset_path, dataset_path2, img_height=150, img_width=150, batch_size=32, epochs=30):
        self.dataset_path = dataset_path
        self.dataset_path2 = dataset_path2
        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.epochs = epochs

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
            transforms.RandomResizedCrop((self.img_height, self.img_width), scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.model = CustomCNN([64, 128, 256], self.img_height)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00001)  
        self.criterion = nn.CrossEntropyLoss()

    def prepare_data(self):
        data = []
        labels = []

        for filename in os.listdir(self.dataset_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append(os.path.join(self.dataset_path, filename))
                labels.append(1)  # mobil

        for filename in os.listdir(self.dataset_path2):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                data.append(os.path.join(self.dataset_path2, filename))
                labels.append(0)  # not mobil

        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2, random_state=42
        )

        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

        train_dataset = CustomDataset(train_data, train_labels, transform=self.transform)
        val_dataset = CustomDataset(val_data, val_labels, transform=self.transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train_and_validate(self):
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_f1_scores = []
        best_val_accuracy = 0.0
        best_val_f1 = 0.0

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            
            train_losses.append(running_loss / len(self.train_loader))

            self.model.eval()
            all_labels = []
            all_predictions = []
            with torch.no_grad():
                val_loss = 0
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    val_loss += self.criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                
                val_losses.append(val_loss / len(self.val_loader))
                val_accuracy = accuracy_score(all_labels, all_predictions)
                val_f1 = f1_score(all_labels, all_predictions, average='weighted')
                val_accuracies.append(val_accuracy)
                val_f1_scores.append(val_f1)

               
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save(self.model.state_dict(), 'best_model_state_dict2.pth')
                    print(f'New best model saved with accuracy: {val_accuracy:.4f}')

                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    torch.save(self.model.state_dict(), 'best_model_state_dict_f12.pth')
                    print(f'New best model saved with F1 score: {val_f1:.4f}')

            print(f'Epoch {epoch + 1}/{self.epochs} - Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}, Validation F1 Score: {val_f1:.4f}')

    def train(self):
        self.prepare_data()
        self.train_and_validate()

