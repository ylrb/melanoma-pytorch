import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from PIL import Image
from datetime import datetime
import time
import numpy as np
from efficientnet_pytorch import EfficientNet
from RandAugment import RandAugment
from torch.optim import AdamW

# Définit le dataset à partir des fichiers fournis
class CustomDataset(Dataset):
    def __init__(self, data_folder, dataframe, transform=None):
        self.data_folder = data_folder
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_folder, self.dataframe.iloc[idx, 0] + ".jpg")
        image = Image.open(img_name)
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

print(datetime.now())

# Vérifie si un GPU est disponible, et définit la variable device en conséquence
print("Running on GPU:", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Liens des images et des labels
data_folder = "cvml-pytorch-main/isic-2020-resized/train-resized"
label_file = "cvml-pytorch-main/isic-2020-resized/train-labels.csv"

# Initialise le modèle utilisé pour le fine-tuning et modifie le dernier fully connected pour notre classification binaire
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
model = model.to(device)

# Hyperparamètres de l'entraînement
batch_size = 16
lr = 0.001
l2_reg = 0.0001
num_epochs = 20

# Fonction de coût et optimiseur utilisés
#optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=l2_reg)
#optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=l2_reg, momentum=0.9)

# Charge les labels et les divise en un dataset d'entraînement et un dataset de validation (90% entraînement, 10% validation)
labels = pd.read_csv(label_file)
train_dataframe, val_dataframe = train_test_split(labels, test_size=0.1, random_state=42)

# Définit les transormations appliquées aux images
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

N = 3
M = np.random.poisson(12)/30
#transform.transforms.insert(0, RandAugment(N, M))

# Crée les deux datasets à partir des deux dataframes, en appliquant les transformations aux images
train_dataset = CustomDataset(data_folder, train_dataframe, transform)
val_dataset = CustomDataset(data_folder, val_dataframe, transform)

# Calcule les poids des deux classes puis crée un sampler avec WeightedRandomSampler
labels_weights = 1.0 / train_dataframe['target'].value_counts()
class_weights = labels_weights[train_dataframe['target']]
sampler = WeightedRandomSampler(torch.FloatTensor(class_weights.values), len(train_dataset))

#criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 30.0]).to(device))
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device))

# Boucle d'entraînement
print("Starting")
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_class0, correct_class1 = 0, 0
    total_class0, total_class1 = 0, 0
    correct_total = 0
    total_total = 0

    # Traitement d'un minibatch, en rendomizant le dataloader à chaque passage
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calcule l'accuracy globale et pour les deux classes séparément
        _, predicted = torch.max(outputs, 1)
        for i in range(len(predicted)):
            if labels[i] == 0:
                total_class0 += 1
                correct_class0 += (predicted[i] == labels[i]).item()
            elif labels[i] == 1:
                total_class1 += 1
                correct_class1 += (predicted[i] == labels[i]).item()

        total_total += labels.size(0)
        correct_total += (predicted == labels).sum().item()

    accuracy = correct_total / total_total * 100
    accuracy_class0 = correct_class0 / total_class0 * 100
    accuracy_class1 = correct_class1 / total_class1 * 100

    print(f"Epoch: {epoch + 1}/{num_epochs} - Loss: {total_loss/len(train_loader):.4f} - Time: {format_time(time.time() - start_time)} - Acc.: {accuracy:.2f}% (0: {accuracy_class0:.2f}%, 1: {accuracy_class1:.2f}%)")

    # Boucle de validation
    model.eval()
    total_val_loss = 0.0
    correct_val = 0
    correct_val_class0 = 0
    correct_val_class1 = 0
    total_val = 0
    total_val_class0 = 0
    total_val_class1 = 0

    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            total_val_loss += val_loss.item()

            # Calcule l'accuracy globale et pour les deux classes séparément
            _, predicted = torch.max(val_outputs, 1)
            for i in range(len(predicted)):
                if labels[i] == 0:
                    total_val_class0 += 1
                    correct_val_class0 += (predicted[i] == labels[i]).item()
                elif labels[i] == 1:
                    total_val_class1 += 1
                    correct_val_class1 += (predicted[i] == labels[i]).item()

            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

        val_accuracy = correct_val / total_val * 100
        val_accuracy_class0 = correct_val_class0 / total_val_class0 * 100
        val_accuracy_class1 = correct_val_class1 / total_val_class1 * 100

        print(f"Validation Accuracy: {val_accuracy:.2f}% (0: {val_accuracy_class0:.2f}%, 1: {val_accuracy_class1:.2f}%)")


# Sauvegarde le modèle
torch.save(model.state_dict(), 'resnet18-'+str(datetime.now())+'.pth')

print("Training completed!")
