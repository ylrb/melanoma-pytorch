import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

NAME = 'EfficientNet'

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name_with_extension = self.image_list[idx]
        img_name = os.path.splitext(img_name_with_extension)[0]  # Remove the extension
        img_path = os.path.join(self.root_dir, img_name_with_extension)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return image, img_name

model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(1280, 2)
model.load_state_dict(torch.load(NAME+'.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset(root_dir='isic-2020-resized/test-resized', transform=transform)
data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

probabilities = []
image_names = []
with torch.no_grad():
    for inputs, names in tqdm(data_loader, desc='Predicting'):
        inputs = inputs.to(device)
        outputs = model(inputs)
        probabilities_batch = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        probabilities.extend(probabilities_batch)
        image_names.extend(names)

df_predictions = pd.DataFrame({'image_name': image_names, 'target': probabilities})
df_predictions.to_csv(NAME+'-predictions.csv', index=False)
