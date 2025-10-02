import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import kagglehub

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



# Data Loading and Preprocessing
print("\n--- Loading and Preprocessing Data ---")
dataset_path = kagglehub.dataset_download("ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes")
print(f"Using dataset from: {dataset_path}")

class_map = {
    'Glioma': 0,
    'Meningioma': 1,
    'No Tumor': 2,
    'Pituitary': 3
}
all_images = []
all_labels = []
extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.png']
data_folders_to_scan = [os.path.join(dataset_path, 'Train'), os.path.join(dataset_path, 'Val')]

for folder in data_folders_to_scan:
    if not os.path.exists(folder):
        continue

    for class_name, label_idx in class_map.items():
        image_list = []
        class_path = os.path.join(folder, class_name, 'images')

        if not os.path.exists(class_path):
            continue

        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(class_path, ext)))

        for file_path in files:
            img_bgr = cv2.imread(file_path)
            if img_bgr is not None:
                img_resized = cv2.resize(img_bgr, (128, 128))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                image_list.append(img_rgb)

        all_images.extend(image_list)
        all_labels.extend([label_idx] * len(image_list))
        print(f"Loaded {len(image_list)} images for class '{class_name}' from {os.path.basename(folder)}.")
all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f"\nFinal shape of all images: {all_images.shape}")
print(f"\nFinal shape of all labels: {all_labels.shape}")


# Custom Pytorch Dataset Class
class MRIDataset(Dataset):
    def __init__(self, images, labels):
    	# Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        # Normalize data and convert to tensors
        self.X_train = torch.tensor(X_train / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.X_val = torch.tensor(X_val / 255.0, dtype=torch.float32).permute(0, 3, 1, 2)
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.mode = 'train'

    def __len__(self):
        return len(self.X_train) if self.mode == 'train' else len(self.X_val)
            
    def __getitem__(self, index):
        if self.mode == 'train':
            image = self.X_train[index]
            label = self.y_train[index]
        else:
            image = self.X_val[index]
            label = self.y_val[index]
        
        return image, label


# Create Dataset and Dataloaders
print("\n--- Creating Dataset and DataLoaders ---")
mri_dataset = MRIDataset(images=all_images, labels=all_labels)

mri_dataset.mode = 'train'
train_loader = DataLoader(mri_dataset, batch_size=32, shuffle=True)
print(f"Number of training samples: {len(mri_dataset)}")

mri_dataset.mode = 'val'
val_loader = DataLoader(mri_dataset, batch_size=32, shuffle=False)
print(f"Number of validation samples: {len(mri_dataset)}")


# CNN Model Definition
class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),# W + 2P - K /S + 1
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        with torch.no_grad():
            num_features = self.cnn_model(torch.randn(1, 3, 128, 128)).flatten(1).shape[1]
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes)
        )
    def forward(self, x):
        x = self.cnn_model(x)
        x = torch.flatten(x, 1)
        x = self.fc_model(x)
        return x


# Training & Validation Loop
print("\n--- Starting Model Training ---")
model = CNN(num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()
EPOCHS = 50

epoch_train_losses = []
epoch_val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for data, label in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        y_hat = model(data)
        loss = criterion(y_hat, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    epoch_train_losses.append(np.mean(train_losses))

    model.eval()
    val_losses = []
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(device), label.to(device)
            y_hat = model(data)
            loss = criterion(y_hat, label)
            val_losses.append(loss.item())
    epoch_val_losses.append(np.mean(val_losses))

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/{EPOCHS}\tTrain Loss: {np.mean(train_losses):.6f}\tValidation Loss: {np.mean(val_losses):.6f}')

# Final Evaluation & Results
print("\n--- Final Model Evaluation ---")
model.eval()
outputs = []
y_true = []
with torch.no_grad():
    for images_batch, labels_batch in val_loader:
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)
        y_hat = model(images_batch)
        predictions = torch.argmax(y_hat, dim=1)
        outputs.append(predictions.cpu().numpy())
        y_true.append(labels_batch.cpu().numpy())

outputs = np.concatenate(outputs)
y_true = np.concatenate(y_true)


accuracy = accuracy_score(y_true, outputs)
print(f"Final Model Accuracy on Validation Set: {accuracy:.4f}")
if not os.path.exists('results'):
    os.makedirs('results')

# Plot Confusion Matrix
class_names = list(class_map.keys())
cm = confusion_matrix(y_true, outputs)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels'); plt.ylabel('True Labels'); plt.title('Confusion Matrix')
plt.savefig('results/confusion_matrix.png', dpi=300)
plt.show()
plt.close()

# Plot Loss Curves
plt.figure(figsize=(12, 6))
plt.plot(epoch_train_losses, c='b', label='Train loss')
plt.plot(epoch_val_losses, c='r', label='Validation loss')
plt.legend(); plt.grid(); plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.title('Training and Validation Loss')
plt.savefig('results/loss_curves.png', dpi=300)
plt.show()
plt.close()


# Visualize Feature Maps
print("\n--- Visualizing Feature Maps ---")
model.eval()
mri_dataset.mode = 'val'
img, _ = mri_dataset[10]
img_batch = img.unsqueeze(0).to(device)

# Get outputs and layer names
feature_map_outputs = []
layer_names = []
x = img_batch
for name, layer in model.cnn_model.named_children():
    x = layer(x)
    if isinstance(layer, (nn.Conv2d, nn.AvgPool2d)):
        feature_map_outputs.append(x)
        layer_names.append(f"After Layer {int(name)+1}: {layer.__class__.__name__}")

for layer_output, layer_name in zip(feature_map_outputs, layer_names):
    
    layer_viz = layer_output.squeeze(0)
    num_kernels = layer_viz.shape[0]
    
    cols = int(np.ceil(np.sqrt(num_kernels)))
    rows = -(-num_kernels // cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(layer_name, fontsize=16)
    
    axes = axes.flatten()
    
    for i, feature_map in enumerate(layer_viz):
        ax = axes[i]
        ax.imshow(feature_map.detach().cpu().numpy())
        ax.axis("off")
        
    for i in range(num_kernels, len(axes)):
        axes[i].axis("off")
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
    plt.close()

print("\n[Finished]")