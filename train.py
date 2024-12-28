import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

# Dataset Loader
class OCRDataset(Dataset):
    def __init__(self, dataset_path, img_width, img_height, char_to_index, set_type='train'):
        with open(os.path.join(dataset_path, "dataset.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        
        # Validate set_type
        if set_type not in self.config:
            raise ValueError(f"Invalid set_type: '{set_type}'. Expected one of 'train', 'val', or 'test'.")
        
        set_config = self.config[set_type]
        self.image_dir = os.path.join(dataset_path, set_config["images"])
        self.label_dir = os.path.join(dataset_path, set_config["labels"])

        self.image_paths = [os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)]
        self.char_to_index = char_to_index
        self.img_width = img_width
        self.img_height = img_height

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        label_filename = os.path.basename(image_path).replace(".jpg", ".txt")
        label_path = os.path.join(self.label_dir, label_filename)
        
        # Read the label
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Preprocess the image
        image = Image.open(image_path).convert("L")
        image = image.resize((self.img_width, self.img_height))
        image = transforms.ToTensor()(image)
        image = image / 255.0  # Normalize

        # Encode label
        encoded_label = [self.char_to_index[char] for char in label]
        label_length = len(encoded_label)

        return image, torch.tensor(encoded_label, dtype=torch.long), label_length, torch.tensor(len(label), dtype=torch.long)

# CRNN Model (with PyTorch)
class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_classes):
        super(CRNN, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # RNN Layers (Bidirectional LSTM)
        self.lstm = nn.LSTM(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        
        # Output layer (dense layer for character predictions)
        self.fc = nn.Linear(128 * 2, num_classes)  # Multiply by 2 because of bidirectional

    def forward(self, x):
        # Apply convolutions
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Reshape the output to feed into the LSTM
        # Flatten image output (batch_size, channels, height, width) -> (batch_size, width, height*channels)
        x = x.permute(0, 3, 1, 2).contiguous()  # (batch_size, width, height * channels)
        x = x.view(x.size(0), x.size(1), -1)

        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply final fully connected layer to output character probabilities
        x = self.fc(x)
        
        return x

# Training the CRNN model
def train_model(dataset_path, img_width, img_height, char_to_index, batch_size, num_epochs, learning_rate):
    # Create the dataset and dataloader
    train_dataset = OCRDataset(dataset_path, img_width, img_height, char_to_index, set_type='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the CRNN model, loss function, and optimizer
    model = CRNN(img_height, img_width, num_classes=len(char_to_index))
    model = model.cuda()  # Use GPU if available
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, labels, label_lengths, _ in train_loader:
            images = images.cuda()  # Use GPU if available
            labels = labels.cuda()
            label_lengths = label_lengths.cuda()

            # Forward pass
            output = model(images)

            # Compute CTC Loss
            output = output.log_softmax(2)  # Log softmax is required for CTC loss
            loss = criterion(output, labels, label_lengths, torch.full((images.size(0),), output.size(1), dtype=torch.long).cuda())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}")

    # Save the model
    torch.save(model.state_dict(), 'crnn_model.pth')

# Predict using the trained CRNN model
def predict_image(model, image_path, img_width, img_height, char_to_index):
    model.eval()
    
    image = Image.open(image_path).convert("L")
    image = image.resize((img_width, img_height))
    image = transforms.ToTensor()(image).unsqueeze(0)  # Add batch dimension
    image = image / 255.0  # Normalize

    image = image.cuda()  # Use GPU if available

    with torch.no_grad():
        output = model(image)
        output = output.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, num_classes)
        output = output.log_softmax(2)
        
        # Decode the prediction (use the greedy decoding approach)
        _, predicted_indices = output.max(2)
        predicted_indices = predicted_indices.squeeze(0).cpu().numpy()

        # Convert indices back to text
        index_to_char = {i: char for char, i in char_to_index.items()}
        pred_text = ''.join([index_to_char.get(idx, '') for idx in predicted_indices if idx != 0])  # Exclude padding
    return pred_text

# Main function to load data and train the model
if __name__ == "__main__":
    dataset_path = "./crnn_dataset"  # Path to dataset directory containing images and labels.yaml
    img_width, img_height = 210, 80
    num_classes = 36  # 26 lowercase + 10 digits
    char_to_index = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789", start=1)}  # Index 0 is reserved for padding

    try:
        # Train the model
        train_model(dataset_path, img_width, img_height, char_to_index, batch_size=32, num_epochs=10, learning_rate=0.001)

        # Load trained model and predict
        model = CRNN(img_height, img_width, num_classes)
        model.load_state_dict(torch.load('crnn_model.pth'))
        model = model.cuda()

        # Test prediction
        test_image_path = 'path_to_test_image.jpg'
        predicted_text = predict_image(model, test_image_path, img_width, img_height, char_to_index)
        print(f"Predicted text: {predicted_text}")

    except Exception as e:
        print(f"Error: {e}")
