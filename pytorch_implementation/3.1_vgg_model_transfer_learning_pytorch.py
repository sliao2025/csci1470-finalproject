import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight

import os
from tqdm import tqdm


# %matplotlib inline
# check cuda availability
torch.cuda.is_available()
mps_device = torch.device("mps")

IMG_DIR = '/users/sliao10/scratch/csci1470-finalproject/fma_spectrogram_images'
IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 16
NUM_EPOCHS = 10
BATCH_SIZE = 32
L2_LAMBDA = 0.001

label_dict = {'Hip':0,
              'Pop':1,
              'Vocal':2,
              'Rhythm':3,
              'Reggae':4,
              'Rock':5,
              'Techno':6,
             }

label_dict_fma = {
    'Hip-Hop': 0,
    'Rock': 1,
    'International': 2,
    'Electronic': 3,
    'Pop': 4,
    'Jazz': 5,
    'Experimental': 6,
    'Folk': 7,
    'Instrumental': 8,
    'Spoken': 9,
    'Classical': 10,
    'Soul-RnB': 11,
    'Old-Time ': 12,
    'Country': 13,
    'Blues': 14,
    'Easy Listening': 15
}
one_hot = OneHotEncoder(categories=[range(NUM_CLASSES)])

all_files = os.listdir(IMG_DIR)

# Get class weights
label_array = []
for file_ in all_files:
    vals = file_[:-4].split('_')
    label_array.append(label_dict[vals[1]])
    
cl_weight = compute_class_weight(class_weight = 'balanced', 
                                 classes = np.unique(label_array), 
                                 y = label_array)

# Train-val-test split of files
train_files, test_files, train_labels, test_labels = train_test_split(all_files, 
                                                                      label_array,
                                                                      random_state = 10, 
                                                                      test_size = 0.1
                                                                     )

# Among the test files, keep half for validation
val_files, test_files, val_labels, test_labels = train_test_split(test_files, test_labels,
                                                                  random_state = 10, 
                                                                  test_size = 0.5
                                                                 )

conv_base = models.vgg16(weights = "IMAGENET1K_V1", progress=True)

in_features = conv_base.classifier[0].in_features

conv_base.classifier = torch.nn.Identity()

conv_base.eval()

# summary(conv_base)
print(conv_base.features)

class L2RegularizedLinear(nn.Module):
  def __init__(self, in_features, out_features, l2_lambda):
    super(L2RegularizedLinear, self).__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.l2_lambda = l2_lambda
  
  def forward(self, x):
    output = self.linear(x)
    l2_reg = torch.norm(self.linear.weight) * self.l2_lambda
    output += l2_reg
    return output


model = nn.Sequential(
  conv_base,
  nn.Flatten(),
  L2RegularizedLinear(in_features, 512, L2_LAMBDA),
  nn.Dropout(p=0.3),
  nn.ReLU(),
  nn.Linear(512, NUM_CLASSES),
  nn.Softmax(dim=-1)
)

#Set the convolution base to not be trainable
for param in conv_base.parameters():
    param.requires_grad = False
# summary(model)

# load images, prepare labels, and normalize 
specto_dir = '/users/sliao10/scratch/csci1470-finalproject/spectrogram_images/'

class CustomDataset(Dataset):
    def __init__(self, files, specto_dir, label_dict, IMG_WIDTH, IMG_HEIGHT):
        self.files = files
        self.specto_dir = specto_dir
        self.label_dict = label_dict
        self.one_hot = one_hot
        self.IMG_WIDTH = IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_ = self.files[idx]
        im = Image.open(self.specto_dir + file_)
        im = im.resize((self.IMG_WIDTH, self.IMG_HEIGHT), Image.Resampling.LANCZOS)
        spectogram = np.array(im) / 255.0
        
        label = file_[:-4].split('_')
        label_array = np.array([self.label_dict[label[1]]])
        label_array = label_array.reshape(1, -1)
        label_array = one_hot.fit_transform(label_array).toarray()

        return spectogram, np.array(label_array[0])
    
# Initialize datasets
train_dataset = CustomDataset(train_files, specto_dir, label_dict_fma, IMG_WIDTH, IMG_HEIGHT)
val_dataset = CustomDataset(val_files, specto_dir, label_dict_fma, IMG_WIDTH, IMG_HEIGHT)
test_dataset = CustomDataset(test_files, specto_dir, label_dict_fma, IMG_WIDTH, IMG_HEIGHT)

# Initialize DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # set training optimizer, loss, and metrics
optimizer = optim.Adam(model.parameters(), lr=5e-5)
loss_function = torch.nn.functional.cross_entropy

def categorical_accuracy(output, target):
    predicted = torch.argmax(output, dim=-1)
    targets = torch.argmax(target, dim=-1)
    # print(predicted.shape, target.shape)
    correct = (predicted == targets).float()
    return correct.sum()

# Calculate number of steps per epoch
STEPS_PER_EPOCH = len(train_files) // BATCH_SIZE
VAL_STEPS = len(val_files) // BATCH_SIZE

# Initialize lists to store training and validation losses and accuracies
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Training loop
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()  # Set the model to train mode
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=STEPS_PER_EPOCH):
        # Permute the inputs to [N, C, H, W] from [N, H, W, C]
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
        inputs = inputs.permute(0, 3, 1, 2)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_function(outputs, targets)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item()
        # aggregate total number 
        correct_train += categorical_accuracy(outputs, targets)
        total_train += targets.size(0)
        # _, predicted = outputs.max(1)
        # correct_train += predicted.eq(targets).sum().item()

    # Calculate average training loss and accuracy
    avg_train_loss = train_loss / STEPS_PER_EPOCH
    train_accuracy = 100. * correct_train / total_train

    # Validation
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=VAL_STEPS):
            inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device,dtype=torch.float32)
            
            inputs = inputs.permute(0, 3, 1, 2)
            
            outputs = model(inputs)  # Forward pass
            loss = loss_function(outputs, targets)  # Calculate the loss
            val_loss += loss.item()
            # _, predicted = outputs.max(1)
            total_val += targets.size(0)
            correct_val += categorical_accuracy(outputs, targets)

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / VAL_STEPS
    val_accuracy = 100. * correct_val / total_val

    # Print training and validation metrics
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, '
          f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, '
          f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model checkpoint
    cur_dir = os.getcwd()
    root_dir = os.path.dirname(cur_dir)
    ckpt_dir = os.path.join(root_dir, f'saved_models/fine_tuning_epoch_{epoch + 1}_{val_accuracy:.4f}.pt')
    torch.save(model.state_dict(), ckpt_dir)

    # Append metrics to lists for plotting later if needed
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    ROOT_DIR = '/users/sliao10/scratch/csci1470-finalproject'
    # save training history
    pkl_dir = os.path.join(ROOT_DIR, 'pickle_files/fine_tuning_vgg16_pytorch_history.pkl')
    history = {
        "train_loss": train_losses,
        "train_accuracy": train_accuracies,
        "val_loss": val_losses,
        "val_accuracy": val_accuracies,
    }
    with open(pkl_dir, 'wb') as f:
        pickle.dump(history, f)

    # save test files
    pkl_dir = os.path.join(ROOT_DIR, 'pickle_files/fine_tuning_vgg16_pytorch_test_files.pkl')
    with open(pkl_dir, 'wb') as f:
        pickle.dump(test_files, f)