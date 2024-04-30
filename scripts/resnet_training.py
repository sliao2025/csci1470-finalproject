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

from tqdm import tqdm
import gc

import os

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set hyperparams
IMG_DIR = 'spectrogram_images/'
IMG_HEIGHT = 216
IMG_WIDTH = 216
NUM_CLASSES = 7
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

one_hot = OneHotEncoder(categories=[range(NUM_CLASSES)])

# get working directory
cur_dir = os.getcwd()
root_dir = os.path.dirname(cur_dir)
specto_dir = os.path.join(root_dir, IMG_DIR)
all_files = os.listdir(specto_dir)

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

conv_base = models.resnet152(weights='DEFAULT')
in_features = conv_base.fc.in_features
conv_base.fc = torch.nn.Identity()

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
  nn.Softmax()
)

# fine tuning, allow resnet pretrained weights to be trainable
for param in conv_base.parameters():
    param.requires_grad = True

# load images, prepare labels, and normalize 
def load_batch(file_list):
    img_array = []
    idx_array = []
    label_array = []

    for file_ in file_list:
        im = Image.open(specto_dir + file_)
        im = im.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        img_array.append(np.array(im))

        vals = file_[:-4].split('_')
        idx_array.append(vals[0])
        label_array.append([label_dict[vals[1]]])

    label_array = one_hot.fit_transform(label_array).toarray()
    img_array = np.array(img_array)/255.0 # Normalize RGB
    
    return img_array, np.array(label_array), np.array(idx_array)


def batch_generator(files, BATCH_SIZE):
    L = len(files)

    #this line is just to make the generator infinite, keras needs that    
    while True:

        batch_start = 0
        batch_end = BATCH_SIZE

        while batch_start < L:
            
            limit = min(batch_end, L)
            file_list = files[batch_start: limit]
            batch_img_array, batch_label_array, batch_idx_array = load_batch(file_list)

            # Convert numpy arrays to PyTorch tensors
            batch_img_array = torch.tensor(batch_img_array, dtype=torch.float32)
            batch_label_array = torch.tensor(batch_label_array, dtype=torch.float32)
            

            yield (batch_img_array, batch_label_array) # a tuple with two numpy arrays with batch_size samples     

            batch_start += BATCH_SIZE   
            batch_end += BATCH_SIZE


# set training optimizer, loss, and metrics
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = torch.nn.functional.cross_entropy

def categorical_accuracy(output, target):
    predicted = torch.argmax(output, dim=0)
    correct = (predicted == target).float()
    return correct.sum() 


# Calculate number of steps per epoch
STEPS_PER_EPOCH = len(train_files) // BATCH_SIZE
VAL_STEPS = len(val_files) // BATCH_SIZE

# Initialize lists to store training and validation losses and accuracies
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

model.to(device)


# Training loop
for epoch in range(NUM_EPOCHS):
    # Training
    model.train()  # Set the model to train mode
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(batch_generator(train_files, BATCH_SIZE)), total=STEPS_PER_EPOCH):
        # Permute the inputs to [N, C, H, W] from [N, H, W, C]
        inputs = inputs.permute(0, 3, 1, 2)
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = loss_function(outputs, targets)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item()
        # aggregate total number correct
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
        for batch_idx, (inputs, targets) in tqdm(enumerate(batch_generator(val_files, BATCH_SIZE)), total=VAL_STEPS):
            # Permute the inputs to [N, C, H, W] from [N, H, W, C]
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            targets = targets.to(device)
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