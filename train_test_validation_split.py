import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

# Define paths
data_dir = 'with_augmentation/augmentation/'  # Directory containing the subdirectories of elephant images
output_dir = 'with_augmentation/split_data/'  # Directory to save the split data

# Create output directories
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Parameters
seed = 42

# Get all image file paths and their corresponding class names
all_files = []
all_labels = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            all_files.append(os.path.join(root, file))
            all_labels.append(os.path.basename(root))

# Convert to numpy arrays for stratified splitting
all_files = np.array(all_files)
all_labels = np.array(all_labels)

# Define the split sizes
test_size = 0.1
val_size = 0.1

print(f'Using test_size={test_size} and val_size={val_size}')

# Split the data into training and test sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
train_val_idx, test_idx = next(sss.split(all_files, all_labels))
train_val_files, test_files = all_files[train_val_idx], all_files[test_idx]
train_val_labels, test_labels = all_labels[train_val_idx], all_labels[test_idx]

# Further split the training data into training and validation sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
train_idx, val_idx = next(sss.split(train_val_files, train_val_labels))
train_files, val_files = train_val_files[train_idx], train_val_files[val_idx]

# Function to copy files to the respective directories
def copy_files(file_list, target_dir):
    for file in file_list:
        class_name = os.path.basename(os.path.dirname(file))
        target_class_dir = os.path.join(target_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        shutil.copy(file, target_class_dir)

# Copy the files to the respective directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print(f'Training data: {len(train_files)} images')
print(f'Validation data: {len(val_files)} images')
print(f'Test data: {len(test_files)} images')