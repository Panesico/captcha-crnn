import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, Bidirectional, LSTM, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# Dataset Loader
def load_ctc_dataset(dataset_path, batch_size, img_width, img_height, char_to_index, set_type='train'):
    # Load the YAML configuration
    with open(os.path.join(dataset_path, "dataset.yaml"), "r") as f:
        config = yaml.safe_load(f)

    # Debugging: Print the loaded config to inspect its structure
    print("Loaded config:", config)

    # Check if the set_type is valid (train, val, or test)
    if set_type not in config:
        raise ValueError(f"Invalid set_type: '{set_type}'. Expected one of 'train', 'val', or 'test'.")
    
    # Get the image and label directories for the specified set_type
    set_config = config[set_type]  # This should be a dictionary
    image_dir = set_config["images"]  # Image directory: train/images, val/images, test/images
    label_dir = set_config["labels"]  # Label directory: train/labels, val/labels, test/labels
    
    # Check if paths are correct
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")

    # Get image paths
    image_paths = [os.path.join(dataset_path, image_dir, f) for f in os.listdir(os.path.join(dataset_path, image_dir))]
    
    images, labels = [], []
    for image_path in image_paths:
        # Get label from corresponding .txt file
        label_filename = os.path.basename(image_path).replace(".jpg", ".txt")
        label_path = os.path.join(dataset_path, label_dir, label_filename)
        
        with open(label_path, 'r') as label_file:
            label = label_file.read().strip()

        # Load and preprocess image
        img = load_img(image_path, target_size=(img_height, img_width), color_mode="grayscale")
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)

        # Encode label
        encoded_label = [char_to_index[char] for char in label]
        labels.append(encoded_label)
    
    # Pad labels
    max_label_length = max(len(label) for label in labels)
    padded_labels = tf.keras.preprocessing.sequence.pad_sequences(labels, maxlen=max_label_length, padding='post')

    # Convert to tf.data.Dataset
    images = np.array(images)
    padded_labels = np.array(padded_labels)

    def gen():
        for i in range(len(images)):
            yield {
                "image_input": images[i],
                "label_input": padded_labels[i],
                "input_length": [img_width // 4],  # Approx. feature map length after CNN
                "label_length": [len(labels[i])]
            }, np.zeros(1)  # Dummy CTC loss output

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            {
                "image_input": tf.TensorSpec(shape=(img_height, img_width, 1), dtype=tf.float32),
                "label_input": tf.TensorSpec(shape=(max_label_length,), dtype=tf.int32),
                "input_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
                "label_length": tf.TensorSpec(shape=(1,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
        )
    )
    return dataset.batch(batch_size)

# Main
if __name__ == "__main__":
    dataset_path = "./crnn_dataset"  # Path to dataset directory containing images and labels.yaml
    img_width, img_height = 210, 80
    num_classes = 36  # 26 lowercase + 10 digits
    char_to_index = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789", start=1)}  # Index 0 is reserved for padding
    
    try:
        dataset = load_ctc_dataset(dataset_path, batch_size=32, img_width=img_width, img_height=img_height, char_to_index=char_to_index, set_type='train')
    except Exception as e:
        print(f"Error: {e}")
