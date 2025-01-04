import os
import json
import shutil
import subprocess
from pathlib import Path
import cv2
import random
import multiprocessing

class CRNNDatasetCreator:
    def __init__(self, output_dir="crnn_dataset"):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path("temp_captchas")
        # Create dataset structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    def generate_captchas(self, num_images):
        """Generate CAPTCHA images using PHP script in parallel."""
        print("Generating CAPTCHA images concurrently...")

        # Split the total number of images into smaller chunks to process concurrently
        num_processes = multiprocessing.cpu_count()  # One process per CPU core
        images_per_process = num_images // num_processes
        # Multiprocessing pool to execute the PHP script in parallel
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(self._generate_captchas_chunk, [
                (images_per_process, i) for i in range(num_processes)
            ])

    def _generate_captchas_chunk(self, num_images_per_process, process_idx):
        """Helper function for generating a chunk of CAPTCHA images."""
        start_idx = process_idx * num_images_per_process
        end_idx = start_idx + num_images_per_process

        # Add a unique seed for randomness in each process
        seed = random.randint(0, 1_000_000)
        result = subprocess.run(
            ['php', 'generate_captchas.php', str(num_images_per_process), str(self.temp_dir), str(seed)],
            capture_output=True,
            text=True
        )
        if result.stderr:
            print(result.stderr)
        try:
            captchas = json.loads(result.stdout)
            # Save the JSON file with the generated CAPTCHA info
            with open(f"{self.temp_dir}/captchas_{start_idx}-{end_idx}.json", "w") as f:
                json.dump(captchas, f)
        except json.JSONDecodeError as e:
            print(f"Error parsing PHP output: {e}")
            print("PHP Output:", result.stdout)
            raise

    def process_image(self, captcha, idx, num_images):
        """Process a single image with augmentations."""
        img = cv2.imread(str(captcha['filename']))
        if img is None:
            raise ValueError(f"Failed to load image: {captcha['filename']}")
        # Apply random augmentations
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # Determine split (80/10/10)
        split = 'train' if idx < 0.8 * num_images else \
                'val' if idx < 0.9 * num_images else 'test'
        # Save image
        output_path = self.output_dir / split / 'images' / f'captcha_{idx:06d}.jpg'
        cv2.imwrite(str(output_path), img)

        # Save label
        label_path = self.output_dir / split / 'labels' / f'captcha_{idx:06d}.txt'
        with open(label_path, 'w') as f:
            f.write(captcha['text'])  # Write the text string as the label

        return idx

    def create_dataset(self, num_images=1000):
        """Create the complete CRNN dataset using multiprocessing."""
        # Generate captchas concurrently
        self.generate_captchas(num_images)
        # Load the generated CAPTCHA information from JSON files
        captchas = []
        for idx in range(multiprocessing.cpu_count()):
            with open(f"{self.temp_dir}/captchas_{idx * (num_images // multiprocessing.cpu_count())}-" 
                      f"{(idx + 1) * (num_images // multiprocessing.cpu_count())}.json") as f:
                captchas.extend(json.load(f))

        print("\nProcessing images and creating CRNN dataset...")

        # Use multiprocessing to process images in parallel
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.process_image, [(captcha, idx, num_images) for idx, captcha in enumerate(captchas)])

        print("\nDataset creation complete!")
        print(f"Processed {len(results)} images.")

        # Create dataset.yaml file
        self.create_yaml()

    def create_yaml(self):
        """Create the dataset.yaml config file for training."""
        yaml_content = f"""
path: {self.output_dir.absolute()}
train:
  images: train/images
  labels: train/labels
val:
  images: val/images
  labels: val/labels
test:
  images: test/images
  labels: test/labels

names:
  0: captcha
"""
        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            f.write(yaml_content)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-images', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default='crnn_dataset')
    args = parser.parse_args()
    creator = CRNNDatasetCreator(args.output_dir)
    creator.create_dataset(args.num_images)
