"""
Script to download Stanford 40 Actions dataset
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm
import config


class DownloadProgressBar(tqdm):
    """Progress bar for download"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_stanford40_dataset():
    """
    Download Stanford 40 Actions dataset
    """
    print("="*80)
    print("Stanford 40 Actions Dataset Downloader")
    print("="*80)
    
    # Dataset URL
    dataset_url = "http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip"
    annotations_url = "http://vision.stanford.edu/Datasets/Stanford40_XMLAnnotations.zip"
    splits_url = "http://vision.stanford.edu/Datasets/Stanford40_ImageSplits.zip"
    
    # Create data directory
    os.makedirs(config.DATASET_PATH, exist_ok=True)
    
    # Download images
    print("\n1. Downloading images...")
    images_zip = os.path.join(config.DATASET_PATH, "images.zip")
    if not os.path.exists(images_zip):
        download_url(dataset_url, images_zip)
        print("Images downloaded successfully!")
    else:
        print("Images already downloaded.")
    
    # Download annotations
    print("\n2. Downloading annotations...")
    annotations_zip = os.path.join(config.DATASET_PATH, "annotations.zip")
    if not os.path.exists(annotations_zip):
        download_url(annotations_url, annotations_zip)
        print("Annotations downloaded successfully!")
    else:
        print("Annotations already downloaded.")
    
    # Download image splits
    print("\n3. Downloading image splits...")
    splits_zip = os.path.join(config.DATASET_PATH, "splits.zip")
    if not os.path.exists(splits_zip):
        download_url(splits_url, splits_zip)
        print("Image splits downloaded successfully!")
    else:
        print("Image splits already downloaded.")
    
    # Extract files
    print("\n4. Extracting files...")
    
    # Extract images
    if not os.path.exists(config.IMAGES_PATH):
        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(config.DATASET_PATH)
        print("Images extracted!")
    else:
        print("Images already extracted.")
    
    # Extract annotations
    if not os.path.exists(config.ANNOTATIONS_PATH):
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(config.DATASET_PATH)
        print("Annotations extracted!")
    else:
        print("Annotations already extracted.")
    
    # Extract splits
    if not os.path.exists(config.IMAGE_SETS_PATH):
        print("Extracting image splits...")
        with zipfile.ZipFile(splits_zip, 'r') as zip_ref:
            zip_ref.extractall(config.DATASET_PATH)
        print("Image splits extracted!")
    else:
        print("Image splits already extracted.")
    
    # Verify dataset
    print("\n5. Verifying dataset...")
    num_images = len([f for f in os.listdir(config.IMAGES_PATH) if f.endswith('.jpg')])
    num_annotations = len([f for f in os.listdir(config.ANNOTATIONS_PATH) if f.endswith('.xml')])
    
    print(f"Number of images: {num_images}")
    print(f"Number of annotations: {num_annotations}")
    
    if num_images > 9000 and num_annotations > 9000:
        print("\n✓ Dataset downloaded and verified successfully!")
        print(f"Dataset location: {config.DATASET_PATH}")
    else:
        print("\n✗ Dataset verification failed. Please check the downloads.")
    
    print("\n" + "="*80)
    print("Download complete! You can now run training scripts.")
    print("="*80)


def main():
    download_stanford40_dataset()


if __name__ == "__main__":
    main()
