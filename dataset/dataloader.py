import warnings

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transform
        # Read the csv file
        data_info = pd.read_csv(csv_path)
        # Second column is the image paths
        self.image_path = data_info.iloc[:, 1].values
        # First column is the image IDs
        self.label_arr = data_info.iloc[:, 0].values
        # Calculate len
        self.data_len = len(data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_path[index]
        # Open image
        img_as_img = Image.open(single_image_name).convert('RGB')
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
