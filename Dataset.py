from torchvision.transforms import v2
import torchvision.tv_tensors as tv_tensors
import os
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import train_transforms, get_boxes_from_mask, init_point_sampling
import json
import random
from utils import find_bounding_box
preprocess = v2.Compose([
    v2.RandomHorizontalFlip(p=1),
    v2.ColorJitter(),
    v2.RandomResizedCrop(size=256),
])

class TestingDataset(Dataset):
    
    def __init__(self, data_path, image_size=256, mode='test', requires_name=True, point_num=1, return_ori_mask=True, prompt_path=None,label=0,pixel_mean=None,pixel_std=None):
        """
        Initializes a TestingDataset object.
        Args:
            data_path (str): The path to the data.
            image_size (int, optional): The size of the image. Defaults to 256.
            mode (str, optional): The mode of the dataset. Defaults to 'test'.
            requires_name (bool, optional): Indicates whether the dataset requires image names. Defaults to True.
            point_num (int, optional): The number of points to retrieve. Defaults to 1.
            return_ori_mask (bool, optional): Indicates whether to return the original mask. Defaults to True.
            prompt_path (str, optional): The path to the prompt file. Defaults to None.
        """
        self.image_size = image_size
        self.return_ori_mask = return_ori_mask
        self.prompt_path = prompt_path
        self.prompt_list = {} if prompt_path is None else json.load(open(prompt_path, "r"))
        self.requires_name = requires_name
        self.point_num = point_num
        self.label = label
        json_file = open(os.path.join(data_path, f'label2image_{mode}.json'), "r")
        self.label_dict = json.load(open('data/cluster_results.json',"r"))
        dataset = json.load(json_file)
    
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
      
        if pixel_mean == None:
            self.pixel_mean = [123.675, 116.28, 103.53]
        else:
            self.pixel_mean = pixel_mean
        if pixel_std == None:
            self.pixel_std = [58.395, 57.12, 57.375]
        else:
            self.pixel_std = pixel_std
    
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an item from the dataset.
        Args:
            index (int): The index of the item to retrieve.
        Returns:
            dict: A dictionary containing the preprocessed image and associated information.
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        mask_path = self.label_paths[index]
        ori_np_mask = cv2.imread(mask_path, 0)
        
        if ori_np_mask.max() == 255:
            ori_np_mask = ori_np_mask / 255

        assert np.array_equal(ori_np_mask, ori_np_mask.astype(bool)), f"Mask should only contain binary values 0 and 1. {self.label_paths[index]}"

        h, w = ori_np_mask.shape
        ori_mask = torch.tensor(ori_np_mask).unsqueeze(0)

        transforms = train_transforms(self.image_size, h, w)
        augments = transforms(image=image, mask=ori_np_mask)
        image, mask = augments['image'], augments['mask'].to(torch.int64)

        if self.prompt_path is None:
            try:
                boxes = get_boxes_from_mask(mask)
                if boxes.numel() == 0:
                    print(self.label_paths[index])
            except:
                boxes= [find_bounding_box(mask.numpy())]
                print(self.label_paths[index])
            point_coords, point_labels = init_point_sampling(mask, self.point_num)
        else:
            prompt_key = mask_path.split('/')[-1]
            boxes = torch.as_tensor(self.prompt_list[prompt_key]["boxes"], dtype=torch.float)
            point_coords = torch.as_tensor(self.prompt_list[prompt_key]["point_coords"], dtype=torch.float)
            point_labels = torch.as_tensor(self.prompt_list[prompt_key]["point_labels"], dtype=torch.int)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(0)
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["boxes"] = boxes
        image_input["original_size"] = (h, w)
        image_input["label_path"] = '/'.join(mask_path.split('/')[:-1])
        # image_input["token"] = self.label
        try:
            image_input["token"] = int(self.label_dict[self.image_paths[index]])
        except:
            image_input["token"] = self.label
        if self.return_ori_mask:
            image_input["ori_label"] = ori_mask
     
        image_name = self.label_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input

    def __len__(self):
        return len(self.label_paths)

class TrainingDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5,label=0,pixel_mean=None,pixel_std=None):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        if pixel_mean == None:
            self.pixel_mean = [123.675, 116.28, 103.53]
        else:
            self.pixel_mean = pixel_mean
        if pixel_std == None:
            self.pixel_std = [58.395, 57.12, 57.375]
        else:
            self.pixel_std = pixel_std
        self.label = label
        dataset = json.load(open(os.path.join(data_dir, f'image2label_{mode}.json'), "r"))
        self.label_dict = json.load(open('data/cluster_results_16_all.json',"r"))
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
        
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for num ,m in enumerate(mask_path):
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)
            try:
                boxes = get_boxes_from_mask(mask_tensor)
                if boxes.numel() == 0:
                    print("try",self.label_paths[index][num])
            except:
                boxes= [find_bounding_box(mask_tensor.numpy())]
                print("except",self.label_paths[index][num])
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(torch.tensor(boxes))
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        try:
            boxes = torch.stack(boxes_list, dim=0)
        except Exception as e:
            print(self.label_paths[index])
            print(e)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        try:
            image_input["token"] = int(self.label_dict[self.image_paths[index]])
        except:
            image_input["token"] = self.label
        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)
    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class EyesDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [20.86, 42.41, 77.64]
        self.pixel_std = [20.03, 36.05, 59.37]
        image_name = os.listdir(os.path.join(data_dir, mode, 'image'))
        self.image_paths = [os.path.join(data_dir, mode, 'image', i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('image','mask') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class DriveDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [41.42, 69.02, 126.84]
        self.pixel_std = [25.18, 45.48, 84.57]
        image_name = os.listdir(os.path.join(data_dir, mode, 'images'))
        self.image_paths = [os.path.join(data_dir, mode, 'images', i) for i in image_name if i.endswith('tif')]
        if mode == 'train':
            self.label_paths = [i.replace('images','1st_manual').replace('training.tif','manual1.png') for i in self.image_paths]
        elif mode == 'test':
            self.label_paths = [i.replace('images','mask').replace('training.tif','manual1.png') for i in self.image_paths]

    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class Drive2Dataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [82.52, 82.52, 82.52]
        self.pixel_std = [52.35, 52.35, 52.35]
        image_name = os.listdir(os.path.join(data_dir, mode, 'images'))
        self.image_paths = [os.path.join(data_dir, mode, 'images', i) for i in image_name if i.endswith('png')]
        self.label_paths = ['_'.join([i.split('_')[0], 'manual1.png']).replace('images','labels') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class Chasedb1Dataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [7.11, 41.80, 115.48]
        self.pixel_std = [8.97, 35.76, 86.67]
        image_name = os.listdir(os.path.join(data_dir, 'Images'))
        self.image_paths = [os.path.join(data_dir, 'Images', i) for i in image_name if i.endswith('jpg')]
        self.label_paths = [i.replace('Images','Masks').replace('.jpg','.png').replace('L','L_2ndHO').replace('R','R_2ndHO') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class FivesDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, require_torch_augment=False):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [16.20, 38.92, 86.11]
        self.pixel_std = [10.52, 26.44, 54.57]
        image_name = os.listdir(os.path.join(data_dir, mode ,'Original'))
        self.image_paths = [os.path.join(data_dir, mode ,'Original',i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('Original','Ground truth') for i in self.image_paths]
        self.prompt_paths = [i.replace('Original','Prompt mask') for i in self.image_paths]
        self.require_torch_augment = require_torch_augment
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int16)
            mask_tensor = tv_tensors.Mask(mask_tensor)
            if self.require_torch_augment:
                image_tensor, mask_tensor = preprocess(image_tensor, mask_tensor)
            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        # prompt_mask = cv2.imread(self.prompt_paths[index],0)
        
        # prompt_mask = cv2.resize(prompt_mask,(self.image_size//4,self.image_size//4))
        # if prompt_mask.max() == 255:
        #     prompt_mask = prompt_mask / 255

        # prompt_mask_tensor = torch.tensor(prompt_mask).to(torch.int16)
        # prompt_mask_tensor = tv_tensors.Mask(prompt_mask_tensor)

        # image_input['prompt_mask'] = prompt_mask_tensor.unsqueeze(0).unsqueeze(0)
        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class StareDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [28.91, 85.12, 150.18]
        self.pixel_std = [17.85, 45.35, 86.13]
        image_name = os.listdir(os.path.join(data_dir, 'image'))
        self.image_paths = [os.path.join(data_dir, 'image', i) for i in image_name if i.endswith('ppm')]
        self.label_paths = [i.replace('image','mask').replace('.ppm','.vk.ppm') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        #boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = None
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

def stack_dict_batched(batched_input):
    out_dict = {}
    for k,v in batched_input.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

class RBVDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [40.85,67.65,132.05]
        self.pixel_std = [24.57,43.84,87.65]
        image_name = os.listdir(os.path.join(data_dir, mode ,'image'))
        self.image_paths = [os.path.join(data_dir, mode ,'image',i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('image','mask') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample
        """
        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int16)
            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        # boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        #image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    
    def __len__(self):
        return len(self.image_paths)
    
    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class HRFDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [26.18, 51.13, 158.34]
        self.pixel_std = [14.01, 26.52, 73.32]
        image_name = os.listdir(os.path.join(data_dir, 'images'))
        self.image_paths = [os.path.join(data_dir, 'images', i) for i in image_name if i.endswith('jpg')]
        self.label_paths = [i.replace('images','manual1').replace('.jpg','.tif') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        #boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = None
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class HAGISDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [40.48, 52.03, 105.85]
        self.pixel_std = [23.98, 31.20, 62.95]
        image_name = os.listdir(os.path.join(data_dir, 'images'))
        self.image_paths = [os.path.join(data_dir, 'images', i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('images','labels') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        #boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = None
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class LESDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [25.76, 44.97, 78.29]
        self.pixel_std = [16.61, 31.60, 51.95]
        image_name = os.listdir(os.path.join(data_dir, 'images'))
        self.image_paths = [os.path.join(data_dir, 'images', i) for i in image_name if i.endswith('png')]
        self.label_paths = [i.replace('images','labels') for i in self.image_paths]
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
    
        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = []
        for _ in range(self.mask_num):
            mask_path.append(self.label_paths[index])
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255

            augments = transforms(image=image, mask=pre_mask)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int64)

            # boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            # boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        #boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        # image_input["boxes"] = None
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")

class RefugeDataset(Dataset):
    def __init__(self, data_dir, image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5, require_torch_augment=False):
        """
        Initializes a training dataset.
        Args:
            data_dir (str): Directory containing the dataset.
            image_size (int, optional): Desired size for the input images. Defaults to 256.
            mode (str, optional): Mode of the dataset. Defaults to 'train'.
            requires_name (bool, optional): Indicates whether to include image names in the output. Defaults to True.
            num_points (int, optional): Number of points to sample. Defaults to 1.
            num_masks (int, optional): Number of masks to sample. Defaults to 5.
        """
        self.image_size = image_size
        self.requires_name = requires_name
        self.point_num = point_num
        self.mask_num = mask_num
        self.pixel_mean = [16.20, 38.92, 86.11]
        self.pixel_std = [10.52, 26.44, 54.57]
        image_name = os.listdir(os.path.join(data_dir, mode ,'Original'))
        self.image_paths = [os.path.join(data_dir, mode ,'Original',i) for i in image_name if i.endswith('jpg')]
        self.label_paths = [i.replace('Original','Ground truth').replace('jpg','bmp') for i in self.image_paths]
        self.prompt_paths = [i.replace('Original','Prompt mask') for i in self.image_paths]
        self.require_torch_augment = require_torch_augment
    
    def __getitem__(self, index):
        """
        Returns a sample from the dataset.
        Args:
            index (int): Index of the sample.
        Returns:
            dict: A dictionary containing the sample data.
        """

        image_input = {}
        try:
            image = cv2.imread(self.image_paths[index])
            image = (image - self.pixel_mean) / self.pixel_std
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = train_transforms(self.image_size, h, w)
        masks_list = []
        boxes_list = []
        image_list = []
        point_coords_list, point_labels_list = [], []
        pre_mask = cv2.imread(self.label_paths[index], 0)
        if pre_mask.max() == 255:
            pre_mask = pre_mask / 255
        pre_masks_list = self.extract_mask(pre_mask)
        for m in pre_masks_list:
            augments = transforms(image=image, mask=m)
            image_tensor, mask_tensor = augments['image'], augments['mask'].to(torch.int16)
            mask_tensor = tv_tensors.Mask(mask_tensor)
            if self.require_torch_augment:
                image_tensor, mask_tensor = preprocess(image_tensor, mask_tensor)
            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)
            image_list.append(image_tensor)
            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)
        image = torch.stack(image_list, dim=0)
        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels

        image_name = self.image_paths[index].split('/')[-1]
        if self.requires_name:
            image_input["name"] = image_name
            return image_input
        else:
            return image_input
    def __len__(self):
        return len(self.image_paths)

    def get_mean_std(self):
        img_channels = 3
        cumulative_mean = np.zeros(img_channels)
        cumulative_std = np.zeros(img_channels)

        for image_path in tqdm(self.image_paths):
            image = cv2.imread(image_path)
            for d in range(3):
                cumulative_mean[d] += image[:, :, d].mean()
                cumulative_std[d] += image[:, :, d].std()
        self.pixel_mean = cumulative_mean / len(self.image_paths)
        self.pixel_std = cumulative_std / len(self.image_paths)
        print(f"mean: {self.pixel_mean}")
        print(f"std: {self.pixel_std}")
    
    def extract_mask(self, arr, background_value=0):
        unique_values = np.unique(arr)

        # 创建一个二维数组
        result_arr = np.zeros_like(arr)
        arr_list = []
        for value in unique_values:
            if value == background_value:
                continue
            extracted_arr = np.where(arr == value, 1, 0)
            arr_list.append(extracted_arr)

        return arr_list

if __name__ == "__main__":
    # train_dataset = TrainingDataset("data_demo", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # print("Dataset:", len(train_dataset))
    # eyes_dataset = EyesDataset("data/eyes_selected", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = RBVDataset("data/RetinaBloodVessel", image_size=256, mode='test', requires_name=True, point_num=1, mask_num=1)
    # eyes_dataset = LESDataset("data/LES-AV", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=1)
    # eyes_dataset = DriveDataset("data/DRIVE", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    eyes_dataset = RefugeDataset("data/Refuge", image_size=256, mode='train', requires_name=True, point_num=10, mask_num=5)
    print(eyes_dataset.label_paths[1])
    print(eyes_dataset[1]['point_coords'].shape)
    # eyes_dataset = Chasedb1Dataset("data/chasedb1_patch", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # eyes_dataset = StareDataset("data/stare", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # train_loader = DataLoader(eyes_dataset, batch_size = 64, shuffle=True, num_workers=0)
    # eyes_dataset = FivesPPO("data/FIVES", image_size=256, mode='train', requires_name=True, point_num=1, mask_num=5)
    # print(eyes_dataset[0])
    # eyes_dataset.get_mean_std()
    print(eyes_dataset[0]['image'].min())
    print(eyes_dataset[1]['label'].unique())
    # cv2.imwrite('test.jpg',eyes_dataset[0]['label'][0].permute(1,2,0).numpy()*255)
    # cv2.imwrite('test1.jpg',eyes_dataset[0]['prompt_mask'][0].permute(1,2,0).numpy()*255)
    # for batch, batched_input in enumerate(train_loader):
    #     print(batched_input)
    # train_batch_sampler = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True, num_workers=4)
    # for i, batched_image in enumerate(tqdm(train_batch_sampler)):
    #     batched_image = stack_dict_batched(batched_image)
    #     print(batched_image["image"].shape, batched_image["label"].shape)

