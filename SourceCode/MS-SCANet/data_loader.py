import torch
import torchvision
import torchvision.transforms.functional as F
import folders
from PIL import Image

class DataLoader(object):
    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True):
        self.batch_size = batch_size
        self.istrain = istrain
        #min_img_size = max(patch_size)  # Ensure image is at least as large as the largest patch size

        if istrain:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.CenterCrop(size=384),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                # 								 std=(0.229, 0.224, 0.225))
            ])
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(size=384),
                torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                # 								 std=(0.229, 0.224, 0.225))
            ])
        # Assuming dataset-specific folder loading logic is handled elsewhere
        self.data = folders.Koniq_10kFolder(root=path, index=img_indx, transform=transforms, patch_num=patch_num)

    def get_data(self):
        if hasattr(self, 'data'):
            dataloader = torch.utils.data.DataLoader(self.data, batch_size=self.batch_size, shuffle=self.istrain)
            return dataloader
        else:
            raise ValueError("Data attribute is not set. Please check dataset initialization.")
