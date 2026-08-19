import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import scipy.io
import numpy as np
import csv
from torchvision import transforms

class Koniq_10kFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        imgname = []
        mos_all = []
        csv_file = os.path.join(root, 'koniq10k_scores_and_distributions.csv')
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['image_name'])
                mos = float(row['MOS_zscore'])
                mos_all.append(mos)

        mos_all = np.array(mos_all)
        mos_all = (mos_all - mos_all.min()) / (mos_all.max() - mos_all.min())

        sample = []
        for i, item in enumerate(index):
            for aug in range(patch_num):
                # Construct path using os.path.join() and normalize it using os.path.normpath()
                path = os.path.normpath(os.path.join(root, '1024x768', imgname[item]))
                sample.append((path, mos_all[item]))

        self.samples = sample
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target
        # Return image data (dist), None, None, MOS (diff)
        #return sample, torch.tensor(0), torch.tensor(0), target

    def __len__(self):
        return len(self.samples)

class LIVEFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        imgname = []
        dmos_all = []
        csv_file = os.path.join(root, 'live_r2_dmos.csv')

        try:
            with open(csv_file) as f:
                reader = csv.reader(f)
                for row in reader:
                    dmos_all.append(float(row[0]))
                    imgname.append(row[2].strip())
        except FileNotFoundError:
            print(f"CSV file {csv_file} not found.")
            raise
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            raise

        dmos_all = np.array(dmos_all)
        dmos_all = (dmos_all - dmos_all.min()) / (dmos_all.max() - dmos_all.min())
        mos_all = 1 - dmos_all

        self.samples = [(os.path.normpath(os.path.join(root, 'dist_images', imgname[item])), mos_all[item]) for item in index for _ in range(patch_num)]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class LIVEChallengeFolder(data.Dataset):
    def __init__(self, root, index, transform, patch_num):
        # Load image paths
        imgpath = scipy.io.loadmat(os.path.join(root, 'Data', 'AllImages_release.mat'))
        imgpath = [x[0][0] for x in imgpath['AllImages_release'][7:1169]]  # Adjusted indexing

        # Load DMOS, normalize to 0-1
        mos = scipy.io.loadmat(os.path.join(root, 'Data', 'AllMOS_release.mat'))
        mos_values = mos['AllMOS_release'].astype(np.float32)[0][7:1169]  # Adjusted indexing
        normalized_mos = mos_values / 100  # Normalize DMOS from 0 to 100 to 0 to 1

        # Compile sample paths and labels
        self.samples = []
        for i in index:
            for aug in range(patch_num):
                self.samples.append((os.path.join(root, 'Images', imgpath[i]), normalized_mos[i]))

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

class CSIQFolder(data.Dataset):
    def __init__(self, root, csv_file, index, transform, patch_num):
        # Load the CSV file containing image names and DMOS scores
        dmos_df = pd.read_csv(csv_file)

        # Define mapping from CSV distortion types to folder names and file extensions
        distortion_type_mapping = {
            'noise': 'AWGN',
            'blur': 'BLUR',
            'contrast': 'contrast',
            'fnoise': 'fnoise',
            'jpeg': 'JPEG',
            'jpeg 2000': 'jpeg2000'
        }

        # Create custom image names based on the dataset structure
        dmos_df['Custom_Image_Name'] = dmos_df.apply(
            lambda row: f"{row['image']}.{distortion_type_mapping[row['dst_type']]}.{row['dst_lev']}.png", axis=1
        )

        # Normalize DMOS scores between 0 and 1
        dmos_values = dmos_df['dmos'].astype(np.float32)
        normalized_dmos = (dmos_values - dmos_values.min()) / (dmos_values.max() - dmos_values.min())
        mos_all=1-normalized_dmos

        # Compile sample paths and labels
        self.samples = []
        for i in index:
            for aug in range(patch_num):
                image_path = os.path.join(root, dmos_df['Custom_Image_Name'].iloc[i])
                if not os.path.exists(image_path):
                    print(f"File not found: {image_path}")
                self.samples.append((image_path, mos_all.iloc[i]))

        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = pil_loader(path)
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

if __name__ == '__main__':
    folder_path = {
        'koniq':    r'./KonIQ',
        'live': r'./LIVE',
        'livec': r'./LIVEC',
        'csiq': r'./CSIQ'
    }

    img_num = {
        'koniq':    list(range(0, 10073)),
        'live': list(range(0, 779)),
        'livec': list(range(0, 1162)),
        'csiq': list(range(0, 866))
    }

    dataset = 'koniq'

    total_num_images = img_num[dataset]

    train_index = total_num_images[:int(round(0.8 * len(total_num_images)))]
    test_index = total_num_images[int(round(0.8 * len(total_num_images))):]

    # Assuming DataLoader here is a custom class you've defined that wraps around PyTorch's DataLoader
    dataloader_train = DataLoader(dataset,
                                  folder_path[dataset],
                                  train_index,
                                  config.PATCH_SIZE,
                                  config.TRAIN_PATCH_NUM,
                                  config.BATCH_SIZE,
                                  istrain=True).get_data()

    dataloader_test = DataLoader(dataset,
                                 folder_path[dataset],
                                 test_index,
                                 config.PATCH_SIZE,
                                 config.TEST_PATCH_NUM,
                                 istrain=False).get_data()

    # Demo to print shape of the first batch, then exit
    for idx, (data, _) in enumerate(dataloader_train):
        print(data.shape)
        break
