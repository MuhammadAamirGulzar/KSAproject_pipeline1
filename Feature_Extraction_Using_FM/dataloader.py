import os
import random
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Sampler, DataLoader, Dataset


class PatchLoader(Dataset):
    def __init__(self, label_file, data_path, transform=None, num_samples=None, mode=2):     
        lib = pd.DataFrame(pd.read_csv(label_file, usecols=['WSI_Id', 'label_id'], keep_default_na=True))
        lib.dropna(inplace=True)
        if num_samples is not None:
            lib = lib.sample(n=num_samples)
        tar = lib['label_id'].values.tolist()
        allslides = lib['WSI_Id'].values.tolist()       
        slides = []
        tiles = []
        ntiles = []
        slideIDX = []
        targets = []
        j = 0
        for i, path in enumerate(allslides):
            t = []
            cpath = os.path.join(data_path, str(path))
            if not os.path.exists(cpath):
                # print('This slide does not exist: {}'.format(path))
                continue
            else:
                # print('This slide exists: {}'.format(path))
                # count = 0
                for f in os.listdir(cpath): 
                    if '.png' in f:
                        # count = count + 1
                        t.append(os.path.join(cpath, f))
                if len(t) > 0:
                    slides.append(path)
                    tiles.extend(t)
                    ntiles.append(len(t))
                    slideIDX.extend([j]*len(t))
                    targets.append(int(tar[i]))
                    j+=1
        print('Number of Slides: {}'.format(len(slides)))
        print('Number of tiles: {}'.format(len(tiles)))
        self.slides = slides
        self.slideIDX = slideIDX
        self.ntiles = ntiles
        self.tiles = tiles
        self.targets = targets
        self.transform = transform
        self.mode = mode

    def maketraindata(self, idxs):
        self.t_data = [(self.slideIDX[x], self.tiles[x], self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        pass
    def __getitem__(self, index):
        if self.mode == 1:# loads all tiles from each slide sequentially for train/validatoin set
            tile = self.tiles[index]
            img = Image.open(str(tile)).convert('RGB')
            slideIDX = self.slideIDX[index]
            target = self.targets[slideIDX]
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        elif self.mode == 2:  # used when a different trainset is prepared e.g. with given tile index    
            slideIDX, tile, target = self.t_data[index]
            img = Image.open(str(tile)).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode == 1:
            length = len(self.tiles)
        elif self.mode == 2:
            length = len(self.t_data)
        else:
            length = 0
        # print(f"__len__ called, mode: {self.mode}, length: {length}")
        return length
    


class SlideBatchSampler(Sampler):
    def __init__(self, ntiles):
        # ntiles contains the number of tiles per slide
        self.ntiles = ntiles
        self.indices = []
        start_idx = 0
        for num_tiles in ntiles:
            self.indices.append(list(range(start_idx, start_idx + num_tiles)))
            start_idx += num_tiles
    def __iter__(self):
        # Yield each set of indices for a single slide (batch contains all tiles for that slide)
        for batch in self.indices:
            yield batch
    def __len__(self):
        return len(self.indices)
    
# Custom collate function to handle the batch of crops
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    batch_size, num_crops, c, h, w = images.size()
    images = images.view(-1, c, h, w)  # flatten the crops into individual images
    labels = torch.tensor(labels).repeat_interleave(num_crops)  # repeat labels for each crop
    return images, labels
