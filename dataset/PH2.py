import os
import cv2
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from .concept_dataset import PH2_dict

label_dict = dict(
    Nevus=0,
    Melanoma=1
)


class PH2(Dataset):
    def __init__(self,
                 dataset_dir,
                 mode='train',
                 transforms=None,
                 config=None):
        super().__init__()
        self.mode = mode
        self.basedir = dataset_dir
        meta_path = os.path.join(self.basedir, 'annotations', self.mode + '_6.xlsx')
        self.meta = pd.read_excel(meta_path)
        self.transforms = transforms
        self.config = config

    def __getitem__(self, index):
        img_path_meta = self.meta['img_name'].iloc[index]
        img_path = os.path.join(self.basedir, 'images', img_path_meta, img_path_meta + '_Dermoscopic_Image', img_path_meta + '.bmp')
        img = cv2.imread(img_path)

        if self.transforms:
            img = self.transforms(img)
        label = label_dict[self.meta['disease'].iloc[index]]

        concept_labels = []
        for concept in PH2_dict.keys():
            clabel = self.meta[concept].iloc[index]
            clabel = PH2_dict[concept].index(clabel)
            concept_labels.append(clabel)
        concept_labels = np.array(concept_labels)

        return img, label, concept_labels

    def __len__(self):
        return len(self.meta)