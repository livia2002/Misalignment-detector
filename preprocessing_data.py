import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from CUSTOM_DATASET import MetadataDataset
from Volume_projection import VolumeProjection
from WrongVolumeProjection import WrongVolumeProjection

class Preprocesing: 
    def __init__(self,metadata_file_path,volume_path ):
        self.md = XmippMetaData(metadata_file_path)
        self.imgs = self.md.getMetaDataImage(list(range(len(self.md)))) #read images
        self.metadata_labels = self.md.getMetaDataLabels() #get the labels of the metadata table  
        self.volume_path=volume_path  
        self.max_shift_x = np.amax(self.md[:, "shiftX"])
        self.min_shift_x = np.amin(self.md[:, "shiftX"]) 
        self.max_shift_y = np.amax(self.md[:, "shiftY"])
        self.min_shift_y = np.amin(self.md[:, "shiftY"]) 


    def process_data(self, metadata):
        image = metadata['image']
        angles = torch.stack([metadata['angleRot'], metadata['angleTilt'], metadata['anglePsi']], dim=-1)
        shift_x = metadata['shiftX']
        shift_y = metadata['shiftY']

        # Correct alignment projection
        projection_model = VolumeProjection(self.volume_path)
        projection = projection_model.forward(angles, shift_x, shift_y)
        subs = torch.abs(image - torch.from_numpy(np.squeeze(projection.numpy())))
        subs_correct = subs.flatten()
        labels_correct = torch.ones(subs.shape[0])

        # Wrong alignment projection
        wrong_projection_model = WrongVolumeProjection(self.volume_path, self.max_shift_x,self.min_shift_x,self.max_shift_y,self.min_shift_y)
        wrong_projection = wrong_projection_model.forward()
        subs_wrong = torch.abs(image - torch.from_numpy(np.squeeze(wrong_projection.numpy())))
        subs_wrong = subs_wrong.flatten()
        labels_wrong = torch.zerod(subs.shape[0])  # Misaligned -> label 0

        subs=np.append(subs_correct,subs_wrong)
        labels=np.append(labels_correct,labels_wrong)
        
        return subs, labels
