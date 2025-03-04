import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

class WrongVolumeProjection:
    def __init__(self, volume_file_path,max_shift_x,min_shift_x, max_shift_y,min_shift_y, batch_size):
        #the input_dim defines the dmensions of the volume and t, he it stores it
        
        self.volume = torch.from_numpy(ImageHandler(volume_file_path).getData().astype(np.float32)) #read volume
        self.input_dim = self.volume.shape[0]  # The size of the 3D volume
        self.batch_size=batch_size
        self.euler_angles=torch.tensor(np.random.uniform(0, 360, size=3))

        max_shift_x= 10*max_shift_x
        min_shift_x= 0.1*min_shift_x
        max_shift_y= 10*max_shift_y
        min_shift_y= 0.1*min_shift_y

        self.shift_x=np.random.uniform(min_shift_x, max_shift_x, size=1).item()
        self.shift_y=np.random.uniform(min_shift_y, max_shift_y, size=1).item()


    def rotate_volume(self,volume,angles):

        rotated_volumes = []
        for i in range(self.batch_size):

        # Rotate the 3D volume using the given Euler angles (in degrees).
            rot, tilt, psi = map(torch.deg2rad, self.euler_angles) #convert the euler angles to radians
            rot, tilt, psi = -rot, -tilt, -psi

            # Define rotation matrices
            # z1 is a rotation around the z-axis
            R_z1 = torch.tensor([
                [torch.cos(rot), -torch.sin(rot), 0.,0.],
                [torch.sin(rot), torch.cos(rot), 0.,0.],
                [0., 0., 1.,0.],
                [0.,0.,0.,1.]
            ], dtype=torch.float)
        #y is a rotation around the y-axis.
            R_y = torch.tensor([
                [torch.cos(tilt), 0., torch.sin(tilt),0.],
                [0., 1., 0.,0.],
                [-torch.sin(tilt), 0., torch.cos(tilt),0.],
                [0.,0.,0.,1.]
            ], dtype=torch.float)
        # z2 is a rotation around the z-axis
            R_z2 = torch.tensor([
                [torch.cos(psi), -torch.sin(psi), 0.,0.],
                [torch.sin(psi), torch.cos(psi), 0.,0.],
                [0., 0., 1.,0.],
                [0.,0.,0.,1.]
            ], dtype=torch.float)

            # Combined rotation matrix, therefore multiply the individual rotation matrices in that sequence, starting with R_z2, followed by R_y, and then R_z1
            R = R_z1.T @ R_y.T @ R_z2.T

            # Create a grid for rotation
            grid = F.affine_grid(R.unsqueeze(0)[:,:-1,:],[1,1,self.input_dim,self.input_dim,self.input_dim], align_corners=False) 
            rotated_volume = F.grid_sample(self.volume[None, None, ...], grid, align_corners=False) #interpolar
            #[None, None, ...] dos ejes nulos al principio 

            rotated_volume.append(rotated_volume)
        return torch.cat(rotated_volume, dim=0)

    def project_volume(self, volume):
    
       # Project the 3D volume into a 2D plane by adding along the z-axis.
        return torch.sum(volume, dim=2) #la 0 es batch size, la 1 es channels y la 2 ya es z

    def shift_projection(self, projection):
  
       # Shift the 2D projection by the given x and y offsets.
        batch_size=1
        #descomentar esta linea cuando tengamos batch size
       # batch_size, height, width = projection.shape 
       #cuando los shift sean una matriz, hacer una matriz en vez de repeat y añadir los shifts
        half_box = 1. / (0.5 * projection.shape[-1])
        grid = F.affine_grid(
            torch.tensor([[1., 0., half_box * self.shift_x], [0., 1., half_box * self.shift_y]], dtype=torch.float)[None], #los shifts tienen q estar normalizados en [-1,1]
            [batch_size,1,self.input_dim,self.input_dim],
            align_corners=False
        )
        #probar con menos shift por si acaso
        shifted_projection = F.grid_sample(projection, grid, align_corners=False)
        return shifted_projection

    def forward(self):

       #rotate, project, and shift the volume.
        rotated_volume = self.rotate_volume(self.volume, self.euler_angles)
        projection = self.project_volume(rotated_volume)
        shifted_projection = self.shift_projection(projection)
        return shifted_projection
    
