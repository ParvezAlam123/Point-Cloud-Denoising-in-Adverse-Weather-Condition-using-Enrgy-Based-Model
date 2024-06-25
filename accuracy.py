import numpy as np 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import open3d as o3d 
import struct 
from dataset import SemanticSTF , ValidationData 
from Unet import UNet 
import time 
import matplotlib.pyplot as plt  




device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


root_dir = "/media/parvez_alam/Expansion/Denoise validation data"


valid_data = ValidationData(root_dir=root_dir)
valid_loader = DataLoader(dataset=valid_data, batch_size=1, shuffle=False)









loaded_checkpoint = torch.load("trained_unet_with_vae_as_negative_sample.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_unet_with_vae_as_negative_sample.pth")



model = UNet(n_channels=3, n_classes=2)

model.load_state_dict(torch.load("model_state_unet_with_vae_as_negative_sample.pth"))
model.to(device)
model.eval()


T = 1

TP = 0 
TN = 0 
FP = 0 
FN = 0 
threshold = -3




for n, data in enumerate(valid_loader):
    range_image_input = data["points"].to(device).float().permute(0, 3, 1, 2)
    labels = data["labels"].to(device).float().reshape(1, 64, 1024)
    energy_score = -T * (torch.log(torch.exp(model(range_image_input)) / T).sum(dim=1)) 

    for i in range(64):
        for j in range(1024):
            energy_value = energy_score[0][i][j]
            if energy_value <= threshold:
                    predicted_label = 1 
            else:
                    predicted_label = 0 
            gt_label = labels[0][i][j] 
            if predicted_label == gt_label and gt_label == 1  :
                    TP = TP + 1 
            if predicted_label == gt_label and gt_label == 0 :
                    FN = FN + 1  
            if predicted_label != gt_label and predicted_label == 1 and gt_label == 0 :
                    FP = FP + 1 
            if predicted_label != gt_label and predicted_label == 0 and gt_label == 1 :
                    FN = FN + 1 






print("precision = ", TP / (TP + FP))
print("recall = ", TP / (TP + FN))
print("accuracy = ", (TP + TN)/(TP+TN+FP+FN))





                  






