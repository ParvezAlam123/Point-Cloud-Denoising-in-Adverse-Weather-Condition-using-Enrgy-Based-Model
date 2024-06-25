import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.optim import SGD, Adam
from dataset import DataScore 
from torch.utils.data import DataLoader
from  vit_backbone import Network 
import matplotlib.pyplot as plt 
import numpy as np 


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


kitti_path = "/media/parvez/Expansion/backup/Data/Kitti/Object/data_object_velodyne/training/velodyne"
SementicSTF_path = "/media/parvez/Expansion1/SemanticSTF/train/velodyne" 

kitti_path_vel = "/media/parvez/Expansion/backup/Data/Kitti/Object/data_object_velodyne/testing/velodyne"
SementicSTF_path_vel = "/media/parvez/Expansion1/SemanticSTF/val/velodyne"



dataset = DataScore(kitti_path=kitti_path_vel, SementicSTF_path=SementicSTF_path_vel)
train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)



embedding_dim = 256 
patch_height = 4 
patch_width = 8 
n_heads = 8 
centroid_1 = 50 
centroid_2 = 100
n_patches = int((64//4)*(1024//8))




loaded_checkpoint = torch.load("Score.pth") 
model_parameters = loaded_checkpoint["model_state"]
torch.save(model_parameters, "model_state_Score.pth")




model = Network(embedding_dim=embedding_dim,patch_height=patch_height, patch_width=patch_width,n_patches=n_patches, n_heads=n_heads)

model.load_state_dict(torch.load("model_state_Score.pth"))
model.to(device)
model.eval()


optimizer = SGD(model.parameters(), lr=0.000005)


training_loss = [] 

mse_loss = nn.MSELoss()

def custom_loss(score, label, centroid_1, centroid_2):
    distance_1 = torch.abs(score - centroid_1)
    distance_2 = torch.abs(score - centroid_2)
    adjusted_distance = torch.where(label==1, distance_1, distance_2)
    return mse_loss(adjusted_distance, torch.zeros(adjusted_distance.shape).to(device))

def train(model, train_loader, epochs):
    for i in range(epochs):
        running_loss = 0.0 
        for n, data in enumerate(train_loader):
            image_patch = data["range_image"].to(device).float() 
            label = data["label"].to(device)
            score = model(image_patch)[0] 

            loss = custom_loss(score, label, centroid_1, centroid_2)

            # apply backprop 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

            running_loss = running_loss + loss.item()  


            
        training_loss.append(running_loss)
        print("running loss = {}, epoch = {}".format(running_loss, i+1))

        checkpoint={
            "epoch_number": i+1,
            "model_state": model.state_dict()
         }
        torch.save(checkpoint, "Score.pth")



#train(model, train_loader, 70)

#plt.plot(np.arange(70)+1, training_loss, color="g")
#plt.xlabel("number of epochs")
#plt.ylabel("training loss")
#plt.title("LAWDS network training loss curve")
#plt.show() 


positive_score = []
negative_score = []
def val(model, valid_loader, epochs):
    for i in range(epochs):
        for n, data in enumerate(valid_loader):
            image_patch = data["range_image"].to(device).float() 
            label = data["label"].to(device)
            score = model(image_patch)[0]
            if label.item() == 1:
                positive_score.append(score.item())
            else:
                negative_score.append(score.item())



val(model, train_loader, 1) 

plt.plot(np.arange(len(positive_score)), positive_score, color='r', label="Normal PCD")
plt.plot(np.arange(len(negative_score)), negative_score, color='g', label="Adverse PCD")
plt.xlabel("point cloud frames")
plt.ylabel("Score")
plt.title("Output of LAWDS Neural Network")
plt.legend()
plt.show()


#plt.plot(np.arange(len(negative_score)), negative_score)
#plt.xlabel("point cloud frames")
#plt.ylabel("Score")
#plt.title("negative")
#plt.show()

#print("positive", positive_score)
#print("negative", negative_score)






















































