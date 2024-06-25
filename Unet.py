import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels, mid_channels=None):
        super().__init__() 

        mid_channels = out_channels 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat((x2, x1), dim=1)
        x = self.conv(x)
        return x 
    



class AttentionHead(nn.Module):
    def __init__(self, d_head):
        super().__init__() 

        self.d_head = d_head 

        self.fc1 = nn.Linear(self.d_head, 2 * self.d_head)
        self.fc2 = nn.Linear(2 * self.d_head, self.d_head)

    def forward(self, x):
        B, num_level, d_head = x.shape 

        scale = np.sqrt(2 * d_head)

        query = self.fc1(x)
        key = F.relu(self.fc1(x))
        value = self.fc1(x)

        attention_weight = torch.bmm(query, key.transpose(1,2)) / scale 
        attention_weight = F.softmax(attention_weight, dim=-1)


        feature = torch.bmm(attention_weight, value)

        feature = F.relu(self.fc2(feature))

        return feature 





class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=8):
        super().__init__()

        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads 

        self.head1 = AttentionHead(self.d_head)
        self.head2 = AttentionHead(self.d_head)
        self.head3 = AttentionHead(self.d_head)
        self.head4 = AttentionHead(self.d_head)
        self.head5 = AttentionHead(self.d_head)
        self.head6 = AttentionHead(self.d_head)
        self.head7 = AttentionHead(self.d_head)
        self.head8 = AttentionHead(self.d_head)

        

        self.batchnorm = nn.BatchNorm1d(num_features=4)  # 4 is the number of level in backbone
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        """ x is in shape B, dim, H, W """

        B, dim, H, W = x.shape 
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(B, H*W, dim)
        # reshape dimension 
        x = x.reshape(self.n_heads, B, H*W, self.d_head)

        head_feature1 = self.head1(x[0])
        head_feature2 = self.head2(x[1])
        head_feature3 = self.head3(x[2])
        head_feature4 = self.head4(x[3])
        head_feature5 = self.head5(x[4])
        head_feature6 = self.head6(x[5])
        head_feature7 = self.head7(x[6])
        head_feature8 = self.head8(x[7])


        feature = torch.cat((head_feature1, head_feature2, 
                            head_feature3, head_feature4, head_feature5, head_feature6,head_feature7, head_feature8), dim=-1)

        
        x = x.reshape(B, H*W, dim) # shape into original dimension

        # add batch norm and residal connection
        feature = x + feature
        feature = self.batchnorm(feature)

        # add mlp
        output = F.relu(self.fc(feature))

        # add residual connection 

        output = feature + output 
        output = self.batchnorm(output) 
        print("hello")

        return output 







class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__() 
        self.n_channels = n_channels 
        self.n_classes = n_classes 

        self.inc = DoubleConv(self.n_channels, 64) 

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=1)



    def forward(self, x): 
        print("input dimentsion  = ", x.shape)
        x_emb = self.inc(x) 
        print("embedding dim = ", x_emb.shape)
        
        x1 = self.down1(x_emb) 
        print("x1 = ", x1.shape)
        x2 = self.down2(x1)
        print("x2 = ", x2.shape)
        x3 = self.down3(x2)
        print("x3 = ", x3.shape)
        x4 = self.down4(x3)
        print("x4 = ", x4.shape)

        x = self.up1(x4, x3)  
        print("merge 4, 3 = ", x.shape)
        x = self.up2(x, x2) 
        print("merge 2 = ", x.shape)
        x = self.up3(x, x1)
        print("merge 1= ", x.shape)
        x = self.up4(x, x_emb) 
        print("merge emb = ", x.shape)

        x = F.relu(self.conv1(x)) 
        print("after conv1 = ", x.shape)
        out = self.conv2(x)
        print("out = ", out.shape)

        return out 
    



        

        












