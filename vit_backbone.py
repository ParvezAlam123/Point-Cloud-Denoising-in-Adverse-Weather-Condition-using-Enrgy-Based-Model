import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 




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
    def __init__(self, d_model, n_patches, n_heads=8):
        super().__init__()

        self.n_heads = n_heads
        self.n_patches = n_patches 

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

        

        self.batchnorm = nn.BatchNorm1d(num_features=self.n_patches)  # 4 is the number of level in backbone
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        """ x is in shape 1, Batch*num_level, dim"""
        B, num_level, dim = x.shape
        # reshape dimension 
        x = x.reshape(self.n_heads, B, num_level, self.d_head)

        head_feature1 = self.head1(x[0])
        head_feature2 = self.head2(x[1])
        head_feature3 = self.head3(x[2]) 
        head_feature4 = self.head4(x[3])
        head_feature5 = self.head5(x[4])
        head_feature6 = self.head6(x[5])
        head_feature7 = self.head7(x[6])
        head_feature8 = self.head8(x[7])



        feature = torch.cat((head_feature1, head_feature2, 
                            head_feature3, head_feature4, head_feature5, head_feature6, head_feature7, head_feature8), dim=-1)
        

        
        x = x.reshape(B, num_level, dim) # shape into original dimension

        # add batch norm and residal connection
        feature = x + feature
        feature = self.batchnorm(feature)

        # add mlp
        output = F.relu(self.fc(feature))

        # add residual connection 

        output = feature + output 
        output = self.batchnorm(output)

        return output 





class Embedding(nn.Module):
    def __init__(self, embedding_dim, patch_height, patch_width):
        super().__init__() 
        self.embedding_dim = embedding_dim 
        self.patch_height = patch_height 
        self.patch_width = patch_width 

        self.fc1 = nn.Linear(self.patch_height*self.patch_width*3, 128)
        self.fc2 = nn.Linear(128, self.embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x))
        return out 




class Encoder(nn.Module):
    def __init__(self, d_model, n_patches, n_heads):
        super().__init__() 
        self.d_model = d_model 
        self.n_patches = n_patches 
        self.n_heads = n_heads 

        # add multiheaded attention blocks 
        self.attention_block1 = MultiHeadAttention(d_model=self.d_model, n_patches=self.n_patches, n_heads=self.n_heads)
        #self.attention_block2 = MultiHeadAttention(d_model=self.d_model, n_patches=self.n_patches, n_heads=self.n_heads)
        #self.attention_block3 = MultiHeadAttention(d_model=self.d_model, n_patches=self.n_patches, n_heads=self.n_heads)
        


    def forward(self, x):
        block1_features = self.attention_block1(x)
        #block2_features = self.attention_block2(block1_features)
        #block3_features = self.attention_block3(block2_features)
        

        return block1_features 
    
     
    
    


class Network(nn.Module):
    def __init__(self, embedding_dim, patch_height, patch_width, n_patches, n_heads):
        super().__init__() 

        self.embedding_dim = embedding_dim 
        self.patch_height = patch_height 
        self.patch_width = patch_width 
        self.n_patches = n_patches  
        self.n_heads = n_heads 

        #self.embedding = Embedding(embedding_dim=self.embedding_dim, patch_height=self.patch_height, patch_width=self.patch_width)
        #self.encoder = Encoder(d_model=self.embedding_dim, n_patches=self.n_patches, n_heads=self.n_heads) 

        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=6)
        self.conv2d_2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=6)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=6)


        # prediction layers 
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        #self.fc4 = nn.Linear(32, 1) 
        


    def forward(self, x):
        B, H, w, c  = x.shape
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = x.reshape(B, 64, -1)
        # reshape (flatten) patch 
        #x = x.reshape(B, N, -1)
        #x = self.embedding(x)
        #x = self.encoder(x)

        # maxpool 
        max_pool_features = torch.max(x, dim=2)[0]
        
        # apply linear layers for score regression 
        x = F.relu(self.fc1(max_pool_features))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        #out = self.fc4(x)
        return out 
    




       
    


    
    









