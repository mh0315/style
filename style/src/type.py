from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image_as_tensor(path):
    img_pil = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    return img_tensor
content_path = os.path.join( "picture", "content.jpg")
style_path   = os.path.join( "picture", "style.jpg")

content_t = load_image_as_tensor(content_path)
style_t   = load_image_as_tensor(style_path)

vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
vgg_feature = vgg.features.to(device)
for param in vgg_feature.parameters():
    param.requires_grad = False
vgg_feature.eval()

def content_feature(img_tensor,layer_idx=21):
    x = img_tensor
    for i, layer in enumerate(vgg_feature):
        x = layer(x)
        if i == layer_idx:
            return x
        

def style_features(img_tensor,layer_idx = [5,10,16,19,28] ):
    x = img_tensor
    features = []
    for i, layer in enumerate(vgg_feature):
        x = layer(x)
        if i in layer_idx:
            features.append(x)
    return features


def gram_matrix(features):
    B, C, H, W = features.shape
    features_reshaped = features.view(B, C, H*W)
    G = features_reshaped @ features_reshaped.transpose(1, 2)
    G = G / (C * H * W)
    return G

def style_transfer_optimized(
    content_tensor, style_tensor,
    alpha,   
    beta,    
    gamma,  
    learning_rate,
    steps
):
    generated_tensor = content_tensor.clone().detach()
    generated_tensor.requires_grad_(True)

    with torch.no_grad():
        style_feats = style_features(style_tensor, layer_idx=[5,10,16,19,28] )
        style_gram_m = [gram_matrix(fm) for fm in style_feats]
        content_feats = content_feature(content_tensor, layer_idx=21)

    optimizer = torch.optim.Adam([generated_tensor], lr=learning_rate)

    for i in range(steps):
        optimizer.zero_grad()
        gen_feats1 = content_feature(generated_tensor, layer_idx=21)
        content_loss = 0
        content_loss = F.mse_loss(gen_feats1, content_feats)

        gen_feats2 = style_features(generated_tensor, layer_idx=[5,10,16,19,28])
        gen_gram = [gram_matrix(fm) for fm in gen_feats2]
        style_loss = 0
        for gen_g, style_g in zip(gen_gram, style_gram_m):
            style_loss += F.mse_loss(gen_g, style_g)

        tv_loss = torch.sum(torch.abs(generated_tensor[:, :, :, :-1] - generated_tensor[:, :, :, 1:])) + torch.sum(torch.abs(generated_tensor[:, :, :-1, :] - generated_tensor[:, :, 1:, :]))
        loss = alpha * content_loss + beta * style_loss + gamma * tv_loss
        loss.backward()
        optimizer.step()
    return generated_tensor

generated_t = style_transfer_optimized(
    content_tensor=content_t,
    style_tensor=style_t,
    alpha=1.0,     
    beta=1e5,      
    gamma=1e-4,    
    learning_rate=0.01,
    steps=800   
)

def save_tensor_as_image(img_tensor, save_path):
    
    img = img_tensor.squeeze(0).detach().cpu()
    
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    
   
    img = torch.clamp(img, 0, 1)
    
   
    img = img.permute(1, 2, 0).numpy()
    
    
    img = (img * 255).astype(np.uint8)
    
    
    Image.fromarray(img).save(save_path)
    print(f"图像已保存至: {save_path}")


save_tensor_as_image(generated_t, "output.jpg")