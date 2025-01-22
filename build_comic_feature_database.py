import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm

class ModifiedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load the pre-trained ResNet18 model
        original_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Copy all layers up to layer4
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4

    def forward(self, x):
        # Forward pass, stopping after layer4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

YSY_COMIC_PATH = 'You.Shou.Yan-comic-en'
target_size = (16000, 900)

model = ModifiedResNet()
model = model.eval().to('cuda')

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# print(model)

def pad_to_target(image, target_shape):
    current_height, current_width = image.shape[:2]
    target_height, target_width = target_shape

    pad_height = target_height - current_height
    pad_width = target_width - current_width

    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    try:
        padded_image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    except:
        print(image.shape)
    
    padded_image = cv2.resize(padded_image, (target_shape[0] // 2, target_shape[1] // 2))

    return padded_image

batch_size = 16
batch_images = []
for i, image_file in tqdm(enumerate(Path(YSY_COMIC_PATH).glob("*.jpg"))):
    image = cv2.imread(str(image_file))

    if image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    padded_image = pad_to_target(image, target_size)

    transformed_image = transform(padded_image)

    if i % batch_size == 0 and i > 0:
        batch_tensor = torch.stack(batch_images).to('cuda')
        with torch.no_grad():
            feature_maps = model(batch_tensor).to('cpu')
        print(feature_maps.shape)
        torch.save(feature_maps, f'/content/saved/YSY_comic_feature_map_{i}.pth')
        batch_images = [transformed_image]
    else:
        batch_images.append(transformed_image)