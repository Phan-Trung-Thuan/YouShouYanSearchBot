import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import types

# Define the new forward function
def custom_forward(self, x):
    # Reshape and permute the input tensor
    x = self._process_input(x)
    n = x.shape[0]

    # Expand the class token to the full batch
    batch_class_token = self.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    x = self.encoder(x)
    return x             # Return embeddings directly

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to model input size (e.g., 224x224)
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

vision_embedding_model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
vision_embedding_model.heads = nn.Identity()

# Override the forward method
vision_embedding_model.forward = types.MethodType(custom_forward, vision_embedding_model)

vision_embedding_model.eval()  # Set model to evaluation mode

def embedding(image):
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        image = vision_embedding_model(image)
        return image.squeeze()  # Return the embeddings as a 1D tensor