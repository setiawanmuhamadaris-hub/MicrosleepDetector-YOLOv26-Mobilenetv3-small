import torch
import torch.nn as nn
import torchvision.transforms as T
import ultralytics.nn.modules.block as block
from ultralytics import YOLO
from torchvision.models import mobilenet_v3_small
import streamlit as st

class AttentionEnhancedBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = block.Conv(c1, c_, k[0], 1)
        self.cv2 = block.Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2 // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2 // 4, c2, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.cv2(self.cv1(x))
        out = out * self.se(out)
        return x + out if self.add else out

# Terapkan patch secara global untuk YOLOv26
block.Bottleneck = AttentionEnhancedBottleneck

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load YOLOv26
    yolo = YOLO('models/best.pt') 
    
    # Load MobileNetV3
    mobilenet = mobilenet_v3_small()
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)
    mobilenet.load_state_dict(torch.load('models/mobilenetv3_best.pth', map_location=device, weights_only=True))
    mobilenet = mobilenet.to(device)
    mobilenet.eval()
    
    return yolo, mobilenet, device

def get_transform():
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
