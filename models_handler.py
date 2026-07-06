import torch
import torch.nn as nn
import torchvision.transforms as T
import ultralytics.nn.modules.block as block
from ultralytics import YOLO
from torchvision.models import mobilenet_v3_small
import streamlit as st
import sys

# AttentionEnhancedBottleneck: disesuaikan dengan arsitektur YOLOv26n
# yang tersimpan di best.pt (tanpa blok SE — agar tidak error saat load checkpoint)
class AttentionEnhancedBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = block.Conv(c1, c_, k[0], 1)
        self.cv2 = block.Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out

# Patch global: ganti Bottleneck Ultralytics dengan versi YOLOv26
block.Bottleneck = AttentionEnhancedBottleneck

# Daftarkan ke __main__ agar pickle menemukan class yang benar saat load checkpoint
# (model YOLOv26 disimpan dengan referensi ke __main__.AttentionEnhancedBottleneck)
sys.modules['__main__'].AttentionEnhancedBottleneck = AttentionEnhancedBottleneck

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load YOLOv26n
    yolo = YOLO('models/best.pt')

    # Load MobileNetV3-Small (2 kelas: Sadar / Microsleep)
    mobilenet = mobilenet_v3_small()
    mobilenet.classifier[3] = nn.Linear(mobilenet.classifier[3].in_features, 2)
    mobilenet.load_state_dict(
        torch.load('models/mobilenetv3_best.pth', map_location=device, weights_only=True)
    )
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
