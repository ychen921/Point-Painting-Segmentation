import sys
import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms as T
import matplotlib.pyplot as plt

from DeepLabV3Plus_Pytorch.datasets.cityscapes import Cityscapes
from DeepLabV3Plus_Pytorch.metrics.stream_metrics import StreamSegMetrics
from DeepLabV3Plus_Pytorch.network import modeling

def predict_segmentation(ckpt_path, img_path):
    # Check gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model for prediction
    model = modeling.__dict__['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = nn.DataParallel(model).to(device)
    model.eval()

    # Data preprocessing
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    img = Image.open(img_path).convert('RGB')
    img_trans = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_trans).max(1)[1].cpu().numpy()[0]
        predictions.append(pred)

        # label to color segmented image
        colorized_preds = Cityscapes.decode_target(pred).astype('uint8')
        colorized_pred_ = Image.fromarray(colorized_preds)
        # colorized_pred_.show()
        # time.sleep(0.5)
        # colorized_pred_.close()
    
    return predictions, colorized_preds