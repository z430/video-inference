import cv2
import numpy as np
import torch

from .utils import nvtx_range


class YOLOv5:
    def __init__(self):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="/opt/ml/models/yolov5l.engine"
        )
        self.img_size = 640

    def predict(self, img):
        with nvtx_range("preprocessing"):
            img = self.preprocess(img)
        with nvtx_range("inference"):
            predictions = self.model(img, size=self.img_size)
        # return xyxy format

    def preprocess(self, img):
        target_height, target_width = 1088, 640
        h, w, _ = img.shape
        scale = min(target_width / w, target_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create a new image of the target size and fill it with padding
        img_padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        top, left = (target_height - new_h) // 2, (target_width - new_w) // 2
        img_padded[top : top + new_h, left : left + new_w, :] = img_resized

        # Convert to float32 and normalize to [0, 1]
        img_padded = img_padded.astype(np.float32) / 255.0

        # Convert to PyTorch tensor
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0)

        # Normalize with mean and standard deviation (used in YOLOv5)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Move the tensor to GPU
        img_tensor = img_tensor.to("cuda")
        return img_tensor
