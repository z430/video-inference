import torch


class YOLOv5:
    def __init__(self, path):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path="/opt/ml/model/yolov5l.engine"
        )
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.img_size = 1088

    def predict(self, img):
        predictions = self.model(img, size=self.img_size)
        # return xyxy format
        return predictions.xyxy[0].cpu().numpy()
