# video-inference

## Compile YOLOv5 Model TensorRT

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
python3 export.py --weights yolov5l.pt --include engine --imgsz 1088 640 --half --device 0
```
