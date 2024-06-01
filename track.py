import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.track(source='',
                project='',
                name='',
                save=True
                )