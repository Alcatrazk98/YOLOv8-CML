import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.predict(source='',
                project='',
                name='',
                save=True,
                )