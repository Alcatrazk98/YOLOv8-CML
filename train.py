import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('')
    model.load('')
    model.train(data='',
                cache=False,
                project='',
                name='',
                epochs=100,
                batch=2,
                close_mosaic=10,
                optimizer='',
                )