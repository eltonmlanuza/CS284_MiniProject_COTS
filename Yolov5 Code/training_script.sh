cd yolov5

python train.py --workers 2 --img 1280 --batch 16 --epochs 300 --name cots_m6 --data ../cots_dataset.yml --weights yolov5m6.pt