Refer to directory.txt for proper structure

1. Extract yolov5 and install required libraries
	cd yolov5 && pip install -r requirements.txt && cd ..
2. Download tensorflow-great-barrier-reef from kaggle
https://www.kaggle.com/c/tensorflow-great-barrier-reef/data
3. Extract dataset
4. Prepare dataset
	python prep.py
5. Run training script
	./training_script.sh