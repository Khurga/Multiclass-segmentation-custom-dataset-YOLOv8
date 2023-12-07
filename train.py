from ultralytics import YOLO

model = YOLO('model_name')  # load a pretrained model here for training

model.train(data='custom_data.yaml', epochs=500, imgsz=640) # load a configured yaml file, input number of epochs and image size
