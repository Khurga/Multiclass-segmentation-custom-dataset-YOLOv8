from ultralytics import YOLO

# Load a pretrained YOLOv8n-seg Segment model

model_path = '.../weights/best.pt'
model = YOLO(model_path)

model = YOLO('best.pt')

#model = YOLO('best_yolov8n-seg.pt')


model.predict(source = 'img', show=True, save=True, hide_labels=False, hide_conf=True, conf=0.5, save_txt=False, save_crop=False, line_thickness=2)


