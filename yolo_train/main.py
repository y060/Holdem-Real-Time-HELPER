from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model.to("cuda")

# Use the model
if __name__ == '__main__':
    results = model.train(data="yolo_train/config.yaml", epochs=3, workers=1)  # train the model

# yolo detect train data=./yolo_train/config.yaml model="yolov8n.yaml" epochs=10