from data_synthesizer import *
from ultralytics import YOLO

if __name__ == '__main__':

     #visualize_random_images(1)

     #create_cocoish_dataset()

     # Load a model
     #model = YOLO('yolov8n.yaml')  # build a new model from scratch
     model = YOLO(model="yolov8n.pt", task = "pose")  # load a pretrained model (recommended for training)
     results = model.train(data="coco-vatem.yaml", epochs=700)  # train the model


     #results = model.val()  # evaluate model performance on the validation set
     #results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
     #success = model.export(format='onnx')  # export the model to ONNX format