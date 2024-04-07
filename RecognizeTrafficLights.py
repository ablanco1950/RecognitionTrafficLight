# -*- coding: utf-8 -*-
"""
Created on Apr 2024

@author:Alfonso Blanco from several sources

https://medium.com/@mayank.siddharth/image-segmentation-lightning-fast-yolov8-unleashed-no-gpu-required-with-a-sleek-6-7mb-model-7854cb29c54d
"""
#######################################################################
# PARAMETERS
######################################################################
#
# Downloaded from https://github.com/ikramnoun/self-driving-car

dirVideo ="Grays - GEC Elliot, Plessey Mellors & PEEK Elite - Microsense Pelican Crossing Traffic Lights, Essex - Trim.mp4"


import cv2
import cvzone
import supervision as sv
import numpy as np
from ultralytics import YOLO

# Loading pretrained YOLOv8 models for segmentation
model = YOLO('yolov8n-seg.pt')
class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
        "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
        "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"
    ]



###########################################################
# MAIN
##########################################################
cap = cv2.VideoCapture(dirVideo)

# https://levelup.gitconnected.com/opencv-python-reading-and-writing-images-and-videos-ed01669c660c
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
fps=5.0
frame_width = 680
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

video_writer = cv2.VideoWriter('demonstration.mp4',fourcc,fps, size) 
ContFrames=0
ContDetected=0
ContNoDetected=0
while (cap.isOpened()):
     ret, img = cap.read()
     if ret != True: break
     else:
        # Predicting on our image
        results = model(img)

        # Getting detections
        detections = sv.Detections.from_ultralytics(results[0])

        # https://roboflow.com/how-to-filter-detections/yolov8
        detections = detections[detections.class_id == 9]
        detections = detections[detections.confidence > 0.5]

        
        # Export as xyxy data
        bounding_boxes = detections.xyxy
        #if bounding_boxes==[]: continue
        if bounding_boxes.size==0: continue
        #print(bounding_boxes)
        #https://supervision.roboflow.com/detection/core/
        x_min=bounding_boxes[0][0]
        y_min=bounding_boxes[0][1]
        x_max=bounding_boxes[0][2]
        y_max=bounding_boxes[0][3]
        #This code will convert your bounding box data into the xyxy format. You can find more details in the Ultimate Guide to Converting Bounding Boxes, Masks and Polygons 9.
        #Ultimate Guide to Converting Bounding Boxes, Masks and Polygons (roboflow.com) 
        #https://blog.roboflow.com/convert-bboxes-masks-polygons/
        # Leo Ueno. (Aug 15, 2023). Ultimate Guide to Converting Bounding Boxes, Masks and Polygons. Roboflow Blog: https://blog.roboflow.com/convert-bboxes-masks-polygons/
        # STEP 2: The program detects thecolor of the object that user chose.
        # inside of rectangle that user draw
        # Object Tracking with Mean shift and Cam shift Algorithms using OpenCV | by siromer | The Deep Hub | Mar, 2024 | Medium 
        #  https://medium.com/thedeephub/object-tracking-with-mean-shift-and-cam-shift-algorithms-using-opencv-fc2f30327199
        #print(bounding_boxes[0])
        object_image=img[int(y_min):int(y_max),int(x_min):int(x_max),:]

        
        # Object Tracking with Mean shift and Cam shift Algorithms using OpenCV | by siromer | The Deep Hub | Mar, 2024 | Medium 
        # https://medium.com/thedeephub/object-tracking-with-mean-shift-and-cam-shift-algorithms-using-opencv-fc2f30327199
        hsv_object=cv2.cvtColor(object_image,cv2.COLOR_BGR2HSV)    
       
        start_point=(int(x_min),int(y_min)) 
        end_point=(int(x_max), int(y_max))
     
        # Draw a rectangle with blue line borders of thickness of 5 px
        img = cv2.rectangle(img, start_point, end_point,(255,0,0), 2)
        # Put text
        text_location = (int(x_min), int(y_min))
        text_color = (255,255,255)

       
        # https://github.com/ikramnoun/self-driving-car

        lower_red = np.array([150, 70, 50])
        upper_red = np.array([180, 255, 255])

        lower_yellow = np.array([21, 39, 64])
        upper_yellow = np.array([40, 255, 255])

        lower_green = np.array([60, 100, 100])
        upper_green = np.array([80, 255, 255])

        mask_red = cv2.inRange(hsv_object, lower_red, upper_red)
        mask_yellow = cv2.inRange(hsv_object, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv_object, lower_green, upper_green)
        text=""
        if mask_red.any():
              text="Stop mandatory"
            
        if mask_yellow.any():
              text="Precaution pedestrians"
              
        if mask_green.any():
               text="May pass"
                            
        cv2.putText(img, text ,text_location
                     , cv2.FONT_HERSHEY_SIMPLEX , 1
                     , text_color, 2 ,cv2.LINE_AA)
      
        cv2.imshow('Frame', img)
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'): break
        """
        # Instantiating annotators
        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_scale=1,text_thickness=2)
        mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

        # Getting the labels
        labels = [
            model.model.names[class_id]
            for class_id in detections.class_id
        ]

        # Annotating the image
        annotated_image = bounding_box_annotator.annotate(scene=img, 
                                                  detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, 
                                           detections=detections, 
                                           labels=labels)
        annotated_image = mask_annotator.annotate(scene=annotated_image, 
                                          detections=detections)
        #sv.plot_image(annotated_image)
        
                          
        """  
cap.release()
video_writer.release()
cv2.destroyAllWindows()
          
print("End")           


