import cv2 as cv

#OpenCV DNN
net = cv.dnn.readNet('dnn_model\yolov4-tiny.weights',
                     'dnn_model\yolov4-tiny.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)

#load classes
classes=[]
with open('dnn_model/classes.txt','r') as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print('Objects lists: ', classes)

#Intialize camera
img = cv.imread('G:/MyImages/cars.jpg')
img = cv.resize(img, (400,400))
(class_ids, scores, bboxes) = model.detect(img)
for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x,y,w,h) = bbox
        print('x,y,w,h: ', x,y,w,h)
        
        class_name = classes[class_id]
        cv.putText(img, class_name, (x, y-10),cv.FONT_ITALIC, 1, (200, 40, 50),2)
        cv.rectangle(img, (x,y), (x+w, y+h), (250,50,100), 2)
    
cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()