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
cap = cv.VideoCapture(0)

while True:
    #get frames
    ret, frame = cap.read()
    
    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    print('class ids', class_ids)
    print('scores', scores)
    print('bboxes', bboxes)
    
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x,y,w,h) = bbox
        print('x,y,w,h: ', x,y,w,h)
        
        class_name = classes[class_id]
        cv.putText(frame, class_name, (x, y-10),cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255),3)
        cv.rectangle(frame, (x,y), (x+w, y+h), (250,50,10), 2)
    
    cv.imshow('Frames', frame)
    k = cv.waitKey(1)
    if k==ord('q'):
        break
    
cv.destroyAllWindows()