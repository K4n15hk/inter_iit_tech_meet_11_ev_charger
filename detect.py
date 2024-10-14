from ultralytics import YOLO
import cv2 as cv
import math

model=YOLO("C:/Users/jayji/OneDrive/Documents/YOLOv8/runs/detect/train/weights/best.pt")

classNames = ["CCS1","CCS2","CCS3"]
            
cap=cv.VideoCapture("C:/Users/jayji/OneDrive/Documents/YOLOv8/Inter_IIT.mp4")
# cap.set(3,640)
# cap.set(4,360)

while True:
    success,img=cap.read()
    # results=model(img,stream=True)
    img=cv.resize(img,(1280,720))
    # cv.circle(img,(640,360),10,(0,0,255),-1)
    results=model.predict(img)
    # print(results)
    for r in results:
        boxes=r.boxes
        # print(boxes)
        for box in boxes:
            # print(box)
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            # print(x1,y1,x2,y2)
            cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),5)
            cv.circle(img,(int((x2+x1)/2),int((y2+y1)/2)),10,(255,255,0),-1)
            print(int((x2+x1)/2),int((y2+y1)/2))

            conf=math.ceil(box.conf[0]*100)/100
            # print(conf)

            cls=box.cls[0]
            cls=int(cls)
            cv.putText(img,f'{classNames[cls]} {conf}',(x1,y1-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,255),3) 

    cv.imshow("Image",img)
    if cv.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()