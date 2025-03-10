from ultralytics import YOLO
import cv2

#Read the model
model = YOLO("C:\Todos mis documentos\Yolo\Project_2_Pik_Dra_Eve\pik_dr_eve.pt")

#Open Camera
cap = cv2.VideoCapture(0)

#Infinite bucle to keep open
while True:
    #Read videocamera
    ret, frame= cap.read()
    
    #Invertir sentido (mirror)
    frame = cv2.flip(frame, 1)

    #read results
    results=model.predict(frame, imgsz=640, conf= 0.70)

    #show results
    anotacion= results[0].plot()

    #Show image
    cv2.imshow('Pikachu Video', anotacion)

    #close the program
    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()
