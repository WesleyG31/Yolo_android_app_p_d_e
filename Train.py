if __name__ == "__main__":
    from ultralytics import YOLO
    model = YOLO("yolo11n.pt")
    model.train(data="C:\Todos mis documentos\Yolo\Project_2_Pik_Dra_Eve\data.yaml", imgsz=640, batch=6, epochs = 50, workers=8 , device=0)
