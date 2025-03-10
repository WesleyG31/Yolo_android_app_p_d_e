from ultralytics import YOLO
model = YOLO("C:\Todos mis documentos\Yolo\Project_2_Pik_Dra_Eve\pik_dr_eve.pt")
model.predict(source="0", show=True, save=True,conf=0.7, line_width=2,save_crop=False,save_txt=False, show_labels=True,show_conf=True )