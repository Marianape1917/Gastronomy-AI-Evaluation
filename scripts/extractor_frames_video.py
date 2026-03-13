import cv2
import os
import numpy as np

def extract_fixed_frames(video_path, output_folder, num_frames=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) 
    else:
        frame_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    
    saved_count = 0
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(output_folder, f"frame_{saved_count:03d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved_count += 1
    
    cap.release()
    print(f"Se guardaron {saved_count} frames en {output_folder}")


root_folder = "/home/ubuntu22/Documentos/proyecto-integrador"
output_root = "/home/ubuntu22/Documentos/proyecto-integrador/frames"

for platillo in os.listdir(root_folder):
    platillo_path = os.path.join(root_folder, platillo)
    if os.path.isdir(platillo_path):
        for video_file in os.listdir(platillo_path):
            if video_file.lower().endswith(".mov"):
                video_path = os.path.join(platillo_path, video_file)
                
                if "CHEF" in video_file.upper():
                    category = "Chef"
                else:
                    category = "Alumnos"
                
                output_folder = os.path.join(output_root, platillo.replace(" ", "_"), category, os.path.splitext(video_file)[0])
                extract_fixed_frames(video_path, output_folder, num_frames=30)
