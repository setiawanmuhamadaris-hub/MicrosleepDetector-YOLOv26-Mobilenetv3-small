import cv2
import torch
import time
from models_handler import load_models, get_transform

class_names = ['Sadar', 'Microsleep']

def get_yolo_and_mobilenet():
    return load_models()

def process_frame_local(frame, conf_threshold, alert_duration, start_sleep_time, fps=0.0):
    yolo_model, mobilenet_model, device = get_yolo_and_mobilenet()
    transform = get_transform()
    
    results = yolo_model(frame, verbose=False, conf=conf_threshold)[0]
    current_status = "Sadar"
    eyes_closed = False
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        h, w, _ = frame.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop_img = frame[y1:y2, x1:x2]
        
        if crop_img.size > 0:
            input_tensor = transform(crop_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = mobilenet_model(input_tensor)
                _, preds = torch.max(outputs, 1)
                label = class_names[preds.item()]
                
            if label == 'Microsleep':
                eyes_closed = True
                
            color = (0, 0, 255) if label == 'Microsleep' else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    current_time = time.time()
    if eyes_closed:
        if start_sleep_time is None:
            start_sleep_time = current_time
        elapsed = current_time - start_sleep_time
        
        cv2.putText(frame, f"Microsleep: {elapsed:.1f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        if elapsed >= alert_duration:
            current_status = "Microsleep"
            cv2.putText(frame, "BAHAYA! BANGUN!", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 10)
    else:
        start_sleep_time = None
        cv2.putText(frame, "Status: AMAN", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Render FPS overlay di sudut kanan atas
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, current_status, start_sleep_time
