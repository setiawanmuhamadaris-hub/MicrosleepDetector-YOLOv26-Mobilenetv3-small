import cv2
import torch
import time
import av
from models_handler import load_models, get_transform

class_names = ['Sadar', 'Microsleep']

def get_yolo_and_mobilenet():
    return load_models()

class VideoProcessor:
    def __init__(self):
        self.start_sleep_time = None
        self.conf_threshold = 0.5
        self.alert_duration = 2.0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        yolo_model, mobilenet_model, device = get_yolo_and_mobilenet()
        transform = get_transform()
        
        results = yolo_model(img, verbose=False, conf=self.conf_threshold)[0]
        eyes_closed = False
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            h, w, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            crop_img = img[y1:y2, x1:x2]
            
            if crop_img.size > 0:
                input_tensor = transform(crop_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = mobilenet_model(input_tensor)
                    _, preds = torch.max(outputs, 1)
                    label = class_names[preds.item()]
                
                if label == 'Microsleep':
                    eyes_closed = True
                
                color = (0, 0, 255) if label == 'Microsleep' else (0, 255, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
        current_time = time.time()
        if eyes_closed:
            if self.start_sleep_time is None:
                self.start_sleep_time = current_time
            elapsed = current_time - self.start_sleep_time
            cv2.putText(img, f"Microsleep: {elapsed:.1f}s", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            if elapsed >= self.alert_duration:
                cv2.putText(img, "BAHAYA! BANGUN!", (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,0,255), 10)
        else:
            self.start_sleep_time = None
            cv2.putText(img, "Status: AMAN", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_frame_local(frame, conf_threshold, alert_duration, start_sleep_time):
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
                    
    return frame, current_status, start_sleep_time
