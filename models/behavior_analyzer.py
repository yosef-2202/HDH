import cv2
import random
from datetime import datetime
import time
from ultralytics import YOLO

class BehaviorAnalyzer:
    def __init__(self, settings_ref):
        self.settings = settings_ref
        # Tải model YOLOv8 pose nano (nhẹ nhất, tự động tải lần đầu chạy)
        self.model = YOLO('yolov8n-pose.pt') 
        self.latest_data = None
        self.stats = {"normal": 0, "abnormal": 0}

    def analyze_frame(self, frame):
        # Predict pose, verbose=False để ẩn log trong terminal
        results = self.model(frame, verbose=False)
        
        analysis_data = None
        
        # Nếu phát hiện ra người và có vẽ được khung xương
        if results and len(results[0].boxes) > 0:
            frame = results[0].plot() # YOLO tự động vẽ khung xương lên frame
            
            # Giả lập logic phân tích hành vi
            is_abnormal = random.random() > 0.85 
            
            behavior = "Hành vi bình thường"
            danger_level = "Thấp"
            color = (0, 255, 0)
            
            if self.settings.danger_analysis_enabled and is_abnormal:
                behavior = "Hành vi bất thường (Đánh nhau/Vấp ngã)"
                danger_level = "Cao"
                color = (0, 0, 255)
            elif not self.settings.danger_analysis_enabled:
                behavior = "Đang theo dõi"
                danger_level = "Không xác định"
                color = (255, 0, 0)

            analysis_data = {
                "id": random.randint(1, 5),
                "features": "Người lớn",
                "behavior": behavior,
                "danger_level": danger_level,
                "time": datetime.now().strftime("%H:%M:%S"),
                "details": "Đã phát hiện chuyển động"
            }
            
            cv2.putText(frame, behavior, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        return frame, analysis_data

    def generate_frames(self, source=0):
        try:
            source = int(source)
        except ValueError:
            pass 
            
        cap = cv2.VideoCapture(source)
        
        try:
            while cap.isOpened():
                if not self.settings.camera_enabled:
                    time.sleep(1)
                    continue
                    
                success, frame = cap.read()
                if not success:
                    if isinstance(source, str) and not source.startswith('rtsp'):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                    
                frame = cv2.resize(frame, (640, 480))
                frame, data = self.analyze_frame(frame)
                
                if data:
                    self.latest_data = data
                    if data['danger_level'] == 'Cao':
                        self.stats['abnormal'] += 1
                    elif data['danger_level'] == 'Thấp':
                        self.stats['normal'] += 1
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except GeneratorExit:
            pass 
        finally:
            cap.release()