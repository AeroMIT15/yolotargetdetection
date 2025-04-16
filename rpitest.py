import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2

def main():
    # Load model (convert to ONNX first for better Pi performance)
    model_path = 'best.onnx'  # Convert your model: model.export(format='onnx')
    model = YOLO(model_path)
    
    # PiCamera2 setup
    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (640, 480)},  # Lower resolution for better FPS
        controls={"FrameRate": 30}
    )
    picam2.configure(config)
    picam2.start()
    
    # Warmup camera
    time.sleep(2)
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame from libcamera
            frame = picam2.capture_array()
            
            # Convert BGR to RGB (libcamera uses RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Start timer for FPS
            start_time = time.time()
            
            # Run inference (use half precision for speed)
            results = model(frame, 
                          conf=0.5,
                          imgsz=320,  # Smaller inference size
                          half=True)   # FP16 for faster inference
            
            # Calculate FPS
            fps = 1.0 / (time.time() - start_time)
            
            # Annotate
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, 
                       f"FPS: {fps:.1f} | Pi 5", 
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            
            # Display
            cv2.imshow("Pi Camera Detection", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
