import cv2
import time
import torch
import numpy as np
from collections import deque
import pandas as pd
import os
from ultralytics import YOLO
import mmcv
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from detect_OCr import DetectionOCRPipeline


class TruckDetectionPipeline:
    def __init__(self, save_dir="./detection_results"):
        # Create directories for saving results
        self.save_dir = save_dir
        self.images_dir = os.path.join(save_dir, "images")
        self.csv_path = os.path.join(save_dir, "detection_log.csv")
        
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize YOLO model
        self.yolo_model = YOLO('./best.pt')
        
        # Initialize DetectionOCRPipeline
        self.lpr_pipeline = DetectionOCRPipeline(
            detection_config='./Weight/co_dino_5scale_swin_l_16xb1_1x_coco.py',
            detection_checkpoint='./Weight/epoch_12.pth',
            ocr_checkpoint='./TR-OCR-Wieght/checkpoint-5000'
        )
        
        # Detection parameters
        self.base_skip_frames = 32 * 1
        self.max_skip_frames = 32 * 60
        self.required_consecutive_detections = 5
        self.max_multiplier_attempts = 5
        
        # State tracking
        self.current_skip_frames = self.base_skip_frames
        self.multiplier_attempt = 0
        self.detection_buffer = deque(maxlen=5)
        self.detection_frames = deque(maxlen=5)  # Store actual frames

        if not os.path.exists(self.csv_path):
            self.create_csv()

    def create_csv(self):
        df = pd.DataFrame(columns=[
            'timestamp',
            'license_plate',
            'confidence',
            'image_path'
        ])
        df.to_csv(self.csv_path, index=False)
        print(f"Created new log file at: {self.csv_path}")
    
    def detect_truck(self, frame):
        results = self.yolo_model(frame, conf=0.5)
        return len(results[0].boxes) > 0  

    def process_saved_image(self, image_path, conf_threshold=0.6):
        ### Process a saved image through the LPR pipeline

        print(f"Processing saved image: {image_path}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None
        
        bbox, license_plate, confidence = self.lpr_pipeline.process_frame(frame, conf_threshold)
        print(f"LPR Results - License Plate: {license_plate}, Confidence: {confidence}")
        
        # Only return results if confidence meets threshold
        if confidence and confidence >= conf_threshold:
            return license_plate, confidence
        else:
            print(f"License plate detection confidence ({confidence}) below threshold ({conf_threshold})")
            # Delete the image file as it didn't pass the threshold
            # try:
            #     os.remove(image_path)
            #     print(f"Deleted image: {image_path}")
            # except OSError as e:
            #     print(f"Error deleting image: {e}")
            return None, None
        

    def save_detection(self, frame):
        """Save detected frame and update log"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(self.images_dir, f"detected_truck_{timestamp}.jpg")
        
        # Save the image
        cv2.imwrite(image_path, frame)
        print(f"Saved frame to: {image_path}")
        
        # Process the saved image
        license_plate, confidence = self.process_saved_image(image_path)
        
        if license_plate:
            # Update CSV log
            new_row = [timestamp, license_plate, confidence, image_path]
            df = pd.DataFrame([new_row], columns=['timestamp', 'license_plate', 'confidence', 'image_path'])
            df.to_csv(self.csv_path, mode='a', header=False, index=False)
            print(f"Logged detection: {new_row}")
            return True
        else:
            print("No license plate detected")
            return False

    def process_video(self, video_path):
        print(f"\nStarting video processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret:
                break
            
            truck_detected = self.detect_truck(frame)
            
            if truck_detected:
                print(f"Truck detected in frame {frame_count}")
                success = self.save_detection(frame)
                
                if success:
                    print("Successfully saved and processed detection")
                    frame_count += 32 * 60  # Big skip after successful LP detection
                else:
                    print("LP detection failed, checking next frame")
                    frame_count += 10  # Move to very next frame when LP fails
            else:
                print(f"No truck detected at frame {frame_count}")
                # Use skip algorithm only when no truck detected
                if self.multiplier_attempt >= self.max_multiplier_attempts:
                    self.current_skip_frames *= 2
                    self.multiplier_attempt = 0
                    
                    if self.current_skip_frames > self.max_skip_frames:
                        self.current_skip_frames = self.max_skip_frames
                    print(f"Increased skip frames to: {self.current_skip_frames}")
                else:
                    self.multiplier_attempt += 1
                    
                frame_count += self.current_skip_frames
                print(f"Skipping to frame {frame_count}")
        
        cap.release()
        print(f"Video processing complete. Total frames processed: {frame_count}")

# Usage
if __name__ == "__main__":
    pipeline = TruckDetectionPipeline()
    pipeline.process_video("./video.mp4")  # Replace with your video path