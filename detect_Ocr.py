import cv2
import os
import time
import threading
import numpy as np
import torch
import mmcv
from PIL import Image
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class DetectionOCRPipeline:
    def init(self, detection_config, detection_checkpoint, ocr_checkpoint, device='cuda:0'):
        # Initialize detection model
        self.det_model = init_detector(detection_config, detection_checkpoint, device=device)
        self.visualizer = DetLocalVisualizer()

        # Initialize OCR model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(ocr_checkpoint)
        self.ocr_model.to(self.device)

    def extract_digits(self, text):
        return ''.join(filter(str.isdigit, str(text)))

    def get_crops(self, image, boxes, padding=10):
        crops = []
        height, width = image.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Add padding while keeping within image bounds
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            crop = image[y1:y2, x1:x2]
            crops.append((crop, (x1, y1, x2, y2)))

        return crops

    def process_frame(self, frame, conf_threshold=0.3):
        if frame is None:
            return None, None, None

        # Convert frame for detection
        image_rgb = mmcv.imconvert(frame, 'bgr', 'rgb')

        # Run detection
        result = inference_detector(self.det_model, frame)

        # Get highest confidence detection
        best_box = None
        best_score = 0

        for pred in result.pred_instances:
            score = float(pred.scores.cpu().numpy()[0])
            if score > conf_threshold and score > best_score:
                best_box = pred.bboxes.cpu().numpy()[0]
                best_score = score

        if best_box is None:
            return None, None, None

        # Get crop for OCR
        crops = self.get_crops(frame, [best_box])
        if not crops:
            return None, None, None

        crop, bbox = crops[0]

        # Convert to PIL Image for OCR
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        # OCR processing
        pixel_values = self.ocr_processor(crop_pil, return_tensors="pt").pixel_values.to(self.device)
        generated_ids = self.ocr_model.generate(pixel_values)
        text = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Extract digits if needed
        cleaned_text = self.extract_digits(text)
        if len(cleaned_text) == 6:
            text = cleaned_text

        return tuple(map(int, bbox)), text, best_score