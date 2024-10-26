import cv2
import torch
import numpy as np
import pandas as pd
from transformers import pipeline
from skimage import segmentation
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_description.log'),
        logging.StreamHandler()
    ]
)

class VideoDescriptionSystem:
    def __init__(self):
        self.config = {
            "video_source": 0,  # Default to webcam
            "frame_width": 640,
            "frame_height": 480,
            "confidence_threshold": 0.5,
            "fps": 30,
            "description_update_interval": 2,  # seconds
            "max_objects_per_description": 5
        }
        
        self.load_models()
        self.last_description_time = time.time()
        self.previous_objects = []
        
    def load_models(self) -> None:
        """Load all required models."""
        try:
            # Load YOLOv5 model
            self.object_detector = torch.hub.load(
                'ultralytics/yolov5:v6.2',
                'yolov5s',
                pretrained=True
            )
            self.object_detector.conf = self.config['confidence_threshold']
            
            # Load text generation model
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                max_length=50
            )
            
            logging.info("All models loaded successfully")
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """Process a single frame and generate description."""
        try:
            # Resize frame
            frame_resized = cv2.resize(
                frame,
                (self.config['frame_width'], self.config['frame_height'])
            )
            
            # Detect objects
            results = self.object_detector(frame_resized)
            objects = results.pandas().xyxy[0].to_dict(orient='records')
            
            # Generate description if enough time has passed
            current_time = time.time()
            if current_time - self.last_description_time >= self.config['description_update_interval']:
                description = self.generate_detailed_description(objects)
                self.last_description_time = current_time
                self.previous_objects = objects
            else:
                description = self.generate_detailed_description(self.previous_objects)
            
            # Draw bounding boxes and labels
            frame = self.draw_detections(frame, objects)
            
            return frame, description
            
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            return frame, "Error processing frame"

    def generate_detailed_description(self, objects: List[Dict[str, Any]]) -> str:
        """Generate detailed scene description."""
        try:
            if not objects:
                return "No objects detected in the scene."
            
            # Count objects by category
            object_counts = {}
            for obj in objects:
                name = obj['name']
                if name in object_counts:
                    object_counts[name] += 1
                else:
                    object_counts[name] = 1
            
            # Create basic description
            descriptions = []
            for name, count in object_counts.items():
                if count > 1:
                    descriptions.append(f"{count} {name}s")
                else:
                    descriptions.append(f"a {name}")
            
            # Add spatial relationships
            if len(objects) >= 2:
                obj1 = objects[0]
                obj2 = objects[1]
                center1 = ((obj1['xmin'] + obj1['xmax']) / 2, (obj1['ymin'] + obj1['ymax']) / 2)
                center2 = ((obj2['xmin'] + obj2['xmax']) / 2, (obj2['ymin'] + obj2['ymax']) / 2)
                
                if center1[0] < center2[0]:
                    descriptions.append(f"{obj1['name']} is to the left of {obj2['name']}")
                else:
                    descriptions.append(f"{obj1['name']} is to the right of {obj2['name']}")
            
            # Combine descriptions
            basic_desc = "I can see " + ", ".join(descriptions[:-1])
            if len(descriptions) > 1:
                basic_desc += f" and {descriptions[-1]}"
            
            return basic_desc
            
        except Exception as e:
            logging.error(f"Error generating description: {str(e)}")
            return "Error generating description"

    def draw_detections(self, frame: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        try:
            for obj in objects:
                # Draw bounding box
                x1, y1 = int(obj['xmin']), int(obj['ymin'])
                x2, y2 = int(obj['xmax']), int(obj['ymax'])
                conf = obj['confidence']
                label = f"{obj['name']} {conf:.2f}"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            return frame
            
        except Exception as e:
            logging.error(f"Error drawing detections: {str(e)}")
            return frame

    def run(self, video_source: Any = None) -> None:
        """Run the video description system."""
        if video_source is not None:
            self.config['video_source'] = video_source
            
        try:
            cap = cv2.VideoCapture(self.config['video_source'])
            if not cap.isOpened():
                raise ValueError("Could not open video source")
            
            logging.info(f"Started video capture from source: {self.config['video_source']}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                frame, description = self.process_frame(frame)
                
                # Display description
                cv2.putText(
                    frame,
                    description[:80],  # Limit text length to fit on screen
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Display timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    frame,
                    timestamp,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Show frame
                cv2.imshow('Video Description System', frame)
                
                # Break loop on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logging.error(f"Error in main loop: {str(e)}")
            
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            logging.info("Video capture ended")

if __name__ == "__main__":
    try:
        # Create and run the video description system
        video_system = VideoDescriptionSystem()
        video_system.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
