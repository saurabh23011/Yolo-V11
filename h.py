import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import threading
import queue
import logging
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core ML Dependencies Check
try:
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    import pyttsx3
    from transformers import pipeline
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    logger.warning(f"Dependency import failed: {e}")

# Define data classes for scene analysis
@dataclass
class DetectedObject:
    id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    distance_estimate: float
    direction: str
    timestamp: float

@dataclass
class AudioEvent:
    event_type: str
    confidence: float
    direction: Optional[str]
    intensity: float
    timestamp: float
    duration: float

@dataclass
class SceneState:
    objects: List[DetectedObject]
    audio_events: List[AudioEvent]
    motion_detected: bool
    timestamp: float

class AdvancedDistanceCalculator:
    """Advanced distance calculation using multiple methods"""
    
    def __init__(self):
        # Real-world object sizes in meters (height, width)
        self.object_real_sizes = {
            'person': {'height': 1.7, 'width': 0.5},
            'car': {'height': 1.5, 'width': 1.8},
            'truck': {'height': 2.5, 'width': 2.2},
            'bus': {'height': 3.0, 'width': 2.5},
            'bicycle': {'height': 1.0, 'width': 0.6},
            'motorcycle': {'height': 1.2, 'width': 0.8},
            'dog': {'height': 0.6, 'width': 0.3},
            'cat': {'height': 0.3, 'width': 0.2},
            'chair': {'height': 0.9, 'width': 0.5},
            'dining table': {'height': 0.75, 'width': 1.5},
            'bottle': {'height': 0.25, 'width': 0.08},
            'cup': {'height': 0.1, 'width': 0.08},
            'traffic light': {'height': 0.8, 'width': 0.3},
            'stop sign': {'height': 0.7, 'width': 0.7}
        }
        
        # Camera parameters
        self.focal_length_pixels = 800
        self.camera_height = 1.5
        self.camera_angle = 0
    
    def calculate_distance_multiple_methods(self, bbox, frame_height, frame_width, label):
        """Calculate distance using multiple methods"""
        x1, y1, x2, y2 = bbox
        
        # Method 1: Object size-based
        distance_size = self._distance_by_object_size(bbox, label)
        
        # Method 2: Perspective-based
        distance_perspective = self._distance_by_perspective(y2, frame_height, label)
        
        # Method 3: Geometric
        distance_geometric = self._distance_by_geometry(bbox, frame_height, frame_width, label)
        
        distances = []
        weights = []
        
        if distance_size > 0:
            distances.append(distance_size)
            weights.append(0.4)
        if distance_perspective > 0:
            distances.append(distance_perspective)
            weights.append(0.3)
        if distance_geometric > 0:
            distances.append(distance_geometric)
            weights.append(0.3)
        
        if distances:
            final_distance = sum(d * w for d, w in zip(distances, weights)) / sum(weights)
            return max(final_distance, 0.5)
        
        return 5.0
    
    def _distance_by_object_size(self, bbox, label):
        if label not in self.object_real_sizes:
            return 0
        
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        
        real_size = self.object_real_sizes[label]
        
        if 'height' in real_size and bbox_height > 0:
            distance = (real_size['height'] * self.focal_length_pixels) / bbox_height
            return distance
        
        return 0
    
    def _distance_by_perspective(self, bottom_y, frame_height, label):
        horizon_y = frame_height * 0.4
        if bottom_y <= horizon_y:
            return 0
        
        ground_pixel_distance = bottom_y - horizon_y
        max_ground_distance = frame_height - horizon_y
        relative_distance = ground_pixel_distance / max_ground_distance
        max_distance = 50.0
        distance = max_distance * (1.0 - relative_distance) ** 2
        return max(distance, 0.5)
    
    def _distance_by_geometry(self, bbox, frame_height, frame_width, label):
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        norm_x = (center_x - frame_width / 2) / (frame_width / 2)
        norm_y = (center_y - frame_height / 2) / (frame_height / 2)
        angle_y = np.arctan(norm_y)
        
        if norm_y > 0:
            distance = self.camera_height / np.tan(abs(angle_y) + 0.1)
            return min(distance, 100.0)
        
        return 0
    
    def classify_distance_category(self, distance):
        if distance < 1.0:
            return "immediate"
        elif distance < 3.0:
            return "very_close"
        elif distance < 7.0:
            return "close"
        elif distance < 15.0:
            return "medium"
        elif distance < 30.0:
            return "far"
        else:
            return "very_far"
    
    def get_distance_description(self, distance):
        category = self.classify_distance_category(distance)
        descriptions = {
            "immediate": f"immediately in front of you at {distance:.1f} meters",
            "very_close": f"very close at {distance:.1f} meters",
            "close": f"nearby at {distance:.1f} meters", 
            "medium": f"at a medium distance of {distance:.0f} meters",
            "far": f"far away at {distance:.0f} meters",
            "very_far": f"very far at {distance:.0f} meters"
        }
        return descriptions.get(category, f"at {distance:.1f} meters")

class WhisperTTSEngine:
    """Enhanced TTS using pyttsx3 with threading for non-blocking narration"""
    
    def __init__(self):
        self.fallback_tts = None
        self.speech_thread = None
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        
        try:
            self.fallback_tts = pyttsx3.init()
            self.fallback_tts.setProperty('rate', 160)
            self.fallback_tts.setProperty('volume', 0.9)
            
            voices = self.fallback_tts.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                        self.fallback_tts.setProperty('voice', voice.id)
                        break
            st.success("✅ Text-to-Speech engine initialized")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            st.warning(f"TTS initialization failed: {e}. Narration will be displayed as text.")
    
    def start_speech_thread(self):
        if self.speech_thread is None or not self.speech_thread.is_alive():
            self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
            self.speech_thread.start()
    
    def _speech_worker(self):
        while True:
            try:
                text = self.speech_queue.get(timeout=1.0)
                if text == "STOP":
                    break
                self._speak_enhanced_pyttsx3(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Speech worker error: {e}")
    
    def speak_with_whisper_enhancement(self, text):
        if not text or not self.fallback_tts:
            st.info(f"🗣️ {text}")
            return
        
        self.start_speech_thread()
        self.speech_queue.put(text)
    
    def _speak_enhanced_pyttsx3(self, text):
        try:
            enhanced_text = self._enhance_text_for_speech(text)
            enhanced_text = enhanced_text.replace('. ', '. ... ')
            enhanced_text = enhanced_text.replace(', ', ', .. ')
            
            self.is_speaking = True
            self.fallback_tts.say(enhanced_text)
            self.fallback_tts.runAndWait()
            self.is_speaking = False
            
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            st.info(f"🗣️ {text}")
    
    def _enhance_text_for_speech(self, text):
        replacements = {
            'bbox': 'bounding box',
            'yolo': 'yah-low',
            'cv': 'computer vision',
            'fps': 'frames per second',
            'ai': 'artificial intelligence',
            'ml': 'machine learning'
        }
        enhanced = text
        for old, new in replacements.items():
            enhanced = enhanced.replace(old, new)
        return enhanced
    
    def stop(self):
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_queue.put("STOP")
            self.speech_thread.join(timeout=2.0)

class StreamlitVisualProcessor:
    """Enhanced visual processor with advanced distance calculation"""
    
    def __init__(self):
        self.model = None
        self.object_tracker = {}
        self.next_object_id = 0
        self.distance_calculator = AdvancedDistanceCalculator()
        
        if DEPENDENCIES_AVAILABLE:
            try:
                self.model = YOLO('yolo11n.pt')
                logger.info("YOLO model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
    
    def process_frame(self, frame: np.ndarray) -> List[DetectedObject]:
        objects = []
        
        if self.model is None or not DEPENDENCIES_AVAILABLE:
            return self._generate_mock_objects_with_distances(frame.shape)
        
        try:
            results = self.model(frame, verbose=False)
            frame_height, frame_width = frame.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        label = self.model.names[class_id]
                        
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        distance = self.distance_calculator.calculate_distance_multiple_methods(
                            (x1, y1, x2, y2), frame_height, frame_width, label
                        )
                        
                        direction = self._calculate_direction(center_x, frame_width)
                        
                        obj = DetectedObject(
                            id=self._get_or_assign_id(center_x, center_y, label),
                            label=label,
                            confidence=float(confidence),
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            center=(center_x, center_y),
                            distance_estimate=distance,
                            direction=direction,
                            timestamp=time.time()
                        )
                        
                        objects.append(obj)
                        
        except Exception as e:
            logger.error(f"Visual processing error: {e}")
            return self._generate_mock_objects_with_distances(frame.shape)
        
        return objects
    
    def _generate_mock_objects_with_distances(self, frame_shape) -> List[DetectedObject]:
        frame_height, frame_width = frame_shape[:2]
        mock_objects = [
            DetectedObject(
                id=1,
                label="person",
                confidence=0.85,
                bbox=(100, 200, 200, 400),
                center=(150, 300),
                distance_estimate=2.3,
                direction="left",
                timestamp=time.time()
            ),
            DetectedObject(
                id=2,
                label="car",
                confidence=0.92,
                bbox=(300, 250, 500, 350),
                center=(400, 300),
                distance_estimate=12.5,
                direction="center",
                timestamp=time.time()
            ),
            DetectedObject(
                id=3,
                label="bicycle",
                confidence=0.78,
                bbox=(520, 280, 580, 360),
                center=(550, 320),
                distance_estimate=6.8,
                direction="right",
                timestamp=time.time()
            )
        ]
        return mock_objects
    
    def _calculate_direction(self, center_x: float, frame_width: int) -> str:
        third = frame_width / 3
        if center_x < third:
            return "left"
        elif center_x < 2 * third:
            return "center"
        else:
            return "right"
    
    def _get_or_assign_id(self, x: float, y: float, label: str) -> int:
        self.next_object_id += 1
        return self.next_object_id

class StreamlitNarrator:
    """Enhanced narrator with LLM-generated narration"""
    
    def __init__(self):
        self.last_narration_time = 0
        self.narration_cooldown = 1.0
        self.tts_engine = WhisperTTSEngine()
        self.distance_calculator = AdvancedDistanceCalculator()
        self.proximity_threshold = 3.0
        self.llm = None
        
        if DEPENDENCIES_AVAILABLE:
            try:
                self.llm = pipeline("text-generation", model="distilgpt2", device=0 if torch.cuda.is_available() else -1)
                logger.info("LLM (distilgpt2) loaded successfully")
                st.success("✅ LLM initialized")
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
                st.warning(f"LLM initialization failed: {e}. Using rule-based narration.")
    
    def generate_narration(self, scene_state: SceneState) -> str:
        current_time = time.time()
        
        if current_time - self.last_narration_time < self.narration_cooldown:
            return ""
        
        # Prepare scene data
        person_count = len([obj for obj in scene_state.objects if obj.label == "person"])
        close_objects = [obj for obj in scene_state.objects if obj.distance_estimate <= 7.0 and obj.label != "person"]
        proximity_info = []
        for person in [obj for obj in scene_state.objects if obj.label == "person"]:
            person_center = person.center
            nearby_objects = [
                obj for obj in scene_state.objects
                if obj.label != "person" and obj.distance_estimate <= self.proximity_threshold
                and np.sqrt((person_center[0] - obj.center[0])**2 + (person_center[1] - obj.center[1])**2) < 100
            ]
            if nearby_objects:
                proximity_info.append(f"Near a person: {', '.join([obj.label for obj in nearby_objects])}")
        
        # LLM-based narration
        if self.llm:
            try:
                prompt = f"""
                You are narrating a scene in real-time. The scene contains:
                - {person_count} person(s).
                - {len(close_objects)} nearby object(s): {', '.join([f'a {obj.label} {self.distance_calculator.get_distance_description(obj.distance_estimate)} on the {obj.direction}' for obj in close_objects[:2]])}.
                - Proximity: {'. '.join(proximity_info) if proximity_info else 'No objects near persons.'}
                - Motion detected: {'yes' if scene_state.motion_detected else 'no'}.
                Generate a concise, natural narration (1-2 sentences) describing the scene.
                """
                narration = self.llm(
                    prompt,
                    max_length=100,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    truncation=True
                )[0]['generated_text'].strip()
                # Extract the generated part after the prompt
                narration = narration.split('\n')[-1].strip()
                if not narration:
                    narration = self._fallback_narration(scene_state)
            except Exception as e:
                logger.error(f"LLM narration failed: {e}")
                narration = self._fallback_narration(scene_state)
        else:
            narration = self._fallback_narration(scene_state)
        
        self.last_narration_time = current_time
        return narration
    
    def _fallback_narration(self, scene_state: SceneState) -> str:
        narration_parts = []
        persons = [obj for obj in scene_state.objects if obj.label == "person"]
        person_count = len(persons)
        narration_parts.append(f"There are {person_count} person{'s' if person_count != 1 else ''} detected.")
        
        close_objects = [obj for obj in scene_state.objects if obj.distance_estimate <= 7.0 and obj.label != "person"]
        if close_objects:
            narration_parts.append(f"{len(close_objects)} object{'s' if len(close_objects) != 1 else ''} are nearby.")
            for obj in close_objects[:2]:
                distance_desc = self.distance_calculator.get_distance_description(obj.distance_estimate)
                narration_parts.append(f"A {obj.label} is {distance_desc} on your {obj.direction}.")
        
        for person in persons:
            person_bbox = person.bbox
            person_center = person.center
            nearby_objects = []
            for obj in scene_state.objects:
                if obj.label == "person" or obj.distance_estimate > self.proximity_threshold:
                    continue
                obj_center = obj.center
                distance = np.sqrt((person_center[0] - obj_center[0])**2 + (person_center[1] - obj_center[1])**2)
                if distance < 100:
                    nearby_objects.append(obj)
            if nearby_objects:
                obj_types = [obj.label for obj in nearby_objects]
                narration_parts.append(f"Near a person, there {'are' if len(obj_types) > 1 else 'is'} {', '.join(obj_types)}.")
        
        if scene_state.motion_detected:
            narration_parts.append("Movement detected in the area.")
        
        return ". ".join(narration_parts) + "." if narration_parts else ""
    
    def speak_narration(self, text: str):
        if not text:
            return
        logger.info(f"🔊 Narrating: {text}")
        self.tts_engine.speak_with_whisper_enhancement(text)

async def process_live_feed(visual_processor, narrator, frame_placeholder, narration_placeholder, stats_placeholder):
    """Process live webcam feed with real-time video and narration"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam")
        return
    
    prev_gray = None
    frame_count = 0
    narrations = []
    start_time = time.time()
    
    try:
        while st.session_state.get('live_mode', False):
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to capture frame")
                continue
            
            frame_count += 1
            
            if frame_count % 5 == 0:
                objects = visual_processor.process_frame(frame)
                motion_detected = False
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        motion_detected = np.mean(diff) > 15
                    prev_gray = gray
                except Exception as e:
                    logger.warning(f"Motion detection error: {e}")
                
                scene_state = SceneState(
                    objects=objects,
                    audio_events=[],
                    motion_detected=motion_detected,
                    timestamp=time.time()
                )
                
                narration = narrator.generate_narration(scene_state)
                annotated_frame = frame
                frame_placeholder.image(annotated_frame, caption=f"Live Frame {frame_count}", use_column_width=True)
                
                if narration:
                    narrations.append({'timestamp': time.time() - start_time, 'text': narration})
                    narration_placeholder.write(f"🗣️ {narration}")
                    narrator.speak_narration(narration)
                
                person_count = len([obj for obj in objects if obj.label == "person"])
                close_objects = len([obj for obj in objects if obj.distance_estimate <= 7.0 and obj.label != "person"])
                stats_text = f"Frames processed: {frame_count}\n"
                stats_text += f"Persons detected: {person_count}\n"
                stats_text += f"Nearby objects: {close_objects}\n"
                stats_text += f"Narrations: {len(narrations)}"
                stats_placeholder.text(stats_text)
            
            await asyncio.sleep(0.1)
        
    finally:
        cap.release()
        narrator.tts_engine.stop()

def main():
    """Main Streamlit app"""
    
    st.set_page_config(
        page_title="Environmental Scene Narrator",
        page_icon="🎥",
        layout="wide"
    )
    
    st.title("🎥 Environmental Audio-Visual Scene Narrator")
    st.markdown("---")
    
    st.sidebar.header("⚙️ Settings")
    
    if not DEPENDENCIES_AVAILABLE:
        st.sidebar.warning("⚠️ Some dependencies missing. Install: `pip install torch torchvision ultralytics pyttsx3 transformers`")
        st.sidebar.info("Running in demo mode with mock data")
    else:
        st.sidebar.success("✅ All dependencies available")
    
    if 'live_mode' not in st.session_state:
        st.session_state.live_mode = False
    
    st.header("📹 Live Webcam Mode")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start Live Mode", type="primary"):
            st.session_state.live_mode = True
    with col2:
        if st.button("🛑 Stop Live Mode"):
            st.session_state.live_mode = False
    
    frame_placeholder = st.empty()
    narration_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    visual_processor = StreamlitVisualProcessor()
    narrator = StreamlitNarrator()
    
    if st.session_state.live_mode:
        asyncio.run(process_live_feed(
            visual_processor, 
            narrator, 
            frame_placeholder, 
            narration_placeholder, 
            stats_placeholder
        ))
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌟 Environmental Scene Narrator | Built with Streamlit & Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()