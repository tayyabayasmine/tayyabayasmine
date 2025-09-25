import cv2
import numpy as np
import mediapipe as mp
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import time
import os
from collections import deque
import math

class HeadPoseEstimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MLP model for head pose estimation
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size='auto',
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the MLP model with synthetic training data"""
        # Generate synthetic training data for head pose estimation
        # This is a simplified approach - in practice you'd use a labeled dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate synthetic face landmarks (468 landmarks with x, y, z coordinates)
        X = np.random.rand(n_samples, 468 * 3)
        
        # Generate corresponding yaw, pitch, roll angles
        yaw = np.random.uniform(-45, 45, n_samples)
        pitch = np.random.uniform(-30, 30, n_samples)
        roll = np.random.uniform(-30, 30, n_samples)
        y = np.column_stack([yaw, pitch, roll])
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X)
        self.mlp_model.fit(X_scaled, y)
        self.is_trained = True
        
    def extract_landmarks(self, image):
        """Extract face landmarks from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            landmark_array = []
            
            for landmark in landmarks.landmark:
                landmark_array.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmark_array).reshape(1, -1)
        
        return None
    
    def estimate_pose(self, image):
        """Estimate head pose angles from image"""
        landmarks = self.extract_landmarks(image)
        
        if landmarks is not None and self.is_trained:
            landmarks_scaled = self.scaler.transform(landmarks)
            pose = self.mlp_model.predict(landmarks_scaled)[0]
            return pose[0], pose[1], pose[2]  # yaw, pitch, roll
        
        return 0.0, 0.0, 0.0

class HeadPoseTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.yaw_history = deque(maxlen=window_size)
        self.pitch_history = deque(maxlen=window_size)
        self.roll_history = deque(maxlen=window_size)
        self.neutral_yaw = 0.0
        self.neutral_pitch = 0.0
        self.neutral_roll = 0.0
        self.is_calibrated = False
        
    def add_measurement(self, yaw, pitch, roll):
        """Add new pose measurement"""
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)
        self.roll_history.append(roll)
    
    def get_smoothed_pose(self):
        """Get smoothed pose angles"""
        if len(self.yaw_history) == 0:
            return 0.0, 0.0, 0.0
        
        yaw = np.mean(self.yaw_history)
        pitch = np.mean(self.pitch_history)
        roll = np.mean(self.roll_history)
        
        return yaw, pitch, roll
    
    def calibrate_neutral(self, frames=60):
        """Calibrate neutral position from recent measurements"""
        if len(self.yaw_history) >= frames:
            self.neutral_yaw = np.mean(list(self.yaw_history)[-frames:])
            self.neutral_pitch = np.mean(list(self.pitch_history)[-frames:])
            self.neutral_roll = np.mean(list(self.roll_history)[-frames:])
            self.is_calibrated = True
            return True
        return False
    
    def get_relative_pose(self):
        """Get pose relative to neutral position"""
        yaw, pitch, roll = self.get_smoothed_pose()
        
        if self.is_calibrated:
            rel_yaw = yaw - self.neutral_yaw
            rel_pitch = pitch - self.neutral_pitch
            rel_roll = roll - self.neutral_roll
            return rel_yaw, rel_pitch, rel_roll
        
        return yaw, pitch, roll

class EnhancedHeadPoseApp:
    def __init__(self):
        self.estimator = HeadPoseEstimator()
        self.tracker = HeadPoseTracker()
        self.cap = None
        self.is_running = False
        self.calibration_frames = 0
        self.max_calibration_frames = 60
        
    def create_large_canvas(self, frame, width=1200, height=800):
        """Create a large canvas for the application"""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Resize frame to fit in the left portion of canvas
        frame_height, frame_width = frame.shape[:2]
        scale = min(600 / frame_width, 600 / frame_height)
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)
        
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Place frame in canvas
        y_offset = (height - new_height) // 2
        x_offset = 50
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame
        
        return canvas
    
    def draw_enhanced_ui(self, canvas, yaw, pitch, roll, rel_yaw, rel_pitch, rel_roll):
        """Draw enhanced UI with pose information"""
        # Right panel for stats
        panel_x = 700
        panel_y = 50
        
        # Background for stats panel
        cv2.rectangle(canvas, (panel_x-10, panel_y-10), (1180, 600), (40, 40, 40), -1)
        cv2.rectangle(canvas, (panel_x-10, panel_y-10), (1180, 600), (100, 100, 100), 2)
        
        # Title
        cv2.putText(canvas, "Head Pose Stats", (panel_x, panel_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Current angles
        y_pos = panel_y + 60
        cv2.putText(canvas, "Current Angles:", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        y_pos += 40
        cv2.putText(canvas, f"Yaw:   {yaw:.1f}°", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(canvas, f"Pitch: {pitch:.1f}°", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_pos += 30
        cv2.putText(canvas, f"Roll:  {roll:.1f}°", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Relative angles (if calibrated)
        if self.tracker.is_calibrated:
            y_pos += 60
            cv2.putText(canvas, "Relative to Neutral:", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_pos += 40
            cv2.putText(canvas, f"Yaw:   {rel_yaw:.1f}°", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos += 30
            cv2.putText(canvas, f"Pitch: {rel_pitch:.1f}°", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            y_pos += 30
            cv2.putText(canvas, f"Roll:  {rel_roll:.1f}°", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Calibration status
        y_pos += 80
        if self.tracker.is_calibrated:
            cv2.putText(canvas, "Status: Calibrated", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "Status: Not Calibrated", (panel_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        y_pos += 60
        cv2.putText(canvas, "Controls:", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_pos += 30
        cv2.putText(canvas, "c - Calibrate neutral", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(canvas, "d - Dual stream mode", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
        cv2.putText(canvas, "q - Quit", (panel_x, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_single_stream_mode(self):
        """Run the single stream head pose tracking"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        self.is_running = True
        print("Single Stream Mode - Press 'c' to calibrate, 'd' for dual stream, 'q' to quit")
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Estimate head pose
            yaw, pitch, roll = self.estimator.estimate_pose(frame)
            self.tracker.add_measurement(yaw, pitch, roll)
            
            # Get smoothed and relative poses
            smooth_yaw, smooth_pitch, smooth_roll = self.tracker.get_smoothed_pose()
            rel_yaw, rel_pitch, rel_roll = self.tracker.get_relative_pose()
            
            # Create large canvas
            canvas = self.create_large_canvas(frame)
            
            # Draw enhanced UI
            self.draw_enhanced_ui(canvas, smooth_yaw, smooth_pitch, smooth_roll,
                                rel_yaw, rel_pitch, rel_roll)
            
            # Display calibration status
            if self.calibration_frames > 0:
                remaining = self.max_calibration_frames - self.calibration_frames
                cv2.putText(canvas, f"Calibrating... {remaining} frames left", 
                           (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                self.calibration_frames += 1
                
                if self.calibration_frames >= self.max_calibration_frames:
                    self.tracker.calibrate_neutral(self.max_calibration_frames)
                    self.calibration_frames = 0
                    print("Calibration complete!")
            
            cv2.imshow('Enhanced Head Pose Exercise App', canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if self.calibration_frames == 0:
                    print("Starting calibration... Keep your head in neutral position")
                    self.calibration_frames = 1
            elif key == ord('d'):
                print("Switching to dual stream mode...")
                self.cleanup()
                self.run_dual_stream_mode()
                break
        
        self.cleanup()
    
    def run_dual_stream_mode(self, reference_video_path="reference.mp4"):
        """Run dual stream mode with user webcam and reference video"""
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Try to open reference video
        ref_cap = cv2.VideoCapture(reference_video_path)
        if not ref_cap.isOpened():
            print(f"Warning: Could not open reference video '{reference_video_path}'")
            print("Creating a sample reference video for demo...")
            self.create_sample_reference_video(reference_video_path)
            ref_cap = cv2.VideoCapture(reference_video_path)
            if not ref_cap.isOpened():
                print("Error: Could not create or open reference video")
                self.cleanup()
                return
        
        # Get reference video properties
        ref_fps = ref_cap.get(cv2.CAP_PROP_FPS)
        ref_total_frames = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize reference tracker for calibration
        ref_tracker = HeadPoseTracker()
        
        # Calibrate reference video neutral position (first 60 frames)
        print("Calibrating reference video neutral position...")
        for i in range(60):
            ret, ref_frame = ref_cap.read()
            if not ret:
                break
            
            ref_yaw, ref_pitch, ref_roll = self.estimator.estimate_pose(ref_frame)
            ref_tracker.add_measurement(ref_yaw, ref_pitch, ref_roll)
        
        ref_tracker.calibrate_neutral(60)
        print("Reference video calibration complete!")
        
        # Reset reference video to beginning
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.is_running = True
        is_playing = False
        start_time = 0
        performance_data = []
        
        print("Dual Stream Mode - Press 's' to start, space to stop, 'q' to quit")
        
        while self.is_running:
            ret, user_frame = self.cap.read()
            if not ret:
                break
            
            user_frame = cv2.flip(user_frame, 1)
            
            # Get current reference frame
            ref_frame = None
            if is_playing:
                current_time = time.time() - start_time
                target_frame = int(current_time * ref_fps) % ref_total_frames
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret_ref, ref_frame = ref_cap.read()
                
                if not ret_ref:
                    # Loop back to beginning
                    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_ref, ref_frame = ref_cap.read()
            else:
                # Show first frame when not playing
                ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_ref, ref_frame = ref_cap.read()
            
            # Estimate poses
            user_yaw, user_pitch, user_roll = self.estimator.estimate_pose(user_frame)
            self.tracker.add_measurement(user_yaw, user_pitch, user_roll)
            
            feedback_text = "Position yourself and press 's' to start"
            feedback_color = (255, 255, 255)
            
            if ref_frame is not None and is_playing:
                ref_yaw, ref_pitch, ref_roll = self.estimator.estimate_pose(ref_frame)
                ref_tracker.add_measurement(ref_yaw, ref_pitch, ref_roll)
                
                # Get relative poses
                user_rel_yaw, user_rel_pitch, user_rel_roll = self.tracker.get_relative_pose()
                ref_rel_yaw, ref_rel_pitch, ref_rel_roll = ref_tracker.get_relative_pose()
                
                # Compare poses and generate feedback
                feedback_text, feedback_color = self.generate_feedback(
                    user_rel_yaw, user_rel_pitch, user_rel_roll,
                    ref_rel_yaw, ref_rel_pitch, ref_rel_roll
                )
                
                # Record performance data
                performance_data.append({
                    'user_yaw': user_rel_yaw,
                    'user_pitch': user_rel_pitch,
                    'user_roll': user_rel_roll,
                    'ref_yaw': ref_rel_yaw,
                    'ref_pitch': ref_rel_pitch,
                    'ref_roll': ref_rel_roll,
                    'feedback': feedback_text
                })
            
            # Create dual stream UI
            canvas = self.create_dual_stream_canvas(user_frame, ref_frame, feedback_text, feedback_color)
            
            # Add enhanced UI
            smooth_yaw, smooth_pitch, smooth_roll = self.tracker.get_smoothed_pose()
            rel_yaw, rel_pitch, rel_roll = self.tracker.get_relative_pose()
            self.draw_enhanced_ui(canvas, smooth_yaw, smooth_pitch, smooth_roll,
                                rel_yaw, rel_pitch, rel_roll)
            
            cv2.imshow('Enhanced Head Pose Exercise App - Dual Stream', canvas)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if not is_playing and self.tracker.is_calibrated:
                    print("Starting synchronized playback...")
                    is_playing = True
                    start_time = time.time()
                    performance_data = []
                elif not self.tracker.is_calibrated:
                    print("Please calibrate user neutral position first (press 'c')")
            elif key == ord(' '):
                if is_playing:
                    print("Stopping playback...")
                    is_playing = False
                    self.show_performance_summary(performance_data)
            elif key == ord('c'):
                if self.calibration_frames == 0:
                    print("Starting user calibration... Keep your head in neutral position")
                    self.calibration_frames = 1
            
            # Handle calibration
            if self.calibration_frames > 0:
                self.calibration_frames += 1
                if self.calibration_frames >= self.max_calibration_frames:
                    self.tracker.calibrate_neutral(self.max_calibration_frames)
                    self.calibration_frames = 0
                    print("User calibration complete!")
        
        ref_cap.release()
        self.cleanup()
    
    def create_dual_stream_canvas(self, user_frame, ref_frame, feedback_text, feedback_color):
        """Create canvas for dual stream mode"""
        canvas = np.zeros((800, 1200, 3), dtype=np.uint8)
        
        # Main user frame (large)
        if user_frame is not None:
            user_height, user_width = user_frame.shape[:2]
            scale = min(600 / user_width, 400 / user_height)
            new_width = int(user_width * scale)
            new_height = int(user_height * scale)
            
            resized_user = cv2.resize(user_frame, (new_width, new_height))
            y_offset = 50
            x_offset = 50
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_user
        
        # Reference frame thumbnail
        if ref_frame is not None:
            ref_height, ref_width = ref_frame.shape[:2]
            thumb_width, thumb_height = 200, 150
            ref_thumb = cv2.resize(ref_frame, (thumb_width, thumb_height))
            
            # Position in top right
            y_offset = 50
            x_offset = 450
            canvas[y_offset:y_offset+thumb_height, x_offset:x_offset+thumb_width] = ref_thumb
            
            # Add border around reference thumbnail
            cv2.rectangle(canvas, (x_offset-2, y_offset-2), 
                         (x_offset+thumb_width+2, y_offset+thumb_height+2), 
                         (255, 255, 255), 2)
            
            # Label
            cv2.putText(canvas, "Reference", (x_offset, y_offset-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Feedback text in center
        text_size = cv2.getTextSize(feedback_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (600 - text_size[0]) // 2 + 50
        text_y = 500
        
        # Background for feedback text
        cv2.rectangle(canvas, (text_x-10, text_y-30), 
                     (text_x+text_size[0]+10, text_y+10), (0, 0, 0), -1)
        cv2.putText(canvas, feedback_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)
        
        return canvas
    
    def generate_feedback(self, user_yaw, user_pitch, user_roll, ref_yaw, ref_pitch, ref_roll):
        """Generate feedback based on pose comparison"""
        threshold = 5.0  # ±5° threshold
        
        yaw_diff = user_yaw - ref_yaw
        pitch_diff = user_pitch - ref_pitch
        roll_diff = user_roll - ref_roll
        
        # Check if all angles are within threshold
        if (abs(yaw_diff) <= threshold and 
            abs(pitch_diff) <= threshold and 
            abs(roll_diff) <= threshold):
            return "On Track", (0, 255, 0)
        
        # Generate corrective feedback
        feedback_parts = []
        
        if abs(yaw_diff) > threshold:
            if yaw_diff > 0:
                feedback_parts.append("Move Left")
            else:
                feedback_parts.append("Move Right")
        
        if abs(pitch_diff) > threshold:
            if pitch_diff > 0:
                feedback_parts.append("Look Down")
            else:
                feedback_parts.append("Look Up")
        
        if abs(roll_diff) > threshold:
            if roll_diff > 0:
                feedback_parts.append("Tilt Left")
            else:
                feedback_parts.append("Tilt Right")
        
        feedback_text = " | ".join(feedback_parts)
        return feedback_text, (0, 0, 255)
    
    def show_performance_summary(self, performance_data):
        """Show performance summary"""
        if not performance_data:
            print("No performance data available")
            return
        
        total_frames = len(performance_data)
        on_track_frames = sum(1 for data in performance_data if data['feedback'] == "On Track")
        accuracy = (on_track_frames / total_frames) * 100
        
        print(f"\n=== Performance Summary ===")
        print(f"Total frames: {total_frames}")
        print(f"On track frames: {on_track_frames}")
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"===========================\n")
    
    def create_sample_reference_video(self, output_path):
        """Create a sample reference video for demonstration"""
        print("Creating sample reference video...")
        
        # Create a simple reference video with head movements
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10.0, (640, 480))
        
        # Create 300 frames (30 seconds at 10 fps)
        for i in range(300):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Draw a simple head representation
            center_x, center_y = 320, 240
            
            # Simulate head movement
            yaw_angle = 20 * math.sin(i * 0.02)  # Gentle left-right movement
            pitch_angle = 15 * math.sin(i * 0.015)  # Gentle up-down movement
            
            # Draw head circle
            cv2.circle(frame, (center_x, center_y), 80, (100, 100, 100), -1)
            
            # Draw eyes
            eye_y = center_y - 20 + int(pitch_angle)
            eye_x_offset = int(yaw_angle)
            cv2.circle(frame, (center_x - 25 + eye_x_offset, eye_y), 8, (255, 255, 255), -1)
            cv2.circle(frame, (center_x + 25 + eye_x_offset, eye_y), 8, (255, 255, 255), -1)
            
            # Draw nose
            nose_y = center_y + int(pitch_angle)
            nose_x = center_x + int(yaw_angle * 0.5)
            cv2.circle(frame, (nose_x, nose_y), 5, (200, 200, 200), -1)
            
            # Add instruction text
            cv2.putText(frame, "Sample Reference Video", (200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {i+1}/300", (50, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"Sample reference video created: {output_path}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_running = False

def main():
    app = EnhancedHeadPoseApp()
    print("Enhanced Head Pose Exercise App")
    print("Starting in single stream mode...")
    app.run_single_stream_mode()

if __name__ == "__main__":
    main()