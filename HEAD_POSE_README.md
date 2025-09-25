# Enhanced Head Pose Exercise App

An advanced head pose tracking application that uses MediaPipe and MLP (Multi-Layer Perceptron) for real-time head pose estimation. The app supports both single-stream and dual-stream modes for exercise tracking and comparison.

## Features

- **Real-time Head Pose Estimation**: Uses MediaPipe for facial landmark detection and MLP for pose angle prediction
- **Single Stream Mode**: Track your head pose in real-time with calibration support
- **Dual Stream Mode**: Compare your live pose with a reference video
- **Calibration System**: Calibrate neutral positions for both user and reference video
- **Real-time Feedback**: Get instant corrective feedback (±5° threshold)
- **Performance Analytics**: View accuracy statistics and performance summaries

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python enhanced_head_pose_app.py
```

## Usage

### Single Stream Mode
- **'c'**: Calibrate neutral head position (keep head straight for 6 seconds)
- **'d'**: Switch to dual stream mode
- **'q'**: Quit application

### Dual Stream Mode
- **'s'**: Start synchronized playback with reference video
- **'c'**: Calibrate user neutral position
- **Space**: Stop playback and show performance summary
- **'q'**: Quit application

## Dual Stream Mode Details

### Video Inputs
- **User Stream**: Live webcam feed (main window, large display)
- **Reference Stream**: Local reference video (thumbnail overlay)

### Feedback System
- **"On Track"**: User pose is within ±5° of reference pose
- **Corrective Feedback**:
  - Yaw: "Move Right" / "Move Left"
  - Pitch: "Look Up" / "Look Down"
  - Roll: "Tilt Right" / "Tilt Left"

### Calibration
- User neutral position: Calibrated separately via 'c' key
- Reference video neutral: Automatically calibrated from first 60 frames

## File Structure

- `enhanced_head_pose_app.py`: Main application file
- `requirements.txt`: Python dependencies
- `reference.mp4`: Reference exercise video (auto-created if missing)

## Technical Details

- **MediaPipe**: Face mesh detection with 468 landmarks
- **MLP Model**: 3-layer neural network (64-32-16 neurons) for pose estimation
- **Pose Angles**: Yaw (left/right), Pitch (up/down), Roll (tilt)
- **Smoothing**: 30-frame moving average for stable tracking
- **Video Support**: Automatic looping of reference video

## Notes

- A sample reference video will be automatically created if `reference.mp4` is not found
- The MLP model is pre-trained with synthetic data for demonstration purposes
- For production use, train the MLP with a proper labeled head pose dataset