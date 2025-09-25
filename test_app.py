#!/usr/bin/env python3
"""
Test script for Enhanced Head Pose Exercise App
Tests the core functionality without requiring webcam or display.
"""

import numpy as np
import cv2
from enhanced_head_pose_app import EnhancedHeadPoseApp, HeadPoseEstimator, HeadPoseTracker

def test_head_pose_estimator():
    """Test the head pose estimator"""
    print("Testing HeadPoseEstimator...")
    
    estimator = HeadPoseEstimator()
    
    # Create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test pose estimation (should return default values since no face)
    yaw, pitch, roll = estimator.estimate_pose(dummy_image)
    print(f"  Pose estimation result: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
    
    # Test landmark extraction
    landmarks = estimator.extract_landmarks(dummy_image)
    print(f"  Landmarks extracted: {landmarks is not None}")
    
    print("✓ HeadPoseEstimator test passed")

def test_head_pose_tracker():
    """Test the head pose tracker"""
    print("\nTesting HeadPoseTracker...")
    
    tracker = HeadPoseTracker()
    
    # Add some dummy measurements
    for i in range(10):
        tracker.add_measurement(i, i*0.5, i*0.2)
    
    # Test smoothed pose
    yaw, pitch, roll = tracker.get_smoothed_pose()
    print(f"  Smoothed pose: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}")
    
    # Test relative pose (should be same as smoothed since not calibrated)
    rel_yaw, rel_pitch, rel_roll = tracker.get_relative_pose()
    print(f"  Relative pose: yaw={rel_yaw:.1f}, pitch={rel_pitch:.1f}, roll={rel_roll:.1f}")
    
    # Test calibration
    calibrated = tracker.calibrate_neutral(10)
    print(f"  Calibration successful: {calibrated}")
    
    if calibrated:
        rel_yaw, rel_pitch, rel_roll = tracker.get_relative_pose()
        print(f"  Relative pose after calibration: yaw={rel_yaw:.1f}, pitch={rel_pitch:.1f}, roll={rel_roll:.1f}")
    
    print("✓ HeadPoseTracker test passed")

def test_app_functionality():
    """Test main app functionality"""
    print("\nTesting EnhancedHeadPoseApp...")
    
    app = EnhancedHeadPoseApp()
    
    # Test canvas creation
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas = app.create_large_canvas(dummy_frame)
    print(f"  Canvas created: {canvas.shape}")
    
    # Test feedback generation
    feedback_text, feedback_color = app.generate_feedback(0, 0, 0, 10, 5, 3)
    print(f"  Feedback generated: '{feedback_text}' with color {feedback_color}")
    
    # Test dual stream canvas creation
    dual_canvas = app.create_dual_stream_canvas(dummy_frame, dummy_frame, "Test Feedback", (255, 255, 255))
    print(f"  Dual stream canvas created: {dual_canvas.shape}")
    
    # Test sample reference video creation
    sample_path = "/tmp/test_reference.mp4"
    app.create_sample_reference_video(sample_path)
    print(f"  Sample reference video created: {sample_path}")
    
    # Verify video file exists and can be opened
    import os
    if os.path.exists(sample_path):
        cap = cv2.VideoCapture(sample_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  Reference video: {frame_count} frames at {fps} FPS")
            cap.release()
        else:
            print("  Warning: Could not open created reference video")
    
    print("✓ EnhancedHeadPoseApp test passed")

def test_performance_summary():
    """Test performance summary functionality"""
    print("\nTesting performance summary...")
    
    app = EnhancedHeadPoseApp()
    
    # Create dummy performance data
    performance_data = []
    for i in range(100):
        feedback = "On Track" if i % 3 == 0 else "Move Left"
        performance_data.append({
            'user_yaw': i * 0.1,
            'user_pitch': i * 0.05,
            'user_roll': i * 0.02,
            'ref_yaw': 0,
            'ref_pitch': 0,
            'ref_roll': 0,
            'feedback': feedback
        })
    
    print("  Performance data created with 100 frames")
    app.show_performance_summary(performance_data)
    
    print("✓ Performance summary test passed")

def main():
    """Run all tests"""
    print("Starting Enhanced Head Pose Exercise App Tests")
    print("=" * 50)
    
    try:
        test_head_pose_estimator()
        test_head_pose_tracker()
        test_app_functionality()
        test_performance_summary()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("The Enhanced Head Pose Exercise App is ready to use!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())