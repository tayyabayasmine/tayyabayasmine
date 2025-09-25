#!/usr/bin/env python3
"""
Demo script for Enhanced Head Pose Exercise App
This script demonstrates how to use the application in different modes.
"""

import sys
import os
from enhanced_head_pose_app import EnhancedHeadPoseApp

def print_instructions():
    """Print usage instructions"""
    print("Enhanced Head Pose Exercise App - Demo")
    print("=" * 40)
    print()
    print("This application provides real-time head pose tracking with two modes:")
    print()
    print("1. SINGLE STREAM MODE:")
    print("   - Real-time head pose tracking using your webcam")
    print("   - Calibrate neutral position with 'c' key")
    print("   - Switch to dual stream mode with 'd' key")
    print("   - Quit with 'q' key")
    print()
    print("2. DUAL STREAM MODE:")
    print("   - Compare your live pose with a reference video")
    print("   - Start synchronized playback with 's' key")
    print("   - Calibrate user neutral position with 'c' key")
    print("   - Stop and view performance summary with Space key")
    print("   - Quit with 'q' key")
    print()
    print("FEEDBACK SYSTEM:")
    print("   - 'On Track': Your pose is within ±5° of reference")
    print("   - Corrective feedback provided for:")
    print("     * Yaw: 'Move Right' / 'Move Left'")
    print("     * Pitch: 'Look Up' / 'Look Down'")
    print("     * Roll: 'Tilt Right' / 'Tilt Left'")
    print()
    print("=" * 40)

def main():
    """Main demo function"""
    print_instructions()
    
    # Check if we can run with display
    try:
        import cv2
        # Try to create a test window to check if display is available
        cv2.namedWindow('test_window', cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow('test_window')
        display_available = True
    except:
        display_available = False
    
    if not display_available:
        print("⚠️  No display detected!")
        print("This demo requires a display to show the GUI.")
        print("To run the application:")
        print("1. Ensure you have a webcam connected")
        print("2. Run on a system with display capability")
        print("3. Execute: python enhanced_head_pose_app.py")
        print()
        print("For headless testing, run: python test_app.py")
        return
    
    # Get user choice
    print("Choose mode:")
    print("1. Single Stream Mode (default)")
    print("2. Dual Stream Mode")
    print("3. Exit")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
    except KeyboardInterrupt:
        print("\nExiting...")
        return
    
    if choice == "3":
        print("Goodbye!")
        return
    
    # Initialize the application
    app = EnhancedHeadPoseApp()
    
    try:
        if choice == "2":
            print("\nStarting Dual Stream Mode...")
            print("Make sure you have a reference video or one will be created automatically.")
            app.run_dual_stream_mode()
        else:
            print("\nStarting Single Stream Mode...")
            app.run_single_stream_mode()
    
    except KeyboardInterrupt:
        print("\nStopping application...")
        app.cleanup()
    except Exception as e:
        print(f"Error: {e}")
        app.cleanup()

if __name__ == "__main__":
    main()