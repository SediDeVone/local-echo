import time
import os
import sys
import threading
from audio_capture import AudioCapture

# Ensure local imports are resolved correctly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    print("==================================================")
    print("  Local Echo - Command-Line Native Capture Test")
    print("==================================================\n")
    print("This script will run native ScreenCaptureKit capture for 10 seconds.")
    print("Please make sure you have granted Screen Recording and Microphone")
    print("permissions to the Terminal/IDE running this script.\n")
    
    import argparse
    parser = argparse.ArgumentParser(description="Native Capture Test")
    parser.add_argument("--no-mic", action="store_true", help="Disable microphone capture to isolate system audio permissions")
    args = parser.parse_args()
    
    use_mic = not args.no_mic
    print(f"Microphone capture enabled: {use_mic}")
    
    capture = AudioCapture(sample_rate=16000, channel_count=1, mock=False, capture_microphone=use_mic)
    
    try:
        print("Starting native capture...")
        capture.start()
        print("Capture started. Listening for 10 seconds (speak or play sound)...")
        
        from CoreFoundation import CFRunLoopRun, CFRunLoopStop, CFRunLoopGetCurrent
        
        # Get the main run loop
        main_loop = CFRunLoopGetCurrent()
        
        # Start a stop timer thread
        def stop_timer():
            time.sleep(10.0)
            print("\n10 seconds complete. Stopping run loop...")
            CFRunLoopStop(main_loop)
            
        timer_thread = threading.Thread(target=stop_timer, daemon=True)
        timer_thread.start()
        
        # Run the CFRunLoop (this blocks the main thread while processing system events and callbacks)
        CFRunLoopRun()
        
        # Now drain chunks from the queue
        chunks_read = 0
        while True:
            chunk = capture.get_audio_chunk(timeout=0.01)
            if chunk is not None:
                chunks_read += 1
            else:
                break
                
        print(f"\nTest finished! Drained {chunks_read} chunks from capture queue.")
        if chunks_read > 0:
            print(">>> [PASS] Native capture is successfully delivering audio packets!")
        else:
            print(">>> [FAIL] Native capture did not deliver any audio packets. This is a macOS Permission lock!")
            print("    Please check System Settings > Privacy & Security > Screen & System Audio Recording.")
            
    except Exception as e:
        print(f"\n>>> [FAIL] Capture failed with error: {e}")
    finally:
        print("\nStopping capture...")
        capture.stop()
        print("Stopped.")

if __name__ == "__main__":
    main()
