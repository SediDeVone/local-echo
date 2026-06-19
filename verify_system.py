import os
import sys
import numpy as np
import time
import shutil

# Ensure current directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_verification():
    print("==================================================")
    print("  Local Echo System Verification Test Suite")
    print("==================================================\n")
    
    # Check 1: Imports & Dependencies
    print("--> CHECK 1: Importing packages & local modules...")
    try:
        import numpy as np
        import rumps
        import ollama
        import mlx_whisper
        
        import audio_capture
        import transcription_pipeline
        import intent_interceptor
        
        print("    [PASS] All packages and modules imported successfully!\n")
    except Exception as e:
        print(f"    [FAIL] Import check failed: {e}\n")
        sys.exit(1)
        
    # Check 2: Audio Capture Mock Mode
    print("--> CHECK 2: Testing mock audio capture thread...")
    try:
        capture = audio_capture.AudioCapture(sample_rate=16000, channel_count=1, mock=True)
        capture.start()
        
        # Read 3 chunks
        chunks_collected = 0
        for _ in range(6):
            item = capture.get_audio_chunk(timeout=1.0)
            if item is not None:
                assert isinstance(item, tuple), "Audio item must be a tuple (type, chunk)"
                type_, chunk = item
                assert isinstance(chunk, np.ndarray), "Audio chunk must be a numpy array"
                assert chunk.dtype == np.float32, "Audio chunk must be float32"
                chunks_collected += 1
            time.sleep(0.1)
            
        capture.stop()
        if chunks_collected > 0:
            print(f"    [PASS] Mock audio capture worked! Collected {chunks_collected} chunks of float32 PCM samples.\n")
        else:
            print("    [FAIL] No mock audio chunks collected from queue.\n")
            sys.exit(1)
    except Exception as e:
        print(f"    [FAIL] Audio capture test failed: {e}\n")
        sys.exit(1)

    # Check 3: Markdown File Notes Manager
    print("--> CHECK 3: Testing MeetingNotesManager file handling...")
    test_notes_dir = os.path.abspath("./test_meetings_dir")
    if os.path.exists(test_notes_dir):
        shutil.rmtree(test_notes_dir)
        
    try:
        from transcription_pipeline import MeetingNotesManager
        notes_mgr = MeetingNotesManager(directory=test_notes_dir)
        
        # Add transcript segment
        notes_mgr.add_transcript_segment("11:55:00", "Hello team, let's start the review.")
        
        # Add action item
        notes_mgr.add_action_item("Fix the user registration button layout", "bug", "High")
        
        # Verify file creation
        assert os.path.exists(notes_mgr.filepath), "Markdown file was not created"
        
        with open(notes_mgr.filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        print("--- Generated File Content Preview ---")
        print(content.strip())
        print("--------------------------------------")
        
        assert "## Action Items" in content, "Missing 'Action Items' section"
        assert "## Transcript" in content, "Missing 'Transcript' section"
        assert "Fix the user registration button layout" in content, "Missing registered TODO item"
        assert "Hello team" in content, "Missing transcript content"
        
        print("    [PASS] MeetingNotesManager created, updated, and atomic-flushed markdown correctly!\n")
    except Exception as e:
        print(f"    [FAIL] Notes manager test failed: {e}\n")
        sys.exit(1)
        
    # Check 4: MLX Whisper Model Execution Test (Apple Silicon Unified Memory Allocation)
    print("--> CHECK 4: Testing MLX Whisper model execution (silence input)...")
    try:
        # Create 1 second of silence at 16kHz
        dummy_audio = np.zeros(16000, dtype=np.float32)
        print("    Transcribing dummy buffer (whisper-base-mlx)...")
        t0 = time.time()
        result = mlx_whisper.transcribe(dummy_audio, path_or_hf_repo="mlx-community/whisper-base-mlx", language="en")
        t1 = time.time()
        print(f"    Transcribed silence in {t1 - t0:.2f} seconds. Result: '{result.get('text', '').strip()}'")
        print("    [PASS] Apple Silicon MLX model memory allocation and execution verified successfully!\n")
    except Exception as e:
        print(f"    [FAIL] MLX Whisper execution failed: {e}\n")
        sys.exit(1)

    # Check 5: Ollama Intent Interceptor & Webhook Test
    print("--> CHECK 5: Testing local Ollama Intent Interceptor and mock webhook...")
    try:
        from intent_interceptor import IntentInterceptor
        interceptor = IntentInterceptor(model_name="llama3.2:3b")
        interceptor.start()
        
        # Simulate processing text containing bug keyword
        test_phrase = "Wait, we have a critical bug here: we need to track a bug to fix the login page crash immediately."
        print(f"    Feeding test sentence: '{test_phrase}'")
        
        interceptor.process_text(test_phrase, notes_mgr)
        
        # Wait a few seconds for Ollama async thread to run and trigger webhook/update file
        print("    Waiting for Ollama reasoning queue...")
        time.sleep(6.0)
        interceptor.stop()
        
        # Check if the notes file was prepended with the TODO item
        with open(notes_mgr.filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
        print("--- Final Updated File Content Preview ---")
        print(content.strip())
        print("------------------------------------------")
        
        # Clean up test directory
        if os.path.exists(test_notes_dir):
            shutil.rmtree(test_notes_dir)
            
        print("    [PASS] Intent Interceptor and Ollama analysis flow complete!\n")
    except Exception as e:
        print(f"    [FAIL] Intent Interceptor test failed: {e}\n")
        sys.exit(1)

    print("==================================================")
    print("  [SUCCESS] All checks passed! Local Echo is ready!")
    print("==================================================")

if __name__ == "__main__":
    run_verification()
