import argparse
import os
import subprocess
import sys
import threading
import rumps

# Add current directory to path to ensure local modules can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

class LocalEchoApp(rumps.App):
    def __init__(self):
        super(LocalEchoApp, self).__init__("Local Echo", icon=None, title="🎙️")
        
        # Define menu items
        self.start_btn = rumps.MenuItem("Start Meeting Capture", callback=self.start_capture)
        self.stop_btn = rumps.MenuItem("Stop & Finalize", callback=self.stop_capture)
        self.open_dir_btn = rumps.MenuItem("Open Notes Directory", callback=self.open_directory)
        
        # Setup initial menu layout
        self.menu = [self.start_btn, self.stop_btn, self.open_dir_btn]
        self.stop_btn.set_callback(None)  # Disable stop callback initially
        
        # Execution components
        self.audio_capture = None
        self.pipeline = None
        self.interceptor = None
        
        # Defaults configuration
        self.notes_dir = os.path.expanduser("~/Documents/Meetings")
        self.whisper_model = "mlx-community/whisper-small-mlx"
        self.ollama_model = "llama3.2:3b"
        self.mock = False
        self.capture_microphone = True
        
        print(f"[Local Echo] App initialized. Menu bar title is '🎙️'. Mock mode={self.mock}")

    def start_capture(self, sender):
        print(f"[Local Echo] Launching capture components (Mock Mode = {self.mock})...")
        
        # Import core modules
        try:
            from audio_capture import AudioCapture
            from transcription_pipeline import TranscriptionPipeline
            from intent_interceptor import IntentInterceptor
            
            # Spin up state
            self.audio_capture = AudioCapture(
                sample_rate=16000, 
                channel_count=1, 
                mock=self.mock, 
                capture_microphone=self.capture_microphone
            )
            self.interceptor = IntentInterceptor(model_name=self.ollama_model)
            self.pipeline = TranscriptionPipeline(
                audio_capture=self.audio_capture,
                model_name=self.whisper_model,
                notes_dir=self.notes_dir,
                intent_interceptor=self.interceptor
            )
            
            # Activate modules (order is critical: consumer -> capture -> interceptor)
            self.interceptor.start()
            self.audio_capture.start()
            self.pipeline.start()
            
            # Pulse the status bar and flip menu states
            self.title = "🔴 Recording..."
            self.start_btn.set_callback(None)  # Disable start capture action
            self.stop_btn.set_callback(self.stop_capture)  # Enable stop action
            print("[Local Echo] Recording started successfully.")
            
        except Exception as e:
            print(f"[Local Echo] Failed to start meeting capture: {e}", file=sys.stderr)
            rumps.alert("Capture Error", f"Could not launch capture pipeline:\n{e}")
            self.stop_capture(None)

    def stop_capture(self, sender):
        print("[Local Echo] Deactivating components and flushing final buffers...")

        self.stop_btn.set_callback(None)
        self.title = "⏳ Finalizing..."

        # Capture references and clear instance variables to prevent re-entry
        pipeline, audio_capture, interceptor = self.pipeline, self.audio_capture, self.interceptor
        self.pipeline = None
        self.audio_capture = None
        self.interceptor = None

        def _finalize():
            notes_path = pipeline.notes_manager.filepath if pipeline else None

            # Stop in correct order: audio first (stop new data), then pipeline (drain + summary), then interceptor
            if audio_capture:
                audio_capture.stop()
            if pipeline:
                pipeline.stop()
            if interceptor:
                interceptor.stop()

            # Show success notification if file exists
            if notes_path and os.path.exists(notes_path):
                rumps.notification("Meeting Saved", "", os.path.basename(notes_path))
            else:
                rumps.notification("Meeting Ended", "", "Notes saved to ~/Documents/Meetings")

            # Restore UI on main thread via rumps safe method
            self.title = "🎙️"
            self.start_btn.set_callback(self.start_capture)

        threading.Thread(target=_finalize, daemon=True).start()

    def open_directory(self, sender):
        os.makedirs(self.notes_dir, exist_ok=True)
        print(f"[Local Echo] Opening notes storage folder: {self.notes_dir}")
        subprocess.run(["open", self.notes_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local Echo - Local-First Live Meeting Transcriber & Automated Agent")
    parser.add_argument("--mock", action="store_true", help="Activate mock simulated audio buffer capture instead of ScreenCaptureKit")
    parser.add_argument("--whisper", type=str, default="mlx-community/whisper-small-mlx", help="MLX Whisper Hugging Face model repository")
    parser.add_argument("--ollama", type=str, default="llama3.2:3b", help="Local Ollama reasoning model name")
    parser.add_argument("--dir", type=str, default=None, help="Custom folder destination to write meeting notes markdown files")
    parser.add_argument("--no-mic", action="store_true", help="Disable native microphone capture to isolate system audio")
    
    args = parser.parse_args()
    
    app = LocalEchoApp()
    app.mock = args.mock
    app.whisper_model = args.whisper
    app.ollama_model = args.ollama
    app.capture_microphone = not args.no_mic
    if args.dir:
        app.notes_dir = os.path.abspath(os.path.expanduser(args.dir))
        
    app.run()
