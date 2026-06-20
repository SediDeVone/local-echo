# Local Echo

Local Echo is a local-first, live meeting transcriber and automated agent for macOS. It runs entirely on your machine, ensuring privacy and low latency by leveraging Apple Silicon's hardware acceleration for transcription and local LLMs for reasoning.

## Features

- **macOS Menu Bar App:** Discreet and easy-to-access interface.
- **Real-time Transcription:** Powered by `mlx-whisper` for high-performance transcription on Apple Silicon.
- **Dual Source Capture:** Captures both system audio (meeting participants) and your own microphone.
- **Automated Note-taking:** Automatically generates meeting notes in Markdown format, including summaries and action items.
- **Local Reasoning:** Uses Ollama (Llama 3.2) to intercept intents and extract tasks during the meeting.
- **Privacy First:** All audio processing and LLM inference happen locally on your device.

## Prerequisites

- **macOS:** Required for `ScreenCaptureKit` and `rumps`.
- **Apple Silicon (M1/M2/M3/M4):** Recommended for optimal performance with `mlx-whisper`.
- **Python 3.10+**
- **Ollama:** Must be installed and running.
  - Pull the required model: `ollama pull llama3.2:3b`

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd audio_notes_capture
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application using Python:

```bash
python app.py
```

### Command Line Arguments

- `--mock`: Activate simulated audio buffer capture instead of real system capture (useful for testing).
- `--whisper <model_name>`: Specify the MLX Whisper model (default: `mlx-community/whisper-small-mlx`).
- `--ollama <model_name>`: Specify the local Ollama model (default: `llama3.2:3b`).
- `--dir <path>`: Custom folder destination for meeting notes.
- `--no-mic`: Disable native microphone capture to isolate system audio.

### How it works

1. Click the **🎙️** icon in the menu bar.
2. Select **Start Meeting Capture**. The icon will change to **🔴 Recording...**.
3. During the meeting, Local Echo transcribes audio in real-time.
4. Select **Stop & Finalize** to finish. The app will generate a summary and action items.
5. Notes are saved to `~/Documents/Meetings` by default as `.md` files, along with the full `.wav` recording.

## Project Structure

- `app.py`: Main entry point and macOS menu bar UI logic.
- `audio_capture.py`: Handles system and microphone audio capture using ScreenCaptureKit.
- `transcription_pipeline.py`: Orchestrates audio processing, transcription, and note management.
- `intent_interceptor.py`: Interfaces with Ollama to detect intents and action items.
- `verify_system.py`: A utility script to verify system compatibility and dependencies.

## License

[Add License Information Here]
