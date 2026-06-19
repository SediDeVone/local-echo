import numpy as np
import os
import queue
import threading
import time
from datetime import datetime
import mlx_whisper
import sys

class MeetingNotesManager:
    def __init__(self, directory=None):
        if directory is None:
            directory = os.path.expanduser("~/Documents/Meetings")
        self.directory = directory
        os.makedirs(self.directory, exist_ok=True)

        now = datetime.now()
        filename = f"meeting_{now.strftime('%Y-%m-%d_%H%M')}.md"
        self.filepath = os.path.join(self.directory, filename)

        self.action_items = []
        self.transcript_segments = []
        self.summary = None
        self.lock = threading.Lock()
        self.started_at = datetime.now()

        print(f"[MeetingNotesManager] Notes will be saved to: {self.filepath}")
        self.flush()

    def add_transcript_segment(self, timestamp_str, text):
        with self.lock:
            self.transcript_segments.append(f"[{timestamp_str}] {text.strip()}")
            self.flush()

    def add_action_item(self, task, type_, priority):
        with self.lock:
            item = f"- [ ] TODO: [{type_.upper()}] {task} (Priority: {priority})"
            self.action_items.append(item)
            self.flush()

    def add_summary(self, summary_text):
        with self.lock:
            self.summary = summary_text
            self.flush()

    def flush(self):
        content = []
        content.append(f"# Meeting Notes: {self.started_at.strftime('%Y-%m-%d %H:%M')}\n")
        
        if self.summary:
            content.append("## Summary")
            content.append(self.summary)
            content.append("")
            
        content.append("## Action Items")
        if self.action_items:
            for item in self.action_items:
                content.append(item)
        else:
            content.append("*No action items recorded yet.*")
        
        content.append("\n---\n")
        content.append("## Transcript")
        if self.transcript_segments:
            for seg in self.transcript_segments:
                content.append(seg)
        else:
            content.append("*Transcription active...*")
        
        # Write atomically using a temporary file
        temp_path = self.filepath + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.filepath)
        except Exception as e:
            print(f"[MeetingNotesManager] Error writing markdown notes: {e}", file=sys.stderr)
            try:
                os.remove(temp_path)
            except OSError:
                pass


class TranscriptionPipeline:
    def __init__(self, audio_capture, model_name="mlx-community/whisper-small-mlx", notes_dir=None, intent_interceptor=None):
        self.audio_capture = audio_capture
        self.model_name = model_name
        self.intent_interceptor = intent_interceptor

        self.notes_manager = MeetingNotesManager(directory=notes_dir)
        self.running = False
        self.thread = None

        self.chunk_seconds = 10.0
        self.required_samples = int(16000 * self.chunk_seconds)
        self.sys_buffer = []
        self.mic_buffer = []

        self.overlap_seconds = 2.0
        self.overlap_samples = int(16000 * self.overlap_seconds)
        self.sys_last_text = ""
        self.mic_last_text = ""

    def start(self):
        if self.running:
            return
        self.running = True
        self.sys_buffer = []
        self.mic_buffer = []
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("Transcription pipeline thread spawned.")

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=60.0)
            self.thread = None
        print("Transcription pipeline thread joined.")

    def _worker_loop(self):
        # Warmup model to load weights into unified memory early and avoid UI hitching later
        print(f"[TranscriptionPipeline] Pre-loading & warming up MLX Whisper model '{self.model_name}'...")
        try:
            # 1 second of silence to initialize model buffers
            warmup_samples = np.zeros(16000, dtype=np.float32)
            mlx_whisper.transcribe(warmup_samples, path_or_hf_repo=self.model_name)
            print("[TranscriptionPipeline] MLX Whisper model loaded and ready.")
        except Exception as e:
            print(f"[TranscriptionPipeline] WARNING: Model warmup failed: {e}", file=sys.stderr)

        while self.running or not self.audio_capture.queue.empty():
            # Non-blocking check for stream buffers (0.1s timeout to drain quickly on stop)
            item = self.audio_capture.get_audio_chunk(timeout=0.1)
            if item is not None:
                type_, chunk = item
                if type_ == 1:
                    self.sys_buffer.append(chunk)
                elif type_ == 2:
                    self.mic_buffer.append(chunk)

            # VAD-based flushing for system audio buffer
            total_sys = sum(len(x) for x in self.sys_buffer)
            if total_sys > 0:
                audio_data = np.concatenate(self.sys_buffer)
                silence_idx = self._find_silence_boundary(audio_data)
                should_flush = (silence_idx is not None) or (total_sys >= 480000)

                if should_flush:
                    if silence_idx is not None:
                        speech_data = audio_data[:silence_idx]
                        remainder = audio_data[silence_idx:]
                        self.sys_buffer = [remainder] if len(remainder) > 0 else []
                    else:
                        speech_data = audio_data
                        self.sys_buffer = []

                    self._process_audio(speech_data, source="System", last_text=self.sys_last_text)

            # VAD-based flushing for microphone audio buffer
            total_mic = sum(len(x) for x in self.mic_buffer)
            if total_mic > 0:
                audio_data = np.concatenate(self.mic_buffer)
                silence_idx = self._find_silence_boundary(audio_data)
                should_flush = (silence_idx is not None) or (total_mic >= 480000)

                if should_flush:
                    if silence_idx is not None:
                        speech_data = audio_data[:silence_idx]
                        remainder = audio_data[silence_idx:]
                        self.mic_buffer = [remainder] if len(remainder) > 0 else []
                    else:
                        speech_data = audio_data
                        self.mic_buffer = []

                    self._process_audio(speech_data, source="Microphone", last_text=self.mic_last_text)

        # Process any remaining audio samples left in buffers upon stopping
        if self.sys_buffer:
            print("[TranscriptionPipeline] Processing residual system audio buffer samples...")
            audio_data = np.concatenate(self.sys_buffer)
            self.sys_buffer = []
            self._process_audio(audio_data, source="System", last_text=self.sys_last_text)
        if self.mic_buffer:
            print("[TranscriptionPipeline] Processing residual microphone audio buffer samples...")
            audio_data = np.concatenate(self.mic_buffer)
            self.mic_buffer = []
            self._process_audio(audio_data, source="Microphone", last_text=self.mic_last_text)

        # Generate automatic meeting summary at the very end of the worker thread
        self.generate_summary()

    def _process_audio(self, audio_data, source="System", last_text=""):
        try:
            audio_data = audio_data.astype(np.float32)

            # Skip near-silent chunks to prevent Whisper hallucinations on ambient noise
            rms = np.sqrt(np.mean(audio_data ** 2))
            if rms < 0.005:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[ASR - {source}] [{timestamp}] (Skipping silent chunk, RMS={rms:.4f})")
                return

            # Run local GPU-accelerated Apple Silicon ASR via mlx-whisper
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.model_name,
                language="en",
                no_speech_threshold=0.6
            )

            text = result.get("text", "").strip()

            # Discard if sentence-level hallucination detected
            if text and self._is_phrase_hallucination(text):
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[ASR - {source}] [{timestamp}] (Discarding phrase hallucination)")
                return

            # Strip any prefix overlap from the previous chunk to avoid duplicates
            if text and last_text:
                text = self._strip_overlap(text, last_text)

            # Post-process to eliminate repetitive Whisper hallucinations (like 'je je je')
            if text:
                words = text.split()
                if len(words) > 0:
                    from collections import Counter
                    counts = Counter([w.lower().strip(".,!?\"'") for w in words])
                    most_common_word, count = counts.most_common(1)[0]
                    # Use ratio-based filter: hallucinations dominate >40% of chunk
                    if (count / len(words)) > 0.4 and len(most_common_word) <= 4:
                        filtered_words = [w for w in words if w.lower().strip(".,!?\"'") != most_common_word]
                        text = " ".join(filtered_words).strip()

            timestamp = datetime.now().strftime("%H:%M:%S")
            if text:
                prefix = "[System]" if source == "System" else "[Mic]"
                print(f"[ASR - {source}] [{timestamp}] {text}")

                # Update source-specific last text for next iteration
                if source == "System":
                    self.sys_last_text = text
                else:
                    self.mic_last_text = text

                # Post to markdown notes timeline
                self.notes_manager.add_transcript_segment(timestamp, f"{prefix} {text}")

                # Check for action item intents
                if self.intent_interceptor:
                    self.intent_interceptor.process_text(text, self.notes_manager)
            else:
                print(f"[ASR - {source}] [{timestamp}] (Processed {len(audio_data)/16000:.1f}s chunk: No speech detected)")

        except Exception as e:
            print(f"[TranscriptionPipeline] ASR Error on {source} audio: {e}", file=sys.stderr)

    def _find_silence_boundary(self, audio_buffer, silence_threshold=0.0025, min_silence_frames=60):
        """
        Find the start of the longest trailing silence in the audio buffer.
        silence_threshold: RMS below this = silent frame
        min_silence_frames: number of consecutive silent frames to trigger (at 16kHz, 60 frames ≈ 1.9s)
        Returns: sample index where trailing silence begins, or None if no silence found
        """
        if len(audio_buffer) < 16000:
            return None

        frame_size = 512
        silence_count = 0
        silence_start = None

        for i in range(len(audio_buffer) - frame_size, -1, -frame_size):
            frame = audio_buffer[i:i+frame_size]
            rms = np.sqrt(np.mean(frame ** 2))

            if rms < silence_threshold:
                silence_count += 1
                if silence_start is None:
                    silence_start = i + frame_size
                if silence_count >= min_silence_frames:
                    return silence_start
            else:
                silence_count = 0
                silence_start = None

        return None

    def _is_phrase_hallucination(self, text):
        """Detect if text is dominated by repeated sentences (Whisper hallucination on silence)."""
        import re
        from collections import Counter
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 8]
        if len(sentences) < 3:
            return False
        _, most_count = Counter(s.lower() for s in sentences).most_common(1)[0]
        return (most_count / len(sentences)) > 0.5

    def _strip_overlap(self, text, last_text):
        """Strip any repeated prefix from text that overlaps with the last chunk's tail."""
        text_words = text.split()
        last_words = last_text.split()
        if not text_words or not last_words:
            return text

        # Try sliding windows from largest to smallest to find overlap
        max_check = min(len(text_words), len(last_words), 8)
        for window in range(max_check, 0, -1):
            last_tail = [w.lower().strip(".,!?\"'") for w in last_words[-window:]]
            text_head = [w.lower().strip(".,!?\"'") for w in text_words[:window]]
            if last_tail == text_head:
                stripped = " ".join(text_words[window:]).strip()
                return stripped if stripped else text

        return text

    def generate_summary(self):
        if not self.notes_manager.transcript_segments:
            return
        
        # Build the full transcript text (stripping timestamps and prefixes for clean LLM context)
        cleaned_segments = []
        for seg in self.notes_manager.transcript_segments:
            parts = seg.split("] ", 1)
            text_part = parts[-1] if parts else seg
            # Also remove speaker prefixes if present
            if text_part.startswith("[System] "):
                text_part = text_part[len("[System] "):]
            elif text_part.startswith("[Mic] "):
                text_part = text_part[len("[Mic] "):]
            cleaned_segments.append(text_part)
            
        full_transcript = "\n".join(cleaned_segments)
        
        if self.intent_interceptor and self.intent_interceptor.model_name:
            print(f"[TranscriptionPipeline] Generating automatic meeting summary via Ollama ({self.intent_interceptor.model_name})...")
            try:
                import ollama
                response = ollama.generate(
                    model=self.intent_interceptor.model_name,
                    prompt=(
                        "You are an expert executive secretary. Please write a concise, high-level summary "
                        "of the following meeting transcript as a bulleted list (max 5 bullet points). "
                        "Do not include any conversational filler, greetings, or introductory remarks. "
                        "Output only the bullet points.\n\n"
                        f"Transcript:\n{full_transcript}"
                    )
                )
                summary_text = response.get("response", "").strip()
                if summary_text:
                    self.notes_manager.add_summary(summary_text)
                    print("[TranscriptionPipeline] Automatic summary successfully appended to notes!")
            except Exception as e:
                print(f"[TranscriptionPipeline] Failed to generate meeting summary: {e}", file=sys.stderr)
