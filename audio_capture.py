import numpy as np
import threading
import queue
import time
import ctypes
import sys

# Lazy-loaded PyObjC imports to avoid issues on non-macOS or during headless tests
AudioCaptureDelegate = None
_GLOBAL_CAPTURE_RETAIN = []

def init_pyobjc():
    global AudioCaptureDelegate
    if AudioCaptureDelegate is not None:
        return
    
    try:
        import objc
        from Foundation import NSObject
        import ScreenCaptureKit
        
        class _AudioCaptureDelegate(NSObject, protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]):
            def initWithQueue_channels_sampleRate_(self, queue_obj, channels, sample_rate):
                self = objc.super(_AudioCaptureDelegate, self).init()
                if self is None:
                    return None
                self.queue = queue_obj
                self.channels = channels
                self.target_sample_rate = sample_rate
                self.has_logged_asbd = False
                return self

            @objc.typedSelector(b'v@:@^{opaqueCMSampleBuffer=}q')
            def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, type_):
                print(f"[SCK Delegate] Callback triggered! type={type_}")
                sys.stdout.flush()
                try:
                    from CoreMedia import (
                        CMSampleBufferGetDataBuffer,
                        CMBlockBufferGetDataLength,
                        CMBlockBufferCopyDataBytes,
                        CMSampleBufferGetFormatDescription,
                        CMAudioFormatDescriptionGetStreamBasicDescription
                    )
                    
                    block_buffer = CMSampleBufferGetDataBuffer(sampleBuffer)
                    if not block_buffer:
                        return
                    
                    length = CMBlockBufferGetDataLength(block_buffer)
                    if length <= 0:
                        return
                    
                    # Copy raw bytes into a mutable ctypes buffer
                    dest_buffer = (ctypes.c_char * length)()
                    status = CMBlockBufferCopyDataBytes(block_buffer, 0, length, dest_buffer)
                    status_code = status[0] if isinstance(status, tuple) else status
                    if status_code != 0:
                        return
                    
                    raw_bytes = bytes(dest_buffer)
                    
                    # Get ASBD format specifications
                    format_desc = CMSampleBufferGetFormatDescription(sampleBuffer)
                    if not format_desc:
                        return
                    
                    asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format_desc)
                    if not asbd:
                        return
                    
                    if isinstance(asbd, tuple):
                        class ASBD:
                            pass
                        asbd_obj = ASBD()
                        asbd_obj.mSampleRate = asbd[0]
                        asbd_obj.mFormatID = asbd[1]
                        asbd_obj.mFormatFlags = asbd[2]
                        asbd_obj.mBytesPerPacket = asbd[3]
                        asbd_obj.mFramesPerPacket = asbd[4]
                        asbd_obj.mBytesPerFrame = asbd[5]
                        asbd_obj.mChannelsPerFrame = asbd[6]
                        asbd_obj.mBitsPerChannel = asbd[7]
                        asbd = asbd_obj
                    
                    if not self.has_logged_asbd:
                        print(f"[SCK Audio] Stream Active. Sample Rate: {asbd.mSampleRate}Hz, "
                              f"Channels: {asbd.mChannelsPerFrame}, Bits/Channel: {asbd.mBitsPerChannel}, "
                              f"FormatFlags: {asbd.mFormatFlags}")
                        self.has_logged_asbd = True
                    
                    # Intercept PCM format (usually float32 in SCK)
                    bits = asbd.mBitsPerChannel
                    flags = asbd.mFormatFlags
                    
                    # Check float format (kAudioFormatFlagIsFloat = 1 << 0)
                    is_float = bool(flags & 1)
                    
                    if is_float:
                        if bits == 32:
                            samples = np.frombuffer(raw_bytes, dtype=np.float32)
                        elif bits == 64:
                            samples = np.frombuffer(raw_bytes, dtype=np.float64).astype(np.float32)
                        else:
                            return
                    else:
                        # Integer formats (e.g. 16-bit / 24-bit signed PCM)
                        if bits == 16:
                            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        elif bits == 24:
                            # 24-bit integer packed little-endian PCM to float32
                            n_samples = len(raw_bytes) // 3
                            if n_samples == 0:
                                return
                            raw_grid = np.frombuffer(raw_bytes[:n_samples * 3], dtype=np.uint8).reshape(-1, 3)
                            padded = np.zeros((len(raw_grid), 4), dtype=np.uint8)
                            padded[:, 1:] = raw_grid
                            samples = padded.view(dtype=np.int32).flatten().astype(np.float32) / 2147483648.0
                        elif bits == 32:
                            samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            return
                    
                    # Downmix if multi-channel
                    channels = asbd.mChannelsPerFrame
                    if channels > 1:
                        # Check planar layout (kAudioFormatFlagIsNonInterleaved = 1 << 3 = 8)
                        is_non_interleaved = bool(flags & 8)
                        if is_non_interleaved:
                            samples_per_channel = len(samples) // channels
                            mono_samples = np.zeros(samples_per_channel, dtype=np.float32)
                            for c in range(channels):
                                mono_samples += samples[c * samples_per_channel : (c + 1) * samples_per_channel]
                            samples = mono_samples / channels
                        else:
                            # Interleaved layout
                            samples = samples.reshape(-1, channels).mean(axis=1)
                    
                    # Resample if sample rate doesn't match target
                    if asbd.mSampleRate != self.target_sample_rate:
                        src_rate = asbd.mSampleRate
                        dst_rate = self.target_sample_rate
                        n_samples_in = len(samples)
                        n_samples_out = int(round(n_samples_in * dst_rate / src_rate))
                        if n_samples_out > 0:
                            x_in = np.arange(n_samples_in)
                            x_out = np.linspace(0, n_samples_in - 1, n_samples_out)
                            samples = np.interp(x_out, x_in, samples).astype(np.float32)
                    
                    # Push mono samples to the queue
                    self.queue.put((type_, samples))
                    
                    # Log volume occasionally for diagnostics
                    if len(samples) > 0:
                        if not hasattr(self, "_log_count"):
                            self._log_count = 0
                        self._log_count += 1
                        if self._log_count % 10 == 0:
                            rms = np.sqrt(np.mean(samples**2))
                            print(f"[SCK Audio] Captured {len(samples)} samples. RMS={rms:.6f}")
                    
                except Exception as e:
                    print(f"[AudioCaptureDelegate] Callback error: {e}", file=sys.stderr)

            def stream_didStopWithError_(self, stream, error):
                print(f"[AudioCaptureDelegate] Stream stopped with error: {error}", file=sys.stderr)
                sys.stderr.flush()

            def streamDidBecomeActive_(self, stream):
                print("[AudioCaptureDelegate] Stream became active!")
                sys.stdout.flush()

            def streamDidBecomeInactive_(self, stream):
                print("[AudioCaptureDelegate] Stream became inactive!")
                sys.stdout.flush()
                
        AudioCaptureDelegate = _AudioCaptureDelegate
    except Exception as e:
        print(f"Failed to bridge PyObjC types: {e}", file=sys.stderr)
        raise


class AudioCapture:
    def __init__(self, sample_rate=16000, channel_count=1, mock=False, capture_microphone=True):
        self.sample_rate = sample_rate
        self.channel_count = channel_count
        self.mock = mock
        self.capture_microphone = capture_microphone
        self.queue = queue.Queue()
        self.running = False
        
        # Native Mac attributes
        self.stream = None
        self.delegate = None
        self.capture_queue = None
        
        # Mock attributes
        self.mock_thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.queue = queue.Queue()  # Reset queue contents
        
        if self.mock:
            self._start_mock_capture()
        else:
            init_pyobjc()
            global _GLOBAL_CAPTURE_RETAIN
            _GLOBAL_CAPTURE_RETAIN.append(self)
            self._start_native_capture()

    def stop(self):
        if not self.running:
            return
        self.running = False
        
        if self.mock:
            if self.mock_thread:
                self.mock_thread.join(timeout=2.0)
                self.mock_thread = None
            print("Mock capture loop stopped.")
        else:
            global _GLOBAL_CAPTURE_RETAIN
            if self in _GLOBAL_CAPTURE_RETAIN:
                _GLOBAL_CAPTURE_RETAIN.remove(self)
            self._stop_native_capture()

    def get_audio_chunk(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _start_mock_capture(self):
        print("Starting mock audio capture thread...")
        self.mock_thread = threading.Thread(target=self._mock_loop, daemon=True)
        self.mock_thread.start()

    def _mock_loop(self):
        # Generate a synthetic 440Hz tone mixed with silence to simulate stream buffers
        duration = 3.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        sine_wave = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        
        # Introduce brief silence sections
        silence = np.zeros(int(self.sample_rate * 1.5), dtype=np.float32)
        mock_chunk = np.concatenate([sine_wave, silence, sine_wave]).astype(np.float32)
        
        chunk_duration = 0.5
        chunk_size = int(self.sample_rate * chunk_duration)
        
        log_count = 0
        while self.running:
            for i in range(0, len(mock_chunk), chunk_size):
                if not self.running:
                    break
                sub_chunk = mock_chunk[i:i+chunk_size]
                if len(sub_chunk) > 0:
                    self.queue.put((1, sub_chunk.copy()))
                    
                    # Print mock diagnostics to mimic native mode
                    log_count += 1
                    if log_count % 10 == 0:
                        rms = np.sqrt(np.mean(sub_chunk**2))
                        print(f"[SCK Audio - MOCK] Generated {len(sub_chunk)} samples. RMS={rms:.6f}")
                        
                time.sleep(chunk_duration)
            time.sleep(1.0)

    def _start_native_capture(self):
        print("Initializing ScreenCaptureKit Native capture session...")
        try:
            # Pre-flight Screen Capture access check
            from Quartz import CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess
            if not CGPreflightScreenCaptureAccess():
                print("[WARNING] ⚠️ Screen Recording permissions are NOT active for this Python session!")
                print("[WARNING] macOS will block all ScreenCaptureKit buffers. Requesting access now...")
                CGRequestScreenCaptureAccess()
                time.sleep(1.5) # Allow system settings modal a moment to load
            
            import objc
            from ScreenCaptureKit import (
                SCStreamConfiguration,
                SCContentFilter,
                SCStream,
                SCShareableContent
            )
            import dispatch
            
            # Fetch shareable screens/windows from SCK
            event = threading.Event()
            shareable_content = [None]
            error_holder = [None]
            
            def completion_handler(content, error):
                shareable_content[0] = content
                error_holder[0] = error
                event.set()
                
            SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
            
            if not event.wait(timeout=10.0):
                raise TimeoutError("Timed out requesting macOS ScreenCaptureKit shareable content permissions.")
                
            if error_holder[0]:
                raise RuntimeError(f"Error fetching shareable content: {error_holder[0]}")
                
            content = shareable_content[0]
            if not content or not content.displays():
                raise RuntimeError("No displays found. ScreenCaptureKit requires an active display display context.")
                
            display = content.displays()[0]
            
            # Setup Content Filter to exclude our own application (avoids empty filter issues on some macOS versions)
            excluding_apps = []
            import os
            current_pid = os.getpid()
            for app in content.applications():
                if app.processID() == current_pid:
                    excluding_apps.append(app)
                    break
            if not excluding_apps and content.applications():
                excluding_apps.append(content.applications()[0])
                
            content_filter = SCContentFilter.alloc().initWithDisplay_excludingApplications_exceptingWindows_(
                display, excluding_apps, []
            )
            
            # Setup Stream Settings
            config = SCStreamConfiguration.alloc().init()
            config.setCapturesAudio_(True)
            if hasattr(config, 'setExcludesCurrentProcessAudio_'):
                config.setExcludesCurrentProcessAudio_(False)
            
            if hasattr(config, 'setCaptureMicrophone_') and self.capture_microphone:
                config.setCaptureMicrophone_(True)
                print("ScreenCaptureKit microphone capture capability detected & enabled!")
            else:
                if hasattr(config, 'setCaptureMicrophone_'):
                    config.setCaptureMicrophone_(False)
                print("Microphone capture disabled or not supported.")
                
            config.setSampleRate_(self.sample_rate)
            config.setChannelCount_(self.channel_count)
            config.setQueueDepth_(8)
            
            # Initialize Delegate
            self.delegate = AudioCaptureDelegate.alloc().initWithQueue_channels_sampleRate_(self.queue, self.channel_count, self.sample_rate)
            
            # Initialize Native Stream
            self.stream = SCStream.alloc().initWithFilter_configuration_delegate_(
                content_filter, config, self.delegate
            )
            
            # Use the global concurrent queue to process callbacks on background threads
            self.capture_queue = dispatch.dispatch_get_global_queue(0, 0)
            
            # Add Stream Output for System Audio (Type 1: SCStreamOutputTypeAudio)
            error = None
            success, error = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self.delegate,
                1, # SCStreamOutputTypeAudio
                self.capture_queue,
                objc.NULL
            )
            if not success:
                raise RuntimeError(f"Could not hook SCStream system audio output. Error: {error}")
            
            # Add Stream Output for Microphone (Type 2: SCStreamOutputTypeMicrophone) if enabled
            if self.capture_microphone:
                success_mic, error_mic = self.stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self.delegate,
                    2, # SCStreamOutputTypeMicrophone
                    self.capture_queue,
                    objc.NULL
                )
                if not success_mic:
                    print(f"[WARNING] Could not hook SCStream microphone output. Error: {error_mic}")
                
            # Start Capturing
            start_event = threading.Event()
            start_error = [None]
            
            def start_handler(err):
                start_error[0] = err
                start_event.set()
                
            self.stream.startCaptureWithCompletionHandler_(start_handler)
            
            if not start_event.wait(timeout=10.0):
                raise TimeoutError("Timed out waiting for ScreenCaptureKit to begin capture stream.")
                
            if start_error[0]:
                raise RuntimeError(f"Error starting SCStream capture: {start_error[0]}")
                
            print("ScreenCaptureKit native audio capturing active!")
            
        except Exception as e:
            self.running = False
            print(f"[FATAL] Native capture startup failed: {e}", file=sys.stderr)
            raise

    def _stop_native_capture(self):
        print("Deactivating native ScreenCaptureKit capture...")
        if self.stream:
            event = threading.Event()
            def stop_handler(err):
                event.set()
            self.stream.stopCaptureWithCompletionHandler_(stop_handler)
            event.wait(timeout=5.0)
            
            self.stream = None
            self.delegate = None
            self.capture_queue = None
            print("Native capture deactivated.")
