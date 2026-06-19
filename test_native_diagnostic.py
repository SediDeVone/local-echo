import time
import os
import sys
import threading
import queue
import ctypes
import numpy as np
from Foundation import NSObject
import objc
import dispatch

# Ensure local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import ScreenCaptureKit & CoreMedia
import ScreenCaptureKit
from ScreenCaptureKit import SCStream, SCStreamConfiguration, SCContentFilter, SCShareableContent
from CoreMedia import (
    CMSampleBufferGetDataBuffer,
    CMBlockBufferGetDataLength,
    CMBlockBufferCopyDataBytes,
    CMSampleBufferGetFormatDescription,
    CMAudioFormatDescriptionGetStreamBasicDescription
)

# Global variables for diagnostics
callback_count = 0
block_buffer_none_count = 0
successful_chunks = 0
exceptions = []

class DiagnosticDelegate(NSObject, protocols=[objc.protocolNamed("SCStreamOutput"), objc.protocolNamed("SCStreamDelegate")]):
    def initWithQueue_(self, q):
        self = objc.super(DiagnosticDelegate, self).init()
        if self is None:
            return None
        self.q = q
        self.logged_format = False
        return self

    @objc.typedSelector(b'v@:@^{opaqueCMSampleBuffer=}q')
    def stream_didOutputSampleBuffer_ofType_(self, stream, sampleBuffer, type_):
        global callback_count, block_buffer_none_count, successful_chunks
        callback_count += 1
        
        # Log the first 5 callbacks to see their details
        if callback_count <= 5:
            print(f"\n[DIAGNOSTIC] Callback #{callback_count}: type={type_}, sampleBuffer={sampleBuffer}")
            sys.stdout.flush()
            
        try:
            if type_ != 2:
                if callback_count <= 5:
                    print(f"  -> Early return: type {type_} != 2")
                    sys.stdout.flush()
                return
                
            block_buffer = CMSampleBufferGetDataBuffer(sampleBuffer)
            if not block_buffer:
                block_buffer_none_count += 1
                if callback_count <= 5:
                    print("  -> Early return: block_buffer is None")
                    sys.stdout.flush()
                return
                
            length = CMBlockBufferGetDataLength(block_buffer)
            if length <= 0:
                if callback_count <= 5:
                    print(f"  -> Early return: length={length} <= 0")
                    sys.stdout.flush()
                return
                
            dest_buffer = (ctypes.c_char * length)()
            status = CMBlockBufferCopyDataBytes(block_buffer, 0, length, dest_buffer)
            status_code = status[0] if isinstance(status, tuple) else status
            if status_code != 0:
                if callback_count <= 5:
                    print(f"  -> Early return: CMBlockBufferCopyDataBytes status={status}")
                    sys.stdout.flush()
                return
                
            raw_bytes = bytes(dest_buffer)
            
            format_desc = CMSampleBufferGetFormatDescription(sampleBuffer)
            if not format_desc:
                if callback_count <= 5:
                    print("  -> Early return: format_desc is None")
                    sys.stdout.flush()
                return
                
            asbd = CMAudioFormatDescriptionGetStreamBasicDescription(format_desc)
            if not asbd:
                if callback_count <= 5:
                    print("  -> Early return: asbd is None")
                    sys.stdout.flush()
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
                
            if not self.logged_format:
                print(f"\n[DIAGNOSTIC] First Audio Format: SR={asbd.mSampleRate}Hz, Channels={asbd.mChannelsPerFrame}, Bits={asbd.mBitsPerChannel}")
                sys.stdout.flush()
                self.logged_format = True
                
            successful_chunks += 1
            self.q.put(raw_bytes)
            
        except Exception as e:
            exceptions.append(str(e))
            print(f"  -> Exception in callback: {e}")
            sys.stdout.flush()

    def stream_didStopWithError_(self, stream, error):
        print(f"\n[DIAGNOSTIC] Stream stopped with error: {error}")

def main():
    print("==================================================")
    print("  Local Echo - Native Capture Diagnostic Tool")
    print("==================================================")
    
    # 1. Fetch content
    event = threading.Event()
    shareable_content = [None]
    error_holder = [None]
    
    def completion_handler(content, error):
        shareable_content[0] = content
        error_holder[0] = error
        event.set()
        
    SCShareableContent.getShareableContentWithCompletionHandler_(completion_handler)
    
    if not event.wait(timeout=5.0):
        print(">>> [FAIL] Timed out fetching shareable content.")
        return
        
    if error_holder[0]:
        print(f">>> [FAIL] Error fetching shareable content: {error_holder[0]}")
        return
        
    content = shareable_content[0]
    if not content or not content.displays():
        print(">>> [FAIL] No displays found.")
        return
        
    display = content.displays()[0]
    
    # Exclude our own application to satisfy non-empty filter
    excluding_apps = []
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
    
    # 2. Configure Stream
    config = SCStreamConfiguration.alloc().init()
    config.setCapturesAudio_(True)
    config.setExcludesCurrentProcessAudio_(False)
    if hasattr(config, 'setCaptureMicrophone_'):
        config.setCaptureMicrophone_(True)
        print("[DIAGNOSTIC] Microphone capture enabled in config!")
    # Set standard system rate/channel to see if native works better
    config.setSampleRate_(48000)
    config.setChannelCount_(2)
    config.setQueueDepth_(8)
    
    q = queue.Queue()
    delegate = DiagnosticDelegate.alloc().initWithQueue_(q)
    
    stream = SCStream.alloc().initWithFilter_configuration_delegate_(
        content_filter, config, delegate
    )
    
    capture_queue = dispatch.dispatch_get_global_queue(0, 0)
    # Add Stream Output for System Audio (Type 1: SCStreamOutputTypeAudio)
    success, error = stream.addStreamOutput_type_sampleHandlerQueue_error_(
        delegate,
        1, # SCStreamOutputTypeAudio
        capture_queue,
        objc.NULL
    )
    if not success:
        print(f">>> [FAIL] Could not add system audio stream output: {error}")
        return
        
    # Add Stream Output for Microphone (Type 2: SCStreamOutputTypeMicrophone)
    success_mic, error_mic = stream.addStreamOutput_type_sampleHandlerQueue_error_(
        delegate,
        2, # SCStreamOutputTypeMicrophone
        capture_queue,
        objc.NULL
    )
    if not success_mic:
        print(f"[WARNING] Could not add microphone stream output: {error_mic}")
        
    # 3. Start Capture
    start_event = threading.Event()
    start_error = [None]
    
    def start_handler(err):
        start_error[0] = err
        start_event.set()
        
    stream.startCaptureWithCompletionHandler_(start_handler)
    
    if not start_event.wait(timeout=5.0):
        print(">>> [FAIL] Timed out starting capture stream.")
        return
        
    if start_error[0]:
        print(f">>> [FAIL] Error starting capture: {start_error[0]}")
        return
        
    print("\nCapture started successfully!")
    print("Capturing for 8 seconds. PLEASE PLAY SOME SOUND/AUDIO ON SYSTEM...")
    
    from CoreFoundation import CFRunLoopRun, CFRunLoopStop, CFRunLoopGetCurrent
    main_loop = CFRunLoopGetCurrent()
    
    def stop_timer():
        time.sleep(8.0)
        print("\nStopping run loop...")
        CFRunLoopStop(main_loop)
        
    threading.Thread(target=stop_timer, daemon=True).start()
    CFRunLoopRun()
    
    # 4. Stop capture
    stop_event = threading.Event()
    stream.stopCaptureWithCompletionHandler_(lambda err: stop_event.set())
    stop_event.wait(timeout=3.0)
    
    # 5. Print results
    print("\n================ DIAGNOSTIC RESULTS ================")
    print(f"Total stream callbacks received: {callback_count}")
    print(f"Callbacks with NULL data buffer: {block_buffer_none_count}")
    print(f"Successfully processed audio chunks: {successful_chunks}")
    print(f"Exceptions caught: {len(exceptions)}")
    for exc in exceptions[:5]:
        print(f"  - Exception: {exc}")
    print("====================================================")

if __name__ == "__main__":
    main()
