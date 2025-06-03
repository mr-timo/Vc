import numpy as np
import sounddevice as sd
import librosa
import asyncio
import platform
from scipy.signal import resample

# Configuration
SAMPLE_RATE = 44100  # Hz
BLOCK_SIZE = 1024    # Samples per block
PITCH_SHIFT = 1.5    # Pitch shift factor (1.5 = up 50%, mimics Ariana Grande's higher range)
FORMANT_SHIFT = 1.2  # Formant shift to adjust vocal tract for female-like timbre

def shift_pitch(audio, pitch_factor):
    # Pitch shifting using librosa
    return librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=np.log2(pitch_factor) * 12)

def shift_formant(audio, formant_factor):
    # Simplified formant shifting via resampling
    # Stretch the signal to simulate vocal tract changes
    stretched = resample(audio, int(len(audio) * formant_factor))
    # Resample back to original length to preserve duration
    return resample(stretched, len(audio))

def process_audio_block(indata, outdata, frames, time, status):
    if status:
        print(status)
    # Convert input to mono if stereo
    input_audio = indata[:, 0] if indata.shape[1] == 2 else indata.flatten()
    
    # Apply pitch shift
    pitched = shift_pitch(input_audio, PITCH_SHIFT)
    
    # Apply formant shift
    processed = shift_formant(pitched, FORMANT_SHIFT)
    
    # Apply a slight gain to mimic Ariana Grande's bright, clear tone
    processed = processed * 1.2
    
    # Ensure output matches input shape
    outdata[:] = processed.reshape(-1, 1) if indata.shape[1] == 1 else np.column_stack((processed, processed))

def setup():
    print("Starting real-time voice changer: Male to Ariana Grande-style female voice")
    # Check available devices
    try:
        devices = sd.query_devices()
        if not devices:
            print("No audio devices detected. Please ensure a microphone and speaker are connected.")
            exit(1)
        # Use default input and output devices
        input_device = sd.default.device[0] if sd.default.device[0] is not None else devices[0]['index']
        output_device = sd.default.device[1] if sd.default.device[1] is not None else devices[0]['index']
        # Verify device compatibility
        for dev in devices:
            if dev['index'] == input_device and dev['max_input_channels'] == 0:
                print(f"Selected input device {dev['name']} does not support input.")
                exit(1)
            if dev['index'] == output_device and dev['max_output_channels'] == 0:
                print(f"Selected output device {dev['name']} does not support output.")
                exit(1)
        # Set up audio stream
        stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            device=(input_device, output_device)
        )
        return stream
    except Exception as e:
        print(f"Error setting up audio: {e}")
        exit(1)

async def update_loop():
    stream = setup()
    stream.start()
    try:
        while True:
            await asyncio.sleep(1.0 / 60)  # Keep stream alive, 60 FPS equivalent
    except KeyboardInterrupt:
        stream.stop()
        stream.close()
        print("Voice changer stopped")

async def main():
    setup()
    await update_loop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        # Note: Run this script in a virtual environment
        # 1. Create: python3 -m venv venv
        # 2. Activate: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
        # 3. Install: pip install numpy sounddevice librosa scipy
        # 4. Run: python this_script.py
        asyncio.run(main())
