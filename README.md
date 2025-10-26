# Wideband Morse Code Decoder

A Python-based signal processing system for detecting and decoding multiple simultaneous morse code transmissions across a wide frequency band.

## Components

### Signal Tracker

The signal tracker (`signal_tracker.py`) is responsible for:

- Processing WAV audio streams containing wideband RF signals
- Detecting multiple morse code signals at different frequencies using FFT
- Extracting individual signals using polyphase filter banks
- Tracking signal energy and detecting pulse timing
- Outputting pulse data in JSON Lines format

#### Features

- **Wideband Detection**: Handles sample rates from 8 ksps up to 500 ksps
- **Multi-Signal Processing**: Detects and tracks multiple simultaneous morse transmissions
- **Noise Filtering**: Adaptive thresholding with hysteresis to reject noise
- **Boundary Handling**: Properly handles pulses that cross chunk boundaries
- **Efficient Processing**: Uses FFT and polyphase filtering for computational efficiency

#### Output Format

The signal tracker outputs JSON Lines format, where each line contains:

```json
{
  "timestamp": 0.523,
  "width": 0.062,
  "frequency": 1000.0
}
```

- `timestamp`: Start of pulse in seconds since stream start
- `width`: Pulse duration in seconds
- `frequency`: Center frequency of the signal in Hz

### Usage

#### Command Line

Basic usage:

```bash
python signal_tracker.py input.wav
```

With options:

```bash
python signal_tracker.py input.wav \
  --output pulses.json \
  --threshold 10.0 \
  --pulse-threshold 6.0 \
  --min-pulse 0.01 \
  --max-pulse 1.0
```

Options:
- `-o, --output`: Output file (default: stdout)
- `-s, --sample-rate`: Override sample rate
- `-t, --threshold`: Signal detection threshold in dB (default: 10.0)
- `-p, --pulse-threshold`: Pulse detection threshold in dB (default: 6.0)
- `--min-pulse`: Minimum pulse width in seconds (default: 0.01)
- `--max-pulse`: Maximum pulse width in seconds (default: 1.0)

#### Python API

```python
from signal_tracker import SignalTracker

# Create tracker
tracker = SignalTracker(
    sample_rate=48000,
    signal_threshold_db=10.0,
    pulse_threshold_db=6.0,
    min_pulse_width=0.01,
    max_pulse_width=1.0
)

# Process WAV file
tracker.process_wav_file('input.wav', 'output.json')

# Or process audio array directly
import numpy as np
audio = np.array([...])  # Your audio data
pulses = tracker.process_audio(audio)
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Requirements:
- Python 3.7+
- numpy >= 1.21.0
- scipy >= 1.7.0

## Testing

Run the test suite to verify functionality:

```bash
python test_signal_tracker.py
```

The test suite generates synthetic morse code signals and validates detection:

1. **Test 1**: Single signal detection (SOS pattern)
2. **Test 2**: Multiple simultaneous signals at different frequencies
3. **Test 3**: Wideband detection across 100 kHz range

Test files are generated in `/tmp/` for inspection.

## Algorithm Overview

### Signal Detection

1. **Spectral Analysis**: Audio is processed in overlapping frames using FFT
2. **Peak Detection**: Local maxima in the spectrum that exceed the noise floor threshold are identified as signals
3. **Frequency Tracking**: Detected peaks are tracked across frames to maintain consistent signal identification

### Signal Extraction

1. **Polyphase Filtering**: Each detected signal is extracted using a bandpass filter
2. **Envelope Detection**: Hilbert transform computes the signal envelope
3. **Smoothing**: Envelope is smoothed to reduce noise

### Pulse Detection

1. **Adaptive Thresholding**: Each signal maintains its own noise floor estimate
2. **Hysteresis**: Turn-on and turn-off thresholds differ by 2dB to prevent chattering
3. **Level History**: Recent signal levels are tracked for noise rejection
4. **Pulse Validation**: Detected pulses are validated against minimum/maximum width constraints

### Boundary Handling

- Overlapping chunks can be processed to ensure pulses crossing boundaries are captured
- Duplicate detection removes pulses that appear in multiple overlapping chunks
- State is maintained across chunks for continuous tracking

## Performance Considerations

- **FFT Size**: Larger FFT provides better frequency resolution but increases latency
- **Hop Size**: Smaller hop size improves time resolution but increases computation
- **Sample Rate**: Higher sample rates allow wider frequency coverage but require more processing
- **Peak Separation**: Minimum frequency separation prevents false detections from spectral spreading

## Future Enhancements

Potential improvements for the signal tracker:

1. **Dynamic Range**: Automatic gain control for varying signal strengths
2. **Frequency Drift**: Track signals that change frequency over time
3. **Adaptive Parameters**: Automatic tuning of thresholds based on signal characteristics
4. **GPU Acceleration**: Use GPU for FFT and filtering operations
5. **Real-time Processing**: Streaming mode for live audio input

## Morse Decoder (Coming Soon)

The morse decoder component will:

- Consume pulse JSON output from the signal tracker
- Group pulses by frequency
- Estimate morse timing parameters (dit, dah, spaces)
- Decode pulses into ASCII text
- Handle both machine-generated and hand-keyed morse
- Output decoded text in JSON Lines format

## License

This project is provided as-is for educational and amateur radio purposes.
