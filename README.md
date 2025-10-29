# Wideband Morse Code Decoder

A Python-based signal processing system for detecting and decoding multiple simultaneous morse code transmissions across a wide frequency band.

**WARNING** This is intentionally vibe coded (using Cursor) so that I could see what all the hype is about. It built the `signal_tracker.py` and `morse_decoder.py` first according to [the outline](outline.md). I then got it to create the streaming versions (which is what I wanted in the first place). Some minor fixes have it working for single signals in a relatively noise free environment. I'm trying to figure out whether it is worth trying to make this code work, or whether I should just start vibeing again! I guess that this is the fate of every vibecoded project. This README is really for the [batch interface](BATCH_VS_STREAMING.md). 

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

### Signal Tracker Tests

Run the signal tracker test suite:

```bash
python test_signal_tracker.py
```

The test suite generates synthetic morse code signals and validates detection:

1. **Test 1**: Single signal detection (SOS pattern)
2. **Test 2**: Multiple simultaneous signals at different frequencies
3. **Test 3**: Wideband detection across 100 kHz range

Test files are generated in `/tmp/` for inspection.

### Morse Decoder Tests

Run the complete pipeline tests:

```bash
python test_morse_decoder.py
```

This tests the full system on real WAV files:

1. **sample.wav**: Single transmission - callsign "N4LSJ CA"
2. **test1.wav**: CQ call - "CQ CQ CQ DE W1ABK"

### Manual Testing

Test the complete pipeline on any WAV file:

```bash
python signal_tracker.py your_file.wav | python morse_decoder.py /dev/stdin --debug
```

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

## Complete Pipeline

The system consists of two main components that work together:

```
WAV Audio → Signal Tracker → Pulse JSON → Morse Decoder → Decoded Text JSON
```

### Example Pipeline

```bash
# Process WAV file and decode morse
python signal_tracker.py sample.wav | python morse_decoder.py /dev/stdin

# With intermediate files for inspection
python signal_tracker.py sample.wav -o pulses.json
python morse_decoder.py pulses.json -o decoded.json --debug

# Save both outputs
python signal_tracker.py sample.wav | tee pulses.json | python morse_decoder.py /dev/stdin -o decoded.json
```

## Future Enhancements

Potential improvements:

**Signal Tracker:**
1. Dynamic range / AGC for varying signal strengths
2. Frequency drift tracking
3. GPU acceleration for FFT operations
4. Real-time streaming mode

**Morse Decoder:**
1. Error correction using context and dictionary
2. Prosign detection (AR, SK, BT, etc.)
3. Multi-transmission grouping (conversations)
4. Confidence scores for decoded text
5. Statistical language models for ambiguous decodes

## Morse Decoder

The morse decoder (`morse_decoder.py`) consumes pulse JSON output from the signal tracker and decodes it into readable text.

### Features

- **Frequency Grouping**: Groups pulses by frequency to handle multiple simultaneous transmissions
- **Adaptive Parameter Estimation**: Automatically estimates morse timing parameters (dit, dah, spaces)
- **Clustering Algorithm**: Intelligently separates intra-symbol, inter-character, and inter-word spaces
- **Variable Timing**: Handles both machine-generated and hand-keyed morse with varying timing
- **Morse Code Dictionary**: Complete International Morse Code table including letters, numbers, and punctuation

### Output Format

The morse decoder outputs JSON Lines format, where each line contains:

```json
{
  "timestamp": 0.503,
  "text": "CQ CQ CQ DE W1ABK",
  "frequency": 7031.0
}
```

- `timestamp`: Start time of first decoded symbol in seconds
- `text`: Decoded ASCII text
- `frequency`: Center frequency of the signal in Hz

### Usage

#### Command Line

Basic usage:

```bash
python morse_decoder.py pulses.json
```

Pipeline from signal tracker:

```bash
python signal_tracker.py input.wav | python morse_decoder.py /dev/stdin
```

With options:

```bash
python morse_decoder.py pulses.json \
  --output decoded.json \
  --min-pulses 5 \
  --debug
```

Options:
- `-o, --output`: Output file (default: stdout)
- `--min-pulses`: Minimum pulses required for decoding (default: 5)
- `--debug`: Enable debug output showing timing parameters

#### Python API

```python
from morse_decoder import MorseDecoder

# Create decoder
decoder = MorseDecoder(
    min_pulses_for_decode=5,
    adaptive_timing=True,
    debug=True
)

# Decode from file
decoder.decode_from_file('pulses.json', 'decoded.json')

# Or decode pulse list directly
from signal_tracker import Pulse
pulses = [...]  # List of Pulse objects
messages = decoder.decode_pulses(pulses)
```

### Algorithm Overview

#### Parameter Estimation

The decoder automatically estimates morse timing parameters:

1. **Pulse Classification**: Clusters pulse widths to separate dits from dahs
2. **Gap Analysis**: Analyzes gaps between pulses to find natural breaks
3. **Three-Cluster Detection**: Identifies intra-symbol, inter-character, and inter-word spaces
4. **Adaptive Thresholds**: Uses the two largest jumps in gap distribution

#### Decoding Process

1. **Group by Frequency**: Separate pulses into frequency channels
2. **Estimate Parameters**: Calculate timing parameters for each channel
3. **Classify Pulses**: Determine if each pulse is a dit (.) or dah (-)
4. **Split Characters**: Use gap thresholds to separate morse characters
5. **Lookup & Decode**: Convert morse patterns to ASCII using lookup table
6. **Format Output**: Generate JSON with timestamp, text, and frequency

### Examples

```bash
# Example 1: Single file processing
python signal_tracker.py sample.wav -o pulses.json
python morse_decoder.py pulses.json -o decoded.json

# Example 2: Pipeline processing
python signal_tracker.py sample.wav | python morse_decoder.py /dev/stdin

# Example 3: Debug mode to see timing analysis
python signal_tracker.py test1.wav | python morse_decoder.py /dev/stdin --debug
```

### Test Results

**sample.wav**: Contains callsign "N4LSJ CA" at 750 Hz
```json
{"timestamp": 0.503, "text": "N4LSJ CA", "frequency": 750}
```

**test1.wav**: Contains CQ call "CQ CQ CQ DE W1ABK" at 7031 Hz
```json
{"timestamp": 0.006, "text": "CQ CQ CQ DE W1ABK", "frequency": 7031}
```

## License

This project is provided as-is for educational and amateur radio purposes.
