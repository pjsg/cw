# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Complete Pipeline (WAV → Text)

Process a WAV file and decode morse code in one command:

```bash
python signal_tracker.py sample.wav | python morse_decoder.py /dev/stdin
```

Output:
```json
{"timestamp": 0.503, "text": "N4LSJ CA", "frequency": 750.0}
```

### Step-by-Step Processing

For inspection of intermediate results:

```bash
# Step 1: Extract pulses from audio
python signal_tracker.py sample.wav -o pulses.json

# Step 2: Decode pulses to text
python morse_decoder.py pulses.json -o decoded.json
```

### With Debug Information

See timing parameters and processing details:

```bash
python signal_tracker.py test1.wav | python morse_decoder.py /dev/stdin --debug
```

## Demo

Run the demo to see the system in action:

```bash
python demo.py
```

## Testing

Run all tests:

```bash
# Signal tracker tests (synthetic signals)
python test_signal_tracker.py

# Complete pipeline tests (real WAV files)
python test_morse_decoder.py
```

## Command Line Options

### Signal Tracker

```bash
python signal_tracker.py input.wav [options]

Options:
  -o, --output FILE       Output JSON file (default: stdout)
  -s, --sample-rate HZ    Override sample rate
  -t, --threshold DB      Signal detection threshold (default: 10.0)
  -p, --pulse-threshold DB Pulse detection threshold (default: 6.0)
  --min-pulse SECONDS     Minimum pulse width (default: 0.01)
  --max-pulse SECONDS     Maximum pulse width (default: 1.0)
```

### Morse Decoder

```bash
python morse_decoder.py input.json [options]

Options:
  -o, --output FILE       Output JSON file (default: stdout)
  --min-pulses N          Minimum pulses for decoding (default: 5)
  --debug                 Show timing analysis
```

## Examples

### Example 1: Standard Processing

```bash
python signal_tracker.py sample.wav | python morse_decoder.py /dev/stdin
```

Result: `{"timestamp": 0.503, "text": "N4LSJ CA", "frequency": 750.0}`

### Example 2: Save Intermediate Files

```bash
python signal_tracker.py sample.wav -o pulses.json
python morse_decoder.py pulses.json -o decoded.json

# Inspect the files
cat pulses.json | head -5
cat decoded.json
```

### Example 3: Adjust Sensitivity

For weak signals:

```bash
python signal_tracker.py weak_signal.wav \
  --threshold 5 \
  --pulse-threshold 3 \
  --min-pulse 0.02 | \
  python morse_decoder.py /dev/stdin
```

### Example 4: Multiple Files

Process multiple WAV files:

```bash
for file in *.wav; do
  echo "Processing $file..."
  python signal_tracker.py "$file" | python morse_decoder.py /dev/stdin
done
```

## Python API

### Signal Tracker

```python
from signal_tracker import SignalTracker

tracker = SignalTracker(sample_rate=8000)
tracker.process_wav_file('sample.wav', 'pulses.json')
```

### Morse Decoder

```python
from morse_decoder import MorseDecoder

decoder = MorseDecoder(debug=True)
decoder.decode_from_file('pulses.json', 'decoded.json')
```

### Complete Pipeline in Python

```python
from signal_tracker import SignalTracker
from morse_decoder import MorseDecoder
import wave

# Read WAV
with wave.open('sample.wav', 'rb') as wav:
    sample_rate = wav.getframerate()

# Track signals
tracker = SignalTracker(sample_rate=sample_rate)
pulses = tracker.process_audio(audio_data)

# Decode morse
decoder = MorseDecoder()
messages = decoder.decode_pulses(pulses)

# Print results
for msg in messages:
    print(f"{msg.frequency:.1f} Hz: {msg.text}")
```

## Troubleshooting

### No signals detected

- Try lowering `--threshold` (e.g., `--threshold 5`)
- Check sample rate matches your file
- Verify the file contains actual morse code signals

### Garbled output

- Adjust `--pulse-threshold` 
- Check if signal is very weak or noisy
- Try `--min-pulse` adjustment for very fast morse

### Unknown characters (?)

- Signal may be corrupted or have timing variations
- Use `--debug` to see timing analysis
- Some signals may be harmonics (ignore lower frequency versions)

## Sample Files

**sample.wav**: Amateur radio callsign "N4LSJ CA" at 750 Hz (8 kHz sample rate)

**test1.wav**: CQ call "CQ CQ CQ DE W1ABK" at 7031 Hz (48 kHz sample rate)

## File Formats

### Input: WAV Audio
- Format: PCM WAV
- Channels: Mono or stereo (uses first channel)
- Sample width: 8, 16, or 32-bit
- Sample rate: 8 kHz to 500 kHz

### Intermediate: Pulse JSON Lines
```json
{"timestamp": 0.503, "width": 0.146, "frequency": 750.0}
```

### Output: Decoded JSON Lines
```json
{"timestamp": 0.503, "text": "N4LSJ CA", "frequency": 750.0}
```

## Performance

Typical processing times (on modern CPU):

- **8 kHz, 6 seconds**: < 1 second
- **48 kHz, 10 seconds**: ~2 seconds  
- **250 kHz, 5 seconds**: ~5 seconds

Processing is CPU-bound and scales with sample rate × duration.

