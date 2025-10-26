#!/usr/bin/env python3
"""
Test script for Signal Tracker

Generates synthetic morse code signals and tests the signal tracker's
ability to detect and track them.
"""

import numpy as np
import wave
import json
import os
import sys
from signal_tracker import SignalTracker


def generate_morse_tone(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 0.3
) -> np.ndarray:
    """
    Generate a morse code tone pulse.

    Args:
        frequency: Carrier frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        amplitude: Amplitude (0-1)

    Returns:
        Audio samples
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate

    # Generate carrier with envelope
    carrier = np.sin(2 * np.pi * frequency * t)

    # Apply rise/fall envelope to reduce key clicks
    rise_fall_samples = int(0.002 * sample_rate)  # 2ms rise/fall
    envelope = np.ones_like(t)

    if len(envelope) > 2 * rise_fall_samples:
        # Rise
        envelope[:rise_fall_samples] = np.linspace(0, 1, rise_fall_samples)
        # Fall
        envelope[-rise_fall_samples:] = np.linspace(1, 0, rise_fall_samples)

    return amplitude * carrier * envelope


def generate_morse_sequence(
    frequency: float,
    morse_code: str,
    dit_duration: float,
    sample_rate: int,
    start_time: float = 0.0
) -> tuple[np.ndarray, list]:
    """
    Generate a sequence of morse code.

    Args:
        frequency: Carrier frequency in Hz
        morse_code: Morse code string (dots and dashes: '.-')
        dit_duration: Duration of a dit in seconds
        sample_rate: Sample rate in Hz
        start_time: Start time offset in seconds

    Returns:
        Tuple of (audio samples, list of pulse info)
    """
    # Morse timing ratios
    dah_duration = 3 * dit_duration
    intra_symbol_space = dit_duration
    inter_symbol_space = 3 * dit_duration

    audio_segments = []
    pulse_info = []
    current_time = start_time

    for i, symbol in enumerate(morse_code):
        if symbol == '.':
            # Dit
            tone = generate_morse_tone(frequency, dit_duration, sample_rate)
            audio_segments.append(tone)
            pulse_info.append({
                'start': current_time,
                'duration': dit_duration,
                'frequency': frequency
            })
            current_time += dit_duration
        elif symbol == '-':
            # Dah
            tone = generate_morse_tone(frequency, dah_duration, sample_rate)
            audio_segments.append(tone)
            pulse_info.append({
                'start': current_time,
                'duration': dah_duration,
                'frequency': frequency
            })
            current_time += dah_duration
        elif symbol == ' ':
            # Inter-symbol space (already included in normal spacing)
            silence = np.zeros(int(inter_symbol_space * sample_rate))
            audio_segments.append(silence)
            current_time += inter_symbol_space
            continue

        # Add intra-symbol spacing
        if i < len(morse_code) - 1:
            silence = np.zeros(int(intra_symbol_space * sample_rate))
            audio_segments.append(silence)
            current_time += intra_symbol_space

    if audio_segments:
        audio = np.concatenate(audio_segments)
    else:
        audio = np.array([])

    return audio, pulse_info


def save_wav(filename: str, audio: np.ndarray, sample_rate: int):
    """Save audio to WAV file."""
    # Convert to int16
    audio_int = (audio * 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int.tobytes())


def test_single_signal():
    """Test detection of a single morse signal."""
    print("=" * 60)
    print("Test 1: Single Signal Detection")
    print("=" * 60)

    sample_rate = 48000
    dit_duration = 0.06  # 60ms dit = ~20 WPM

    # Generate "SOS" pattern: ... --- ...
    sos_pattern = "... --- ..."

    frequency = 1000.0  # 1 kHz tone

    audio, expected_pulses = generate_morse_sequence(
        frequency, sos_pattern, dit_duration, sample_rate
    )

    # Add some silence at beginning and end
    silence_duration = 0.5  # 500ms
    silence = np.zeros(int(silence_duration * sample_rate))
    audio = np.concatenate([silence, audio, silence])

    # Adjust expected pulse times
    for pulse in expected_pulses:
        pulse['start'] += silence_duration

    # Save test file
    test_file = '/tmp/test_single_signal.wav'
    save_wav(test_file, audio, sample_rate)
    print(f"Generated test file: {test_file}")
    print(f"Expected {len(expected_pulses)} pulses")

    # Run tracker
    tracker = SignalTracker(
        sample_rate=sample_rate,
        signal_threshold_db=8.0,
        pulse_threshold_db=5.0,
        min_pulse_width=0.03,
        max_pulse_width=0.5
    )

    output_file = '/tmp/test_single_signal_output.json'
    tracker.process_wav_file(test_file, output_file)

    # Read and display results
    with open(output_file, 'r') as f:
        detected_pulses = [json.loads(line) for line in f]

    print(f"Detected {len(detected_pulses)} pulses")
    print("\nExpected pulses:")
    for i, pulse in enumerate(expected_pulses, 1):
        print(f"  {i}. Time: {pulse['start']:.3f}s, "
              f"Width: {pulse['duration']:.3f}s, "
              f"Freq: {pulse['frequency']:.1f}Hz")

    print("\nDetected pulses:")
    for i, pulse in enumerate(detected_pulses, 1):
        print(f"  {i}. Time: {pulse['timestamp']:.3f}s, "
              f"Width: {pulse['width']:.3f}s, "
              f"Freq: {pulse['frequency']:.1f}Hz")

    print()


def test_multiple_signals():
    """Test detection of multiple simultaneous morse signals."""
    print("=" * 60)
    print("Test 2: Multiple Signal Detection")
    print("=" * 60)

    sample_rate = 48000
    dit_duration = 0.08  # 80ms dit

    # Generate two different patterns at different frequencies
    # Signal 1: "CQ" at 800 Hz
    cq_pattern = "-.-. --.-"  # C Q
    freq1 = 800.0

    # Signal 2: "DE" at 1500 Hz
    de_pattern = "-.. ."  # D E
    freq2 = 1500.0

    audio1, pulses1 = generate_morse_sequence(freq1, cq_pattern, dit_duration, sample_rate, 0.0)
    audio2, pulses2 = generate_morse_sequence(freq2, de_pattern, dit_duration, sample_rate, 0.3)

    # Make both signals the same length
    max_len = max(len(audio1), len(audio2))
    if len(audio1) < max_len:
        audio1 = np.concatenate([audio1, np.zeros(max_len - len(audio1))])
    if len(audio2) < max_len:
        audio2 = np.concatenate([audio2, np.zeros(max_len - len(audio2))])

    # Mix signals
    audio = audio1 + audio2

    # Add silence
    silence_duration = 0.5
    silence = np.zeros(int(silence_duration * sample_rate))
    audio = np.concatenate([silence, audio, silence])

    # Adjust expected pulse times
    for pulse in pulses1:
        pulse['start'] += silence_duration
    for pulse in pulses2:
        pulse['start'] += silence_duration + 0.3

    expected_pulses = pulses1 + pulses2

    # Add some noise
    noise_level = 0.02
    noise = noise_level * np.random.randn(len(audio))
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save test file
    test_file = '/tmp/test_multiple_signals.wav'
    save_wav(test_file, audio, sample_rate)
    print(f"Generated test file: {test_file}")
    print(f"Expected {len(expected_pulses)} pulses total")
    print(f"  - {len(pulses1)} pulses at {freq1:.1f} Hz")
    print(f"  - {len(pulses2)} pulses at {freq2:.1f} Hz")

    # Run tracker
    tracker = SignalTracker(
        sample_rate=sample_rate,
        signal_threshold_db=10.0,
        pulse_threshold_db=6.0,
        min_pulse_width=0.03,
        max_pulse_width=0.5,
        peak_separation_hz=100.0
    )

    output_file = '/tmp/test_multiple_signals_output.json'
    tracker.process_wav_file(test_file, output_file)

    # Read and display results
    with open(output_file, 'r') as f:
        detected_pulses = [json.loads(line) for line in f]

    print(f"Detected {len(detected_pulses)} pulses")

    # Group by frequency
    freq_groups = {}
    for pulse in detected_pulses:
        freq = round(pulse['frequency'] / 100) * 100  # Round to nearest 100Hz
        if freq not in freq_groups:
            freq_groups[freq] = []
        freq_groups[freq].append(pulse)

    print("\nDetected pulses by frequency:")
    for freq in sorted(freq_groups.keys()):
        print(f"\n  {freq:.0f} Hz ({len(freq_groups[freq])} pulses):")
        for pulse in freq_groups[freq][:5]:  # Show first 5
            print(f"    Time: {pulse['timestamp']:.3f}s, Width: {pulse['width']:.3f}s")
        if len(freq_groups[freq]) > 5:
            print(f"    ... and {len(freq_groups[freq]) - 5} more")

    print()


def test_wideband():
    """Test with signals across a wide frequency range."""
    print("=" * 60)
    print("Test 3: Wideband Signal Detection")
    print("=" * 60)

    sample_rate = 250000  # 250 ksps
    dit_duration = 0.06

    # Generate signals at different frequencies across 100 kHz band
    frequencies = [10000, 25000, 50000, 75000, 95000]  # Hz

    signals = []
    all_pulses = []

    for i, freq in enumerate(frequencies):
        pattern = "... "  # Simple pattern
        start_time = i * 0.1  # Stagger start times

        audio, pulses = generate_morse_sequence(freq, pattern, dit_duration, sample_rate, start_time)

        for pulse in pulses:
            pulse['start'] += 0.5  # Account for initial silence

        signals.append(audio)
        all_pulses.extend(pulses)

    # Make all signals the same length
    max_len = max(len(sig) for sig in signals)
    for i in range(len(signals)):
        if len(signals[i]) < max_len:
            signals[i] = np.concatenate([signals[i], np.zeros(max_len - len(signals[i]))])

    # Mix all signals
    audio = np.sum(signals, axis=0)

    # Add silence
    silence = np.zeros(int(0.5 * sample_rate))
    audio = np.concatenate([silence, audio, silence])

    # Add noise
    noise = 0.01 * np.random.randn(len(audio))
    audio = audio + noise

    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8

    # Save test file
    test_file = '/tmp/test_wideband.wav'
    save_wav(test_file, audio, sample_rate)
    print(f"Generated test file: {test_file}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Frequency range: {min(frequencies)}-{max(frequencies)} Hz")
    print(f"Expected {len(all_pulses)} pulses across {len(frequencies)} frequencies")

    # Run tracker
    tracker = SignalTracker(
        sample_rate=sample_rate,
        fft_size=4096,
        hop_size=1024,
        signal_threshold_db=8.0,
        pulse_threshold_db=5.0,
        min_pulse_width=0.03,
        max_pulse_width=0.3,
        peak_separation_hz=1000.0
    )

    output_file = '/tmp/test_wideband_output.json'
    tracker.process_wav_file(test_file, output_file)

    # Read and display results
    with open(output_file, 'r') as f:
        detected_pulses = [json.loads(line) for line in f]

    print(f"Detected {len(detected_pulses)} pulses")

    # Group by frequency
    freq_groups = {}
    for pulse in detected_pulses:
        freq = round(pulse['frequency'] / 1000) * 1000  # Round to nearest 1kHz
        if freq not in freq_groups:
            freq_groups[freq] = []
        freq_groups[freq].append(pulse)

    print("\nDetected frequencies:")
    for freq in sorted(freq_groups.keys()):
        print(f"  {freq:.0f} Hz: {len(freq_groups[freq])} pulses")

    print()


def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("* Signal Tracker Test Suite")
    print("*" * 60)
    print("\n")

    try:
        test_single_signal()
        test_multiple_signals()
        test_wideband()

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
        print("\nTest files generated in /tmp/:")
        print("  - test_single_signal.wav")
        print("  - test_single_signal_output.json")
        print("  - test_multiple_signals.wav")
        print("  - test_multiple_signals_output.json")
        print("  - test_wideband.wav")
        print("  - test_wideband_output.json")
        print()

    except Exception as e:
        print(f"\nError during testing: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
