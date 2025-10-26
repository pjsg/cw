#!/usr/bin/env python3
"""
Demo script for Wideband Morse Code Decoder

Demonstrates the complete pipeline on sample files.
"""

import subprocess
import json
import sys


def demo_file(wav_file, description=""):
    """Demonstrate the decoder on a WAV file."""
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print(f"Processing: {wav_file}\n")

    # Run signal tracker
    print("1. Signal Tracker:")
    print("   Analyzing spectrum and detecting pulses...")

    tracker_cmd = ['python', 'signal_tracker.py', wav_file]
    tracker_result = subprocess.run(
        tracker_cmd,
        capture_output=True,
        text=True
    )

    if tracker_result.returncode != 0:
        print(f"   ✗ Error: {tracker_result.stderr}")
        return

    pulse_lines = [l for l in tracker_result.stdout.strip().split('\n') if l.strip()]
    print(f"   ✓ Detected {len(pulse_lines)} pulses\n")

    # Run morse decoder
    print("2. Morse Decoder:")
    print("   Grouping by frequency and decoding...")

    decoder_cmd = ['python', 'morse_decoder.py', '/dev/stdin']
    decoder_result = subprocess.run(
        decoder_cmd,
        input=tracker_result.stdout,
        capture_output=True,
        text=True
    )

    if decoder_result.returncode != 0:
        print(f"   ✗ Error: {decoder_result.stderr}")
        return

    # Parse results
    messages = []
    for line in decoder_result.stdout.strip().split('\n'):
        if line.strip():
            try:
                messages.append(json.loads(line))
            except:
                pass

    if not messages:
        print("   ✗ No messages decoded")
        return

    print(f"   ✓ Decoded {len(messages)} message(s)\n")

    # Display results
    print("3. Results:")
    print("-" * 70)

    for msg in messages:
        freq = msg['frequency']
        text = msg['text']
        timestamp = msg['timestamp']

        # Format output
        print(f"   [{timestamp:6.3f}s] {freq:7.1f} Hz: \"{text}\"")

    print()


def main():
    """Run demo."""
    print("\n" + "*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  Wideband Morse Code Decoder - Demo".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Demo files
    demos = [
        ('sample.wav', 'Sample 1: Amateur Radio Callsign'),
        ('test1.wav', 'Sample 2: CQ Call'),
    ]

    for wav_file, description in demos:
        demo_file(wav_file, description)

    print("=" * 70)
    print("  Demo Complete!")
    print("=" * 70)
    print("\nTo decode your own files:")
    print("  python signal_tracker.py yourfile.wav | python morse_decoder.py /dev/stdin")
    print()


if __name__ == '__main__':
    main()
