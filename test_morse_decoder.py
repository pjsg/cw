#!/usr/bin/env python3
"""
Test script for Morse Code Decoder

Tests the complete pipeline: signal tracker → morse decoder
"""

import subprocess
import json
import sys
import os


def run_pipeline(wav_file, description=""):
    """Run the complete signal tracker + decoder pipeline."""
    print("=" * 70)
    if description:
        print(f"Test: {description}")
    print(f"Input: {wav_file}")
    print("=" * 70)

    # Check if file exists
    if not os.path.exists(wav_file):
        print(f"✗ File not found: {wav_file}")
        return False

    try:
        # Run signal tracker
        tracker_cmd = ['python', 'signal_tracker.py', wav_file]
        tracker_result = subprocess.run(
            tracker_cmd,
            capture_output=True,
            text=True,
            check=True
        )

        pulses_json = tracker_result.stdout
        if not pulses_json.strip():
            print("✗ No pulses detected by signal tracker")
            return False

        # Count pulses
        pulse_lines = [line for line in pulses_json.strip().split('\n') if line.strip()]
        print(f"Signal Tracker: Detected {len(pulse_lines)} pulses")

        # Run morse decoder
        decoder_cmd = ['python', 'morse_decoder.py', '/dev/stdin']
        decoder_result = subprocess.run(
            decoder_cmd,
            input=pulses_json,
            capture_output=True,
            text=True,
            check=True
        )

        # Parse and display results
        messages = []
        for line in decoder_result.stdout.strip().split('\n'):
            if line.strip():
                try:
                    msg = json.loads(line)
                    messages.append(msg)
                except json.JSONDecodeError:
                    continue

        if not messages:
            print("✗ No messages decoded")
            return False

        print(f"Morse Decoder: Decoded {len(messages)} message(s)\n")

        for i, msg in enumerate(messages, 1):
            print(f"Message {i}:")
            print(f"  Frequency: {msg['frequency']:.1f} Hz")
            print(f"  Timestamp: {msg['timestamp']:.3f}s")
            print(f"  Text: '{msg['text']}'")

        print("\n✓ Test passed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Error running pipeline: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("* Morse Code Decoder Test Suite")
    print("*" * 70)
    print("\n")

    tests = [
        ('sample.wav', 'Single transmission - callsign N4LSJ CA'),
        ('test1.wav', 'CQ call from W1ABK'),
    ]

    passed = 0
    failed = 0

    for wav_file, description in tests:
        if run_pipeline(wav_file, description):
            passed += 1
        else:
            failed += 1
        print()

    # Summary
    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
