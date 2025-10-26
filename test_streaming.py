#!/usr/bin/env python3
"""
Test script for streaming pipeline

Converts WAV files to raw PCM and pipes through streaming components.
"""

import subprocess
import wave
import sys
import os


def wav_to_raw_pcm(wav_path):
    """
    Convert WAV file to raw PCM stream.

    Args:
        wav_path: Path to WAV file

    Yields:
        Raw PCM data chunks
    """
    with wave.open(wav_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        print(f"Converting {wav_path}:", file=sys.stderr)
        print(f"  Sample rate: {sample_rate} Hz", file=sys.stderr)
        print(f"  Channels: {channels}", file=sys.stderr)
        print(f"  Sample width: {sample_width} bytes", file=sys.stderr)

        # Read and yield chunks
        chunk_size = 8000  # Process in small chunks for streaming
        while True:
            data = wav.readframes(chunk_size)
            if not data:
                break
            yield data, sample_rate, channels, sample_width


def test_streaming_pipeline(wav_path):
    """
    Test streaming pipeline on a WAV file.

    Args:
        wav_path: Path to test WAV file
    """
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"Testing Streaming Pipeline: {wav_path}", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    # Get WAV properties
    with wave.open(wav_path, 'rb') as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

    # Start signal tracker
    tracker_cmd = [
        'python', 'signal_tracker_streaming.py',
        '--sample-rate', str(sample_rate),
        '--channels', str(channels),
        '--sample-width', str(sample_width)
    ]

    # Start morse decoder
    decoder_cmd = [
        'python', 'morse_decoder_streaming.py',
        '--debug'
    ]

    try:
        # Start both processes
        tracker = subprocess.Popen(
            tracker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr
        )

        decoder = subprocess.Popen(
            decoder_cmd,
            stdin=tracker.stdout,
            stdout=subprocess.PIPE,
            stderr=sys.stderr
        )

        # Allow tracker to write to decoder
        tracker.stdout.close()

        # Feed WAV data to tracker
        with wave.open(wav_path, 'rb') as wav:
            while True:
                data = wav.readframes(4000)
                if not data:
                    break
                tracker.stdin.write(data)

        # Close tracker input
        tracker.stdin.close()

        # Read decoder output
        print(f"\n{'='*80}", file=sys.stderr)
        print("Decoded Output:", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)

        decoded_count = 0
        for line in decoder.stdout:
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(decoded_line)
                decoded_count += 1

        # Wait for completion
        tracker.wait()
        decoder.wait()

        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Test complete: {decoded_count} messages decoded", file=sys.stderr)
        print(f"{'='*80}\n", file=sys.stderr)

        return decoded_count > 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


def main():
    """Run streaming tests."""
    import argparse

    parser = argparse.ArgumentParser(description='Test streaming pipeline')
    parser.add_argument('wav_files', nargs='+', help='WAV files to test')

    args = parser.parse_args()

    results = []
    for wav_file in args.wav_files:
        if not os.path.exists(wav_file):
            print(f"File not found: {wav_file}", file=sys.stderr)
            results.append(False)
            continue

        success = test_streaming_pipeline(wav_file)
        results.append(success)

    # Summary
    print(f"\n{'='*80}", file=sys.stderr)
    print("Test Summary:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)

    for wav_file, success in zip(args.wav_files, results):
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {wav_file}", file=sys.stderr)

    total = len(results)
    passed = sum(results)
    print(f"\nTotal: {passed}/{total} passed", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
