#!/usr/bin/env python3
"""
Streaming Signal Tracker for Wideband Morse Decoder

This module processes continuous audio streams in real-time, detecting morse code
signals and outputting pulse timing information as JSON lines immediately.

Input: Raw PCM audio stream (stdin)
Output: JSON lines with pulse data (stdout)
"""

import numpy as np
import scipy.signal as signal
import json
import sys
import struct
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import defaultdict
import wave


@dataclass
class StreamingSignal:
    """Represents a tracked signal in streaming mode."""
    frequency: float
    bin_index: int
    pulse_start: Optional[float] = None
    last_level: float = 0.0
    last_seen: float = 0.0


class SignalTrackerStreaming:
    """
    Streaming version of signal tracker for real-time audio processing.

    Processes audio in chunks, maintains state across chunks, and outputs
    pulses immediately as they are detected.
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_duration: float = 1.0,
        fft_size: int = 2048,
        hop_size: int = 512,
        signal_threshold_db: float = 10.0,
        pulse_threshold_db: float = 6.0,
        min_pulse_width: float = 0.01,
        max_pulse_width: float = 1.0,
        peak_separation_hz: float = 50.0,
        signal_timeout: float = 5.0
    ):
        """
        Initialize streaming signal tracker.

        Args:
            sample_rate: Sample rate of input audio in Hz
            chunk_duration: Duration of processing chunks in seconds
            fft_size: Size of FFT for spectral analysis
            hop_size: Number of samples between FFT frames
            signal_threshold_db: Threshold above noise floor to detect signals
            pulse_threshold_db: Threshold for pulse on/off detection
            min_pulse_width: Minimum pulse width in seconds
            max_pulse_width: Maximum pulse width in seconds
            peak_separation_hz: Minimum frequency separation between signals
            signal_timeout: Remove signals not seen for this many seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.signal_threshold_db = signal_threshold_db
        self.pulse_threshold_db = pulse_threshold_db
        self.min_pulse_width = min_pulse_width
        self.max_pulse_width = max_pulse_width
        self.peak_separation_hz = peak_separation_hz
        self.signal_timeout = signal_timeout

        # Time tracking
        self.hop_time = hop_size / sample_rate
        self.total_time = 0.0

        # Window and frequency bins
        self.window = np.hamming(fft_size)
        self.freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

        # Active signals being tracked
        self.active_signals: Dict[int, StreamingSignal] = {}

        # Overlap buffer for chunk boundaries
        self.overlap_size = fft_size
        self.overlap_buffer = np.array([], dtype=np.float32)

    def process_chunk(self, audio_chunk: np.ndarray):
        """
        Process a chunk of audio and emit detected pulses immediately.

        Args:
            audio_chunk: Audio samples (float32)
        """
        # Prepend overlap from previous chunk
        if len(self.overlap_buffer) > 0:
            audio = np.concatenate([self.overlap_buffer, audio_chunk])
        else:
            audio = audio_chunk

        # Save overlap for next chunk
        if len(audio) >= self.overlap_size:
            self.overlap_buffer = audio[-self.overlap_size:].copy()

        # Process audio using envelope detection for each detected frequency
        self._process_chunk_envelope(audio)

        # Clean up stale signals
        self._cleanup_stale_signals()

    def _process_chunk_envelope(self, audio: np.ndarray):
        """
        Process chunk using envelope detection method.

        Args:
            audio: Audio chunk with overlap
        """
        chunk_start_time = self.total_time

        # First, detect active frequencies in this chunk
        signal_freqs = self._find_signal_frequencies_in_chunk(audio)

        # Update active signals
        current_time = self.total_time + len(audio) / self.sample_rate
        for freq in signal_freqs:
            bin_idx = np.argmin(np.abs(self.freq_bins - freq))
            if bin_idx not in self.active_signals:
                self.active_signals[bin_idx] = StreamingSignal(
                    frequency=freq,
                    bin_index=bin_idx,
                    last_seen=current_time
                )
            else:
                self.active_signals[bin_idx].last_seen = current_time

        # Extract and emit pulses for each active signal
        for sig in list(self.active_signals.values()):
            pulses = self._extract_pulses_from_chunk(audio, sig.frequency, chunk_start_time)

            # Emit pulses immediately
            for pulse in pulses:
                self._emit_pulse(pulse)

        # Update time
        self.total_time += len(audio) / self.sample_rate

    def _find_signal_frequencies_in_chunk(self, audio: np.ndarray) -> List[float]:
        """
        Find active signal frequencies in audio chunk.

        Args:
            audio: Audio chunk

        Returns:
            List of detected frequencies
        """
        # Compute average spectrum
        n_frames = (len(audio) - self.fft_size) // self.hop_size
        if n_frames <= 0:
            return []

        spectra = []
        for i in range(0, len(audio) - self.fft_size + 1, self.hop_size):
            frame = audio[i:i + self.fft_size]
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            spectra.append(magnitude)

        if not spectra:
            return []

        avg_spectrum = np.mean(spectra, axis=0)
        magnitude_db = 20 * np.log10(avg_spectrum + 1e-10)

        # Find peaks
        noise_floor = np.median(magnitude_db)
        threshold = noise_floor + self.signal_threshold_db

        peak_indices = signal.argrelmax(magnitude_db, order=5)[0]

        valid_freqs = []
        for idx in peak_indices:
            if magnitude_db[idx] > threshold:
                freq = self.freq_bins[idx]

                # Check separation
                too_close = False
                for other_freq in valid_freqs:
                    if abs(freq - other_freq) < self.peak_separation_hz:
                        too_close = True
                        break

                if not too_close:
                    valid_freqs.append(freq)

        return valid_freqs

    def _extract_pulses_from_chunk(
        self,
        audio: np.ndarray,
        center_freq: float,
        chunk_start_time: float
    ) -> List[Dict]:
        """
        Extract pulses at a specific frequency from chunk.

        Args:
            audio: Audio chunk
            center_freq: Center frequency to extract
            chunk_start_time: Start time of this chunk

        Returns:
            List of pulse dictionaries
        """
        pulses = []

        # Design bandpass filter
        bandwidth = 200.0
        nyquist = self.sample_rate / 2
        low = max(center_freq - bandwidth / 2, 10) / nyquist
        high = min(center_freq + bandwidth / 2, nyquist - 10) / nyquist

        if low >= high or low <= 0 or high >= 1:
            return []

        try:
            # Apply filter and get envelope
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio)

            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)

            # Smooth envelope
            smooth_window = max(1, int(0.005 * self.sample_rate))
            kernel = np.ones(smooth_window) / smooth_window
            envelope = np.convolve(envelope, kernel, mode='same')

            # Detect pulses
            envelope_threshold = np.max(envelope) * 0.3

            is_high = envelope > envelope_threshold
            transitions = np.diff(is_high.astype(int))

            on_indices = np.where(transitions == 1)[0]
            off_indices = np.where(transitions == -1)[0]

            if len(on_indices) == 0 or len(off_indices) == 0:
                return []

            # Match pairs
            if off_indices[0] < on_indices[0]:
                off_indices = off_indices[1:]
            if len(on_indices) > len(off_indices):
                on_indices = on_indices[:len(off_indices)]

            # Create pulses
            for on_idx, off_idx in zip(on_indices, off_indices):
                timestamp = chunk_start_time + on_idx / self.sample_rate
                width = (off_idx - on_idx) / self.sample_rate

                if self.min_pulse_width <= width <= self.max_pulse_width:
                    pulses.append({
                        'timestamp': timestamp,
                        'width': width,
                        'frequency': center_freq
                    })

            return pulses

        except Exception:
            return []

    def _emit_pulse(self, pulse: Dict):
        """
        Emit a pulse as JSON line to stdout immediately.

        Args:
            pulse: Pulse dictionary with timestamp, width, frequency
        """
        print(json.dumps(pulse), flush=True)

    def _cleanup_stale_signals(self):
        """Remove signals that haven't been seen recently."""
        current_time = self.total_time
        stale = [
            bin_idx for bin_idx, sig in self.active_signals.items()
            if current_time - sig.last_seen > self.signal_timeout
        ]
        for bin_idx in stale:
            del self.active_signals[bin_idx]


def read_audio_stream(sample_rate: int, chunk_duration: float, channels: int = 1, sample_width: int = 2):
    """
    Read raw PCM audio from stdin in chunks.

    Args:
        sample_rate: Sample rate in Hz
        chunk_duration: Chunk duration in seconds
        channels: Number of channels (1=mono, 2=stereo)
        sample_width: Bytes per sample (2=int16, 4=int32)

    Yields:
        Audio chunks as numpy arrays (mono, float32)
    """
    chunk_samples = int(sample_rate * chunk_duration)
    chunk_bytes = chunk_samples * channels * sample_width

    while True:
        # Read chunk from stdin
        data = sys.stdin.buffer.read(chunk_bytes)

        if not data:
            break

        # Convert to numpy array
        if sample_width == 2:
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo - use first channel
        if channels == 2:
            audio = audio.reshape(-1, 2)[:, 0]

        yield audio


def main():
    """Command line interface for streaming signal tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Streaming signal tracker for continuous audio input'
    )
    parser.add_argument(
        '--wav-file',
        action='store_true',
        default=False,
        help='Input is a WAV file (default: False, raw PCM from stdin)'
    )
    parser.add_argument(
        '-r', '--sample-rate',
        type=int,
        default=8000,
        help='Sample rate in Hz (default: 8000)'
    )
    parser.add_argument(
        '-c', '--channels',
        type=int,
        default=1,
        choices=[1, 2],
        help='Number of channels (default: 1)'
    )
    parser.add_argument(
        '-w', '--sample-width',
        type=int,
        default=2,
        choices=[2, 4],
        help='Sample width in bytes (default: 2 for int16)'
    )
    parser.add_argument(
        '--chunk-duration',
        type=float,
        default=1.0,
        help='Processing chunk duration in seconds (default: 1.0)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=10.0,
        help='Signal detection threshold in dB (default: 10.0)'
    )
    parser.add_argument(
        '-p', '--pulse-threshold',
        type=float,
        default=6.0,
        help='Pulse detection threshold in dB (default: 6.0)'
    )
    parser.add_argument(
        '--min-pulse',
        type=float,
        default=0.01,
        help='Minimum pulse width in seconds (default: 0.01)'
    )
    parser.add_argument(
        '--max-pulse',
        type=float,
        default=1.0,
        help='Maximum pulse width in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    if args.wav_file:
        # Process WAV file

        # just read the wav file header
        with wave.open(sys.stdin.buffer, mode='rb') as wav_file:
            args.sample_rate = wav_file.getframerate()
            args.channels = wav_file.getnchannels()
            args.sample_width = wav_file.getsampwidth()

    # Create streaming tracker
    tracker = SignalTrackerStreaming(
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        signal_threshold_db=args.threshold,
        pulse_threshold_db=args.pulse_threshold,
        min_pulse_width=args.min_pulse,
        max_pulse_width=args.max_pulse
    )

    # Process audio stream
    try:
        for audio_chunk in read_audio_stream(
            args.sample_rate,
            args.chunk_duration,
            args.channels,
            args.sample_width
        ):
            tracker.process_chunk(audio_chunk)
    except KeyboardInterrupt:
        pass
    except BrokenPipeError:
        # Output pipe closed
        pass


if __name__ == '__main__':
    main()
