#!/usr/bin/env python3
"""
Signal Tracker for Wideband Morse Decoder

This module detects and tracks morse code signals across a wide frequency range.
It processes WAV audio streams, identifies signals using FFT, extracts them using
polyphase filter banks, and outputs pulse timing information in JSON format.
"""

import numpy as np
import scipy.signal as signal
import wave
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


@dataclass
class Pulse:
    """Represents a detected morse code pulse."""
    timestamp: float  # Seconds since start of stream
    width: float      # Duration in seconds
    frequency: float  # Center frequency in Hz


@dataclass
class Signal:
    """Represents a tracked signal at a specific frequency."""
    frequency: float
    bin_index: int
    last_level: float = 0.0
    pulse_start: Optional[float] = None
    samples_processed: int = 0
    # Noise filtering - keep track of recent levels
    level_history: List[float] = None

    def __post_init__(self):
        if self.level_history is None:
            self.level_history = []


class SignalTracker:
    """
    Tracks multiple morse code signals across a wide frequency band.

    Uses FFT for signal detection and polyphase filter banks for efficient
    signal extraction and pulse detection.
    """

    def __init__(
        self,
        sample_rate: int,
        fft_size: int = 2048,
        hop_size: int = 512,
        signal_threshold_db: float = 10.0,
        pulse_threshold_db: float = 6.0,
        min_pulse_width: float = 0.01,  # 10ms minimum
        max_pulse_width: float = 1.0,   # 1s maximum
        peak_separation_hz: float = 50.0,
        overlap_time: float = 0.5  # Overlap for chunk boundary handling
    ):
        """
        Initialize the signal tracker.

        Args:
            sample_rate: Sample rate of input audio in Hz
            fft_size: Size of FFT for spectral analysis
            hop_size: Number of samples between FFT frames
            signal_threshold_db: Threshold above noise floor to detect signals
            pulse_threshold_db: Threshold for pulse on/off detection
            min_pulse_width: Minimum pulse width in seconds
            max_pulse_width: Maximum pulse width in seconds
            peak_separation_hz: Minimum frequency separation between signals
            overlap_time: Overlap duration for chunk boundary handling
        """
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.signal_threshold_db = signal_threshold_db
        self.pulse_threshold_db = pulse_threshold_db
        self.min_pulse_width = min_pulse_width
        self.max_pulse_width = max_pulse_width
        self.peak_separation_hz = peak_separation_hz
        self.overlap_time = overlap_time

        # Time per hop in seconds
        self.hop_time = hop_size / sample_rate

        # Window function for FFT
        self.window = np.hamming(fft_size)

        # Frequency bins
        self.freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)

        # Active signals being tracked
        self.active_signals: Dict[int, Signal] = {}

        # Total time processed
        self.total_time = 0.0

        # Buffer for overlap processing
        self.overlap_buffer = np.array([], dtype=np.float32)

    def process_wav_file(self, wav_path: str, output_path: Optional[str] = None):
        """
        Process a WAV file and output detected pulses.

        Args:
            wav_path: Path to input WAV file
            output_path: Path to output JSON lines file (stdout if None)
        """
        # Open WAV file
        with wave.open(wav_path, 'rb') as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()

            if rate != self.sample_rate:
                print(f"Warning: WAV file sample rate ({rate}) doesn't match "
                      f"configured rate ({self.sample_rate})", file=sys.stderr)

            # Read all audio data
            audio_bytes = wav_file.readframes(n_frames)

        # Convert to numpy array
        if sample_width == 1:
            audio = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
            audio = (audio - 128) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32)
            audio = audio / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        # Handle stereo - use first channel only
        if channels == 2:
            audio = audio.reshape(-1, 2)[:, 0]

        # Process audio
        pulses = self.process_audio(audio)

        # Output pulses
        self._output_pulses(pulses, output_path)

    def process_audio(self, audio: np.ndarray, use_envelope_detection: bool = True) -> List[Pulse]:
        """
        Process audio data and detect pulses.

        Args:
            audio: Audio samples as float32 array
            use_envelope_detection: Use accurate envelope detection for pulse timing

        Returns:
            List of detected pulses
        """
        if use_envelope_detection:
            return self._process_with_envelope_detection(audio)

        all_pulses = []

        # Process audio in frames with overlap
        for i in range(0, len(audio) - self.fft_size + 1, self.hop_size):
            frame = audio[i:i + self.fft_size]
            frame_time = self.total_time

            # Detect active signals in this frame
            self._update_signal_detection(frame, frame_time)

            # Track pulse states for active signals
            pulses = self._track_pulses(frame, frame_time)
            all_pulses.extend(pulses)

            self.total_time += self.hop_time

        # Finalize any ongoing pulses
        final_pulses = self._finalize_pulses()
        all_pulses.extend(final_pulses)

        # Remove duplicate pulses (from chunk overlap processing)
        all_pulses = self._remove_duplicate_pulses(all_pulses)

        return all_pulses

    def _process_with_envelope_detection(self, audio: np.ndarray) -> List[Pulse]:
        """
        Process audio using envelope detection for accurate pulse timing.

        This method:
        1. Scans the entire audio to find signal frequencies
        2. For each signal, applies bandpass filter and envelope detection
        3. Detects pulses from envelope with accurate timing

        Args:
            audio: Audio samples as float32 array

        Returns:
            List of detected pulses
        """
        # Step 1: Find signal frequencies by analyzing spectrum
        signal_freqs = self._find_signal_frequencies(audio)

        if not signal_freqs:
            return []

        # Step 2: For each frequency, extract pulses using envelope detection
        all_pulses = []
        for freq in signal_freqs:
            pulses = self._extract_pulses_at_frequency(audio, freq)
            all_pulses.extend(pulses)

        return all_pulses

    def _find_signal_frequencies(self, audio: np.ndarray) -> List[float]:
        """
        Find all signal frequencies present in the audio.

        Args:
            audio: Audio samples

        Returns:
            List of detected signal frequencies
        """
        # Compute average spectrum across the audio
        n_frames = (len(audio) - self.fft_size) // self.hop_size
        spectra = []

        for i in range(0, len(audio) - self.fft_size + 1, self.hop_size):
            frame = audio[i:i + self.fft_size]
            windowed = frame * self.window
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            spectra.append(magnitude)

        # Average spectrum
        avg_spectrum = np.mean(spectra, axis=0)
        magnitude_db = 20 * np.log10(avg_spectrum + 1e-10)

        # Find peaks
        noise_floor = np.median(magnitude_db)
        threshold = noise_floor + self.signal_threshold_db

        # Find local maxima
        peak_indices = signal.argrelmax(magnitude_db, order=5)[0]

        # Filter by threshold and separation
        valid_freqs = []
        for idx in peak_indices:
            if magnitude_db[idx] > threshold:
                freq = self.freq_bins[idx]

                # Check separation from existing peaks
                too_close = False
                for other_freq in valid_freqs:
                    if abs(freq - other_freq) < self.peak_separation_hz:
                        too_close = True
                        break

                if not too_close:
                    valid_freqs.append(freq)

        return valid_freqs

    def _extract_pulses_at_frequency(self, audio: np.ndarray, center_freq: float) -> List[Pulse]:
        """
        Extract pulses at a specific frequency using envelope detection.

        Args:
            audio: Input audio signal
            center_freq: Center frequency to extract

        Returns:
            List of pulses at this frequency
        """
        # Design bandpass filter
        bandwidth = 200.0  # Hz
        nyquist = self.sample_rate / 2
        low = max(center_freq - bandwidth / 2, 10) / nyquist
        high = min(center_freq + bandwidth / 2, nyquist - 10) / nyquist

        # Validate filter parameters
        if low >= high or low <= 0 or high >= 1:
            return []

        try:
            # Apply bandpass filter
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            filtered = signal.sosfilt(sos, audio)

            # Compute envelope using Hilbert transform
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)

            # Smooth envelope
            smooth_window = max(1, int(0.005 * self.sample_rate))  # 5ms smoothing
            kernel = np.ones(smooth_window) / smooth_window
            envelope = np.convolve(envelope, kernel, mode='same')

            # Determine threshold for pulse detection
            envelope_threshold = np.max(envelope) * 0.3  # 30% of peak

            # Find threshold crossings
            is_high = envelope > envelope_threshold
            transitions = np.diff(is_high.astype(int))

            on_indices = np.where(transitions == 1)[0]
            off_indices = np.where(transitions == -1)[0]

            # Match up on/off pairs
            if len(on_indices) == 0 or len(off_indices) == 0:
                return []

            # Handle edge cases
            if off_indices[0] < on_indices[0]:
                off_indices = off_indices[1:]
            if len(on_indices) > len(off_indices):
                on_indices = on_indices[:len(off_indices)]

            # Create pulses
            pulses = []
            for on_idx, off_idx in zip(on_indices, off_indices):
                timestamp = on_idx / self.sample_rate
                width = (off_idx - on_idx) / self.sample_rate

                # Validate pulse width
                if self.min_pulse_width <= width <= self.max_pulse_width:
                    pulse = Pulse(
                        timestamp=timestamp,
                        width=width,
                        frequency=center_freq
                    )
                    pulses.append(pulse)

            return pulses

        except Exception as e:
            print(f"Error extracting pulses at {center_freq} Hz: {e}", file=sys.stderr)
            return []

    def _remove_duplicate_pulses(self, pulses: List[Pulse]) -> List[Pulse]:
        """
        Remove duplicate pulses that may occur from overlapping chunk processing.

        Args:
            pulses: List of pulses possibly containing duplicates

        Returns:
            List of unique pulses
        """
        if not pulses:
            return pulses

        # Sort by timestamp
        sorted_pulses = sorted(pulses, key=lambda p: (p.frequency, p.timestamp))

        # Remove duplicates - two pulses are duplicates if they're on the same
        # frequency and their timestamps are within 50ms of each other
        unique_pulses = []
        for pulse in sorted_pulses:
            if not unique_pulses:
                unique_pulses.append(pulse)
                continue

            last_pulse = unique_pulses[-1]

            # Check if this is a duplicate
            is_duplicate = (
                abs(pulse.frequency - last_pulse.frequency) < 10 and  # Same frequency
                abs(pulse.timestamp - last_pulse.timestamp) < 0.05  # Within 50ms
            )

            if not is_duplicate:
                unique_pulses.append(pulse)
            else:
                # Keep the pulse with better quality (longer duration typically better)
                if pulse.width > last_pulse.width:
                    unique_pulses[-1] = pulse

        return unique_pulses

    def _update_signal_detection(self, frame: np.ndarray, frame_time: float):
        """
        Detect signals in the current frame using FFT.

        Args:
            frame: Audio frame
            frame_time: Timestamp of frame
        """
        # Apply window and compute FFT
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)

        # Convert to dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Estimate noise floor (median of spectrum)
        noise_floor = np.median(magnitude_db)

        # Find peaks above threshold
        threshold = noise_floor + self.signal_threshold_db

        # Find local maxima
        peak_indices = signal.argrelmax(magnitude_db, order=5)[0]

        # Filter peaks by threshold and separation
        valid_peaks = []
        for idx in peak_indices:
            if magnitude_db[idx] > threshold:
                # Check separation from existing peaks
                freq = self.freq_bins[idx]
                too_close = False
                for other_freq, _ in valid_peaks:
                    if abs(freq - other_freq) < self.peak_separation_hz:
                        too_close = True
                        break

                if not too_close:
                    valid_peaks.append((freq, idx))

        # Update active signals
        current_bins = {idx for _, idx in valid_peaks}

        # Remove signals that are no longer present
        bins_to_remove = [b for b in self.active_signals.keys() if b not in current_bins]
        for bin_idx in bins_to_remove:
            del self.active_signals[bin_idx]

        # Add new signals
        for freq, idx in valid_peaks:
            if idx not in self.active_signals:
                self.active_signals[idx] = Signal(
                    frequency=freq,
                    bin_index=idx,
                    last_level=magnitude_db[idx]
                )

    def _extract_signal_with_filter(
        self,
        audio: np.ndarray,
        center_freq: float,
        bandwidth: float = 200.0
    ) -> np.ndarray:
        """
        Extract a signal using a bandpass filter (polyphase filter bank approach).

        Args:
            audio: Input audio signal
            center_freq: Center frequency to extract
            bandwidth: Bandwidth of filter in Hz

        Returns:
            Filtered signal envelope
        """
        # Design bandpass filter
        nyquist = self.sample_rate / 2
        low = max(center_freq - bandwidth / 2, 0) / nyquist
        high = min(center_freq + bandwidth / 2, nyquist - 1) / nyquist

        # Ensure valid filter parameters
        if low >= high or low <= 0 or high >= 1:
            return np.zeros(len(audio))

        # Create bandpass filter using polyphase structure
        # Using a decimated approach for efficiency
        decimate_factor = max(1, int(self.sample_rate / (4 * bandwidth)))

        try:
            # Design filter
            sos = signal.butter(6, [low, high], btype='band', output='sos')

            # Apply filter
            filtered = signal.sosfilt(sos, audio)

            # Compute envelope using Hilbert transform
            analytic = signal.hilbert(filtered)
            envelope = np.abs(analytic)

            # Smooth envelope
            smooth_window = int(0.001 * self.sample_rate)  # 1ms smoothing
            if smooth_window > 0:
                envelope = np.convolve(
                    envelope,
                    np.ones(smooth_window) / smooth_window,
                    mode='same'
                )

            return envelope

        except Exception as e:
            print(f"Filter design failed for {center_freq} Hz: {e}", file=sys.stderr)
            return np.zeros(len(audio))

    def _track_pulses(self, frame: np.ndarray, frame_time: float) -> List[Pulse]:
        """
        Track pulse states for active signals using polyphase filtering.

        Args:
            frame: Audio frame
            frame_time: Timestamp of frame

        Returns:
            List of completed pulses
        """
        pulses = []

        if not self.active_signals:
            return pulses

        # Compute spectrum for this frame
        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        # Estimate noise floor for pulse threshold
        noise_floor = np.median(magnitude_db)
        pulse_threshold = noise_floor + self.pulse_threshold_db

        # Check each active signal
        for bin_idx, sig in list(self.active_signals.items()):
            level = magnitude_db[bin_idx]

            # Add to level history for noise filtering
            sig.level_history.append(level)
            if len(sig.level_history) > 10:
                sig.level_history.pop(0)

            # Use smoothed level for better noise rejection
            if len(sig.level_history) >= 3:
                smoothed_level = np.median(sig.level_history[-3:])
            else:
                smoothed_level = level

            # Adaptive threshold - compute local noise floor for this signal
            if len(sig.level_history) >= 5:
                local_noise = np.percentile(sig.level_history, 25)
                adaptive_threshold = local_noise + self.pulse_threshold_db
            else:
                adaptive_threshold = pulse_threshold

            # Pulse detection using threshold crossing with hysteresis
            # Use hysteresis to prevent chattering
            turn_on_threshold = adaptive_threshold
            turn_off_threshold = adaptive_threshold - 2.0  # 2dB hysteresis

            if smoothed_level > turn_on_threshold:
                # Signal is high - pulse is on
                if sig.pulse_start is None:
                    # Start of new pulse
                    sig.pulse_start = frame_time
            elif smoothed_level < turn_off_threshold:
                # Signal is low - pulse is off
                if sig.pulse_start is not None:
                    # End of pulse
                    pulse_width = frame_time - sig.pulse_start

                    # Validate pulse width
                    if self.min_pulse_width <= pulse_width <= self.max_pulse_width:
                        pulse = Pulse(
                            timestamp=sig.pulse_start,
                            width=pulse_width,
                            frequency=sig.frequency
                        )
                        pulses.append(pulse)

                    sig.pulse_start = None

            sig.last_level = smoothed_level

        return pulses

    def _finalize_pulses(self) -> List[Pulse]:
        """
        Finalize any ongoing pulses at the end of processing.

        Returns:
            List of completed pulses
        """
        pulses = []

        for sig in self.active_signals.values():
            if sig.pulse_start is not None:
                pulse_width = self.total_time - sig.pulse_start

                # Validate pulse width
                if self.min_pulse_width <= pulse_width <= self.max_pulse_width:
                    pulse = Pulse(
                        timestamp=sig.pulse_start,
                        width=pulse_width,
                        frequency=sig.frequency
                    )
                    pulses.append(pulse)

        return pulses

    def _output_pulses(self, pulses: List[Pulse], output_path: Optional[str] = None):
        """
        Output pulses in JSON lines format.

        Args:
            pulses: List of pulses to output
            output_path: Path to output file (stdout if None)
        """
        if output_path:
            with open(output_path, 'w') as f:
                for pulse in pulses:
                    json_obj = {
                        'timestamp': pulse.timestamp,
                        'width': pulse.width,
                        'frequency': pulse.frequency
                    }
                    f.write(json.dumps(json_obj) + '\n')
        else:
            for pulse in pulses:
                json_obj = {
                    'timestamp': pulse.timestamp,
                    'width': pulse.width,
                    'frequency': pulse.frequency
                }
                print(json.dumps(json_obj))


def main():
    """Command line interface for signal tracker."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Track morse code signals in wideband audio'
    )
    parser.add_argument(
        'input',
        help='Input WAV file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON lines file (default: stdout)'
    )
    parser.add_argument(
        '-s', '--sample-rate',
        type=int,
        help='Override sample rate (default: use WAV file rate)'
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

    # Determine sample rate
    if args.sample_rate:
        sample_rate = args.sample_rate
    else:
        # Read sample rate from WAV file
        with wave.open(args.input, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()

    # Create tracker
    tracker = SignalTracker(
        sample_rate=sample_rate,
        signal_threshold_db=args.threshold,
        pulse_threshold_db=args.pulse_threshold,
        min_pulse_width=args.min_pulse,
        max_pulse_width=args.max_pulse
    )

    # Process file
    tracker.process_wav_file(args.input, args.output)


if __name__ == '__main__':
    main()
