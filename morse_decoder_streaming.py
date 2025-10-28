#!/usr/bin/env python3
"""
Streaming Morse Code Decoder

This module decodes morse code pulses from a continuous stream in real-time.
It processes pulses as they arrive, maintains state for each frequency, and
outputs decoded text immediately when characters/words are complete.

Input: JSON lines with pulse data (stdin)
Output: JSON lines with decoded text (stdout)
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np


# Morse code lookup table
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '.-.-.-': '.', '--..--': ',', '..--..': '?', '.----.': "'", '-.-.--': '!',
    '-..-.': '/', '-.--.': '(', '-.--.-': ')', '.-...': '&', '---...': ':',
    '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@'
}


@dataclass
class FrequencyState:
    """State for a single frequency channel."""
    frequency: float
    pulses: List[Dict] = field(default_factory=list)
    current_symbol: str = ""
    current_word: str = ""
    last_pulse_time: float = 0.0
    morse_params: Optional[Dict] = None
    params_estimated: bool = False
    first_pulse_time: float = 0.0
    pulses_count: int = 0
    pulses_count_last_estimate: int = 0
    pulse_time_last_estimate: float = 0.0

    def add_pulse(self, pulse: Dict):
        """Add a pulse and update state."""
        self.pulses.append(pulse)
        self.pulses_count += 1
        if len(self.pulses) > 100:
            self.pulses.pop(0)  # Keep pulse list manageable
        if not self.first_pulse_time:
            self.first_pulse_time = pulse['timestamp']
        self.last_pulse_time = pulse['timestamp'] + pulse['width']


class MorseDecoderStreaming:
    """
    Streaming morse code decoder for real-time operation.

    Processes pulses as they arrive, maintains per-frequency state,
    and outputs decoded text immediately when complete.
    """

    def __init__(
        self,
        char_timeout: float = 1.0,
        word_timeout: float = 3.0,
        min_pulses_for_params: int = 10,
        debug: bool = False
    ):
        """
        Initialize streaming decoder.

        Args:
            char_timeout: Time in seconds to wait before ending a character
            word_timeout: Time in seconds to wait before ending a word
            min_pulses_for_params: Minimum pulses before estimating parameters
            debug: Enable debug output to stderr
        """
        self.char_timeout = char_timeout
        self.word_timeout = word_timeout
        self.min_pulses_for_params = min_pulses_for_params
        self.debug = debug

        # Per-frequency state
        self.frequency_states: Dict[float, FrequencyState] = {}

    def process_pulse(self, pulse: Dict):
        """
        Process a single pulse and check for completed characters/words.

        Args:
            pulse: Pulse dictionary with timestamp, width, frequency
        """
        freq = round(pulse['frequency'])

        # Get or create frequency state
        if freq not in self.frequency_states:
            self.frequency_states[freq] = FrequencyState(frequency=freq)

        state = self.frequency_states[freq]

        # Use pulse timestamp for timeout checking (not real-time clock)
        pulse_time = pulse['timestamp']

        # Check for timeouts before adding new pulse
        self._check_timeouts(state, pulse_time)

        # Add pulse to state
        state.add_pulse(pulse)

        # Estimate parameters if we have enough pulses
        if (not state.params_estimated and len(state.pulses) >= self.min_pulses_for_params) \
            or state.pulses_count - state.pulses_count_last_estimate >= 50 \
            or state.last_pulse_time - state.pulse_time_last_estimate >= 20.0:
            state.morse_params = self._estimate_params(state.pulses)
            state.pulses_count_last_estimate = state.pulses_count
            state.pulse_time_last_estimate = state.last_pulse_time

            if not state.params_estimated:
                # This is the first estimate. We should backup and replay the pulses.
                state.params_estimated = True
                pulses_to_replay = state.pulses.copy()
                state.pulses = []
                state.current_symbol = ""
                state.current_word = ""
                state.first_pulse_time = 0.0
                state.last_pulse_time = 0.0

                for replay_pulse in pulses_to_replay:
                    self.process_pulse(replay_pulse)
                return  # Already processed this pulse in replay

            if self.debug:
                print(f"[{freq} Hz] Params: dit={state.morse_params['dit_time']*1000:.1f}ms "
                      f"dah={state.morse_params['dah_time']*1000:.1f}ms",
                      file=sys.stderr, flush=True)

        # If we have parameters, process the pulse
        if state.params_estimated:
            # Check gap from previous pulse to determine if we need to complete current symbol
            if len(state.pulses) > 1:
                prev_pulse = state.pulses[-2]
                gap = pulse['timestamp'] - (prev_pulse['timestamp'] + prev_pulse['width'])

                # Use adaptive gap thresholds based on dit time
                char_gap_threshold = state.morse_params['dit_time'] * 2.5
                word_gap_threshold = state.morse_params['dit_time'] * 6.0

                if gap >= char_gap_threshold and state.current_symbol:
                    # Character boundary - decode symbol and add to word
                    char = self._decode_symbol(state.current_symbol)
                    if char:
                        state.current_word += char
                    state.current_symbol = ""

                if gap >= word_gap_threshold and state.current_word:
                    # Word boundary - emit word
                    self._emit_text(state.frequency, state.current_word + " ", state.first_pulse_time)
                    state.current_word = ""
                    state.current_symbol = ""
                    state.first_pulse_time = pulse['timestamp']


            # Classify pulse as dit or dah and add to current symbol
            symbol = self._classify_pulse(pulse, state.morse_params)
            state.current_symbol += symbol

    def check_all_timeouts(self, current_time: float):
        """
        Check timeouts for all frequencies.

        Args:
            current_time: Current time
        """
        for state in self.frequency_states.values():
            self._check_timeouts(state, current_time)

    def _check_timeouts(self, state: FrequencyState, current_time: float):
        """
        Check if character or word should be completed based on timeout.

        Args:
            state: Frequency state
            current_time: Current time
        """
        if state.last_pulse_time == 0:
            return

        time_since_last = current_time - state.last_pulse_time

        # Character timeout - decode current symbol and add to word
        if time_since_last > self.char_timeout and state.current_symbol:
            char = self._decode_symbol(state.current_symbol)
            if char != '?':  # Only add valid characters
                state.current_word += char
            state.current_symbol = ""

        # Word timeout - output word and reset
        if time_since_last > self.word_timeout and state.current_word:
            self._emit_text(state.frequency, state.current_word, state.first_pulse_time)
            state.current_word = ""
            state.first_pulse_time = 0.0

    def _estimate_params(self, pulses: List[Dict]) -> Dict:
        """
        Estimate morse timing parameters from pulses.

        Uses same algorithm as batch decoder for consistency.

        Args:
            pulses: List of pulse dictionaries

        Returns:
            Dictionary with timing parameters
        """
        widths = np.array([p['width'] for p in pulses])

        # Use same clustering as batch decoder
        sorted_widths = np.sort(widths)

        # Find natural break between dit and dah
        if len(sorted_widths) >= 3:
            # Look for largest gap in ENTIRE width distribution
            width_diffs = np.diff(sorted_widths)

            # Find the largest gap anywhere in the distribution
            max_gap_idx = np.argmax(width_diffs)
            max_gap = width_diffs[max_gap_idx]

            # Only use this threshold if the gap is significant (>20% of mean width)
            if max_gap > np.mean(widths) * 0.2:
                threshold = (sorted_widths[max_gap_idx] + sorted_widths[max_gap_idx + 1]) / 2
            else:
                # No clear gap - assume 60% are dits, 40% are dahs (typical morse ratio)
                threshold_idx = int(len(sorted_widths) * 0.6)
                if threshold_idx >= len(sorted_widths):
                    threshold_idx = len(sorted_widths) - 1
                if threshold_idx > 0:
                    threshold = (sorted_widths[threshold_idx-1] + sorted_widths[threshold_idx]) / 2
                else:
                    threshold = sorted_widths[0]
        else:
            threshold = np.median(widths)

        dits = widths[widths < threshold]
        dahs = widths[widths >= threshold]

        dit_time = np.median(dits) if len(dits) > 0 else np.min(widths)
        dah_time = np.median(dahs) if len(dahs) > 0 else np.max(widths)

        return {
            'dit_time': dit_time,
            'dah_time': dah_time,
            'threshold': threshold
        }

    def _classify_pulse(self, pulse: Dict, params: Dict) -> str:
        """
        Classify pulse as dit or dah.

        Args:
            pulse: Pulse dictionary
            params: Morse parameters

        Returns:
            '.' for dit, '-' for dah
        """
        return '.' if pulse['width'] < params['threshold'] else '-'

    def _decode_symbol(self, symbol: str) -> str:
        """
        Decode a morse symbol to a character.

        Args:
            symbol: Morse symbol (string of dots and dashes)

        Returns:
            Decoded character or '?'
        """
        if symbol in MORSE_CODE_DICT:
            return MORSE_CODE_DICT[symbol]
        else:
            if self.debug:
                print(f"Unknown symbol: {symbol}", file=sys.stderr, flush=True)
            return "["+symbol+"]"

    def _emit_text(self, frequency: float, text: str, timestamp: float):
        """
        Emit decoded text as JSON line.

        Args:
            frequency: Frequency of signal
            text: Decoded text
        """
        if text:
            output = {
                'timestamp': timestamp,
                'text': text,
                'frequency': frequency
            }
            print(json.dumps(output), flush=True)


def read_pulse_stream():
    """
    Read pulse JSON lines from stdin.

    Yields:
        Pulse dictionaries
    """
    for line in sys.stdin:
        line = line.strip()
        if line:
            try:
                pulse = json.loads(line)
                yield pulse
            except json.JSONDecodeError:
                continue


def main():
    """Command line interface for streaming morse decoder."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Streaming morse decoder for continuous pulse input'
    )
    parser.add_argument(
        '--char-timeout',
        type=float,
        default=1.0,
        help='Character timeout in seconds (default: 1.0)'
    )
    parser.add_argument(
        '--word-timeout',
        type=float,
        default=3.0,
        help='Word timeout in seconds (default: 3.0)'
    )
    parser.add_argument(
        '--min-pulses',
        type=int,
        default=20,
        help='Minimum pulses for parameter estimation (default: 20)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )

    args = parser.parse_args()

    # Create streaming decoder
    decoder = MorseDecoderStreaming(
        char_timeout=args.char_timeout,
        word_timeout=args.word_timeout,
        min_pulses_for_params=args.min_pulses,
        debug=args.debug
    )

    # Process pulse stream
    try:
        last_pulse_time = 0.0

        for pulse in read_pulse_stream():
            # Process the pulse
            decoder.process_pulse(pulse)

            # Track latest pulse time
            last_pulse_time = pulse['timestamp'] + pulse['width']

        # Final timeout check when stream ends
        # Use last pulse time + enough delay to complete all words
        decoder.check_all_timeouts(last_pulse_time + args.word_timeout + 1.0)

    except KeyboardInterrupt:
        # Final flush on interrupt
        if last_pulse_time > 0:
            decoder.check_all_timeouts(last_pulse_time + args.word_timeout + 1.0)
    except BrokenPipeError:
        pass


if __name__ == '__main__':
    main()
