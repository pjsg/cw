#!/usr/bin/env python3
"""
Morse Code Decoder

This module decodes morse code pulses into ASCII text. It consumes JSON output
from the signal tracker, groups pulses by frequency, estimates timing parameters,
and decodes the morse patterns into readable text.
"""

import numpy as np
import json
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# International Morse Code lookup table
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
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
}


@dataclass
class Pulse:
    """Represents a morse code pulse."""
    timestamp: float
    width: float
    frequency: float


@dataclass
class MorseParameters:
    """Estimated morse code timing parameters."""
    dit_time: float
    dah_time: float
    intra_symbol_space: float
    inter_symbol_space: float
    inter_word_space: float
    
    def __str__(self):
        return (f"Dit: {self.dit_time*1000:.1f}ms, Dah: {self.dah_time*1000:.1f}ms, "
                f"Intra: {self.intra_symbol_space*1000:.1f}ms, "
                f"Inter-char: {self.inter_symbol_space*1000:.1f}ms, "
                f"Inter-word: {self.inter_word_space*1000:.1f}ms")


@dataclass
class DecodedMessage:
    """Represents a decoded morse message."""
    timestamp: float
    text: str
    frequency: float


class MorseDecoder:
    """
    Decodes morse code pulses into ASCII text.
    
    Handles both machine-generated morse (standard ratios) and human-generated
    morse with variable timing.
    """
    
    def __init__(
        self,
        min_pulses_for_decode: int = 5,
        adaptive_timing: bool = True,
        debug: bool = False
    ):
        """
        Initialize the morse decoder.
        
        Args:
            min_pulses_for_decode: Minimum pulses required to attempt decoding
            adaptive_timing: Use adaptive timing parameter estimation
            debug: Enable debug output
        """
        self.min_pulses_for_decode = min_pulses_for_decode
        self.adaptive_timing = adaptive_timing
        self.debug = debug
    
    def decode_from_file(self, input_path: str, output_path: Optional[str] = None):
        """
        Decode morse code from a JSON lines file.
        
        Args:
            input_path: Path to input JSON lines file (from signal tracker)
            output_path: Path to output JSON lines file (stdout if None)
        """
        # Read pulses from file
        pulses = []
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pulse = Pulse(
                        timestamp=data['timestamp'],
                        width=data['width'],
                        frequency=data['frequency']
                    )
                    pulses.append(pulse)
        
        # Decode pulses
        messages = self.decode_pulses(pulses)
        
        # Output messages
        self._output_messages(messages, output_path)
    
    def decode_pulses(self, pulses: List[Pulse]) -> List[DecodedMessage]:
        """
        Decode a list of pulses into messages.
        
        Args:
            pulses: List of morse code pulses
            
        Returns:
            List of decoded messages
        """
        # Group pulses by frequency
        frequency_groups = self._group_by_frequency(pulses)
        
        # Decode each frequency group
        all_messages = []
        for freq, freq_pulses in frequency_groups.items():
            if len(freq_pulses) >= self.min_pulses_for_decode:
                messages = self._decode_frequency_group(freq, freq_pulses)
                all_messages.extend(messages)
            elif self.debug:
                print(f"Skipping {freq:.1f} Hz: only {len(freq_pulses)} pulses "
                      f"(min {self.min_pulses_for_decode})", file=sys.stderr)
        
        return all_messages
    
    def _group_by_frequency(self, pulses: List[Pulse]) -> Dict[float, List[Pulse]]:
        """
        Group pulses by frequency.
        
        Args:
            pulses: List of pulses
            
        Returns:
            Dictionary mapping frequency to list of pulses
        """
        groups = defaultdict(list)
        for pulse in pulses:
            # Round frequency to nearest Hz for grouping
            freq = round(pulse.frequency)
            groups[freq].append(pulse)
        
        # Sort pulses within each group by timestamp
        for freq in groups:
            groups[freq].sort(key=lambda p: p.timestamp)
        
        return groups
    
    def _decode_frequency_group(
        self, 
        frequency: float, 
        pulses: List[Pulse]
    ) -> List[DecodedMessage]:
        """
        Decode pulses at a specific frequency.
        
        Args:
            frequency: Center frequency
            pulses: Sorted list of pulses at this frequency
            
        Returns:
            List of decoded messages
        """
        if len(pulses) < self.min_pulses_for_decode:
            return []
        
        # Estimate morse parameters
        params = self._estimate_morse_parameters(pulses)
        
        if self.debug:
            print(f"\n{frequency:.1f} Hz: {len(pulses)} pulses", file=sys.stderr)
            print(f"  Timing: {params}", file=sys.stderr)
        
        # Classify pulses as dits or dahs
        symbols = self._classify_pulses(pulses, params)
        
        # Split into characters based on gaps
        characters = self._split_into_characters(pulses, symbols, params)
        
        # Decode characters
        messages = self._decode_characters(characters, frequency)
        
        return messages
    
    def _estimate_morse_parameters(self, pulses: List[Pulse]) -> MorseParameters:
        """
        Estimate morse timing parameters from pulses.
        
        Uses clustering to separate dits from dahs, and gap analysis for spaces.
        
        Args:
            pulses: List of pulses
            
        Returns:
            Estimated morse parameters
        """
        # Get pulse widths and gaps
        widths = np.array([p.width for p in pulses])
        
        # Compute gaps between pulses
        gaps = []
        for i in range(len(pulses) - 1):
            gap = pulses[i+1].timestamp - (pulses[i].timestamp + pulses[i].width)
            gaps.append(gap)
        gaps = np.array(gaps) if gaps else np.array([])
        
        # Cluster pulse widths into dits and dahs
        # Use simple threshold: assume dits are shorter than dahs
        sorted_widths = np.sort(widths)
        
        # Try to find natural break between dit and dah
        # Assume at least 1/3 of pulses are dits
        if len(sorted_widths) >= 3:
            # Find the largest gap in the sorted widths
            width_diffs = np.diff(sorted_widths)
            # Look for gap in middle third to half of distribution
            start_idx = len(width_diffs) // 3
            end_idx = len(width_diffs) * 2 // 3
            if start_idx < end_idx:
                search_region = width_diffs[start_idx:end_idx]
                if len(search_region) > 0:
                    max_gap_idx = start_idx + np.argmax(search_region)
                    threshold = (sorted_widths[max_gap_idx] + sorted_widths[max_gap_idx + 1]) / 2
                else:
                    threshold = np.median(widths)
            else:
                threshold = np.median(widths)
        else:
            threshold = np.median(widths)
        
        # Classify pulses
        dits = widths[widths < threshold]
        dahs = widths[widths >= threshold]
        
        # Calculate timing parameters
        if len(dits) > 0:
            dit_time = np.median(dits)
        else:
            dit_time = np.min(widths)
        
        if len(dahs) > 0:
            dah_time = np.median(dahs)
        else:
            dah_time = np.max(widths)
        
        # Analyze gaps to find space durations
        if len(gaps) > 0:
            sorted_gaps = np.sort(gaps)
            
            # Cluster gaps into groups using simple thresholding
            # Look for natural breaks in the gap distribution
            
            # Find the minimum gap (intra-symbol)
            intra_symbol_space = np.min(gaps)
            
            # Cluster gaps by finding significant jumps
            # We expect 3 clusters: intra-symbol, inter-char, inter-word
            gap_diffs = np.diff(sorted_gaps)
            
            # Find two largest jumps in gap sizes
            if len(gap_diffs) >= 2:
                # Get indices of largest jumps
                largest_jumps_idx = np.argsort(gap_diffs)[-2:]
                largest_jumps_idx = np.sort(largest_jumps_idx)
                
                # These divide the gaps into 3 clusters
                if len(largest_jumps_idx) == 2:
                    idx1, idx2 = largest_jumps_idx
                    
                    # Intra-symbol gaps (smallest cluster)
                    intra_gaps = sorted_gaps[:idx1+1]
                    # Inter-character gaps (middle cluster)
                    inter_gaps = sorted_gaps[idx1+1:idx2+1]
                    # Inter-word gaps (largest cluster)
                    word_gaps = sorted_gaps[idx2+1:]
                    
                    intra_symbol_space = np.median(intra_gaps) if len(intra_gaps) > 0 else np.min(gaps)
                    inter_symbol_space = np.median(inter_gaps) if len(inter_gaps) > 0 else np.percentile(gaps, 60)
                    inter_word_space = np.median(word_gaps) if len(word_gaps) > 0 else np.max(gaps)
                else:
                    # Fallback to percentiles
                    inter_symbol_space = np.percentile(gaps, 60)
                    inter_word_space = np.percentile(gaps, 95)
            else:
                # Not enough gaps, use simple percentiles
                inter_symbol_space = np.percentile(gaps, 60)
                inter_word_space = np.percentile(gaps, 90)
            
            # Ensure reasonable minimum separations
            if inter_symbol_space < intra_symbol_space * 2:
                inter_symbol_space = intra_symbol_space * 3
            if inter_word_space < inter_symbol_space * 1.5:
                inter_word_space = inter_symbol_space * 2.0
        else:
            # Default ratios based on dit time
            intra_symbol_space = dit_time
            inter_symbol_space = dit_time * 3
            inter_word_space = dit_time * 7
        
        return MorseParameters(
            dit_time=dit_time,
            dah_time=dah_time,
            intra_symbol_space=intra_symbol_space,
            inter_symbol_space=inter_symbol_space,
            inter_word_space=inter_word_space
        )
    
    def _classify_pulses(
        self, 
        pulses: List[Pulse], 
        params: MorseParameters
    ) -> List[str]:
        """
        Classify each pulse as dit or dah.
        
        Args:
            pulses: List of pulses
            params: Morse parameters
            
        Returns:
            List of symbols ('.' or '-')
        """
        threshold = (params.dit_time + params.dah_time) / 2
        symbols = []
        
        for pulse in pulses:
            if pulse.width < threshold:
                symbols.append('.')
            else:
                symbols.append('-')
        
        return symbols
    
    def _split_into_characters(
        self,
        pulses: List[Pulse],
        symbols: List[str],
        params: MorseParameters
    ) -> List[Tuple[float, str]]:
        """
        Split symbols into characters based on gaps.
        
        Args:
            pulses: List of pulses
            symbols: List of symbols ('.' or '-')
            params: Morse parameters
            
        Returns:
            List of (timestamp, character_pattern) tuples
        """
        if not pulses or not symbols:
            return []
        
        characters = []
        current_char = symbols[0]
        char_start_time = pulses[0].timestamp
        
        for i in range(len(pulses) - 1):
            gap = pulses[i+1].timestamp - (pulses[i].timestamp + pulses[i].width)
            
            # Determine if this gap ends a character
            # Use adaptive threshold between intra-symbol and inter-symbol space
            char_boundary_threshold = (params.intra_symbol_space + params.inter_symbol_space) / 2
            
            if gap >= char_boundary_threshold:
                # End of character
                characters.append((char_start_time, current_char))
                current_char = symbols[i+1]
                char_start_time = pulses[i+1].timestamp
                
                # Check for word boundary
                word_boundary_threshold = (params.inter_symbol_space + params.inter_word_space) / 2
                if gap >= word_boundary_threshold:
                    # Add a space marker
                    characters.append((pulses[i+1].timestamp - gap, ' '))
            else:
                # Continue current character
                current_char += symbols[i+1]
        
        # Add the last character
        if current_char:
            characters.append((char_start_time, current_char))
        
        return characters
    
    def _decode_characters(
        self,
        characters: List[Tuple[float, str]],
        frequency: float
    ) -> List[DecodedMessage]:
        """
        Decode morse characters into text.
        
        Args:
            characters: List of (timestamp, pattern) tuples
            frequency: Frequency of the signal
            
        Returns:
            List of decoded messages
        """
        if not characters:
            return []
        
        # Decode each character
        decoded_chars = []
        timestamps = []
        
        for timestamp, pattern in characters:
            if pattern == ' ':
                decoded_chars.append(' ')
            elif pattern in MORSE_CODE_DICT:
                decoded_chars.append(MORSE_CODE_DICT[pattern])
                timestamps.append(timestamp)
            else:
                # Unknown pattern
                if self.debug:
                    print(f"  Unknown pattern: {pattern}", file=sys.stderr)
                decoded_chars.append('?')
                timestamps.append(timestamp)
        
        # Combine into message(s)
        # For now, create one message with all decoded text
        text = ''.join(decoded_chars)
        
        # Clean up multiple spaces
        while '  ' in text:
            text = text.replace('  ', ' ')
        text = text.strip()
        
        if text:
            # Use timestamp of first character
            start_time = characters[0][0]
            message = DecodedMessage(
                timestamp=start_time,
                text=text,
                frequency=frequency
            )
            return [message]
        
        return []
    
    def _output_messages(self, messages: List[DecodedMessage], output_path: Optional[str] = None):
        """
        Output decoded messages in JSON lines format.
        
        Args:
            messages: List of decoded messages
            output_path: Path to output file (stdout if None)
        """
        if output_path:
            with open(output_path, 'w') as f:
                for message in messages:
                    json_obj = {
                        'timestamp': message.timestamp,
                        'text': message.text,
                        'frequency': message.frequency
                    }
                    f.write(json.dumps(json_obj) + '\n')
        else:
            for message in messages:
                json_obj = {
                    'timestamp': message.timestamp,
                    'text': message.text,
                    'frequency': message.frequency
                }
                print(json.dumps(json_obj))


def main():
    """Command line interface for morse decoder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Decode morse code from signal tracker JSON output'
    )
    parser.add_argument(
        'input',
        help='Input JSON lines file from signal tracker'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output JSON lines file (default: stdout)'
    )
    parser.add_argument(
        '--min-pulses',
        type=int,
        default=5,
        help='Minimum pulses required for decoding (default: 5)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Create decoder
    decoder = MorseDecoder(
        min_pulses_for_decode=args.min_pulses,
        debug=args.debug
    )
    
    # Decode file
    decoder.decode_from_file(args.input, args.output)


if __name__ == '__main__':
    main()

