"""
Fuzzer that formats sequences with line numbers and line breaks.
Format: 1 actgactgactg\n61 actgactgactg\n...
"""
import re


def numerical_interleave(sequence: str, line_length: int = 60) -> str:
    """
    Format a sequence with line numbers and line breaks.
    
    Args:
        sequence: Original DNA sequence
        line_length: Number of characters per line (before adding line number)
    
    Returns:
        Formatted sequence with line numbers
    """
    if not sequence:
        return "1 "
    
    lines = []
    line_number = 1
    
    # Split sequence into chunks
    for i in range(0, len(sequence), line_length):
        chunk = sequence[i:i + line_length]
        lines.append(f"{line_number} {chunk}")
        line_number += line_length
    
    return '\n'.join(lines)


def fuzz_numerical_interleaving(sequence: str, compressed: str, text_with_dna: str, text_without_dna: str) -> dict:
    """
    Apply numerical interleaving fuzzing to a data entry.
    
    Args:
        sequence: Original DNA sequence
        compressed: Compressed sequence
        text_with_dna: Text with embedded DNA
        text_without_dna: Text without DNA
    
    Returns:
        Dictionary with fuzzed data
    """
    # Fuzz the sequence
    fuzzed_sequence = numerical_interleave(sequence)
    
    # Fuzz compressed version
    fuzzed_compressed = numerical_interleave(compressed)
    
    # Update text_with_dna - replace sequences with fuzzed versions
    # Replace compressed version first (it's more likely to be in the text)
    fuzzed_text_with_dna = text_with_dna.replace(compressed, fuzzed_compressed)
    # Also replace full sequence if present
    fuzzed_text_with_dna = fuzzed_text_with_dna.replace(sequence, fuzzed_sequence)
    
    return {
        'sequence': fuzzed_sequence,
        'compressed': fuzzed_compressed,
        'text_with_dna': fuzzed_text_with_dna,
        'text_without_dna': text_without_dna  # This doesn't change
    }

