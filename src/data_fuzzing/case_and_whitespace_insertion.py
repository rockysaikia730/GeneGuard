"""
Fuzzer that mixes case and inserts whitespace, tabs, and line breaks into sequences.
"""
import random
import re


def mix_case_and_whitespace(sequence: str, whitespace_rate: float = 0.05) -> str:
    """
    Mix case and insert whitespace characters into a sequence.
    
    Args:
        sequence: Original DNA sequence
        whitespace_rate: Probability of inserting whitespace at each position
    
    Returns:
        Sequence with mixed case and whitespace inserted
    """
    result = []
    
    for char in sequence:
        # Randomly decide if this character should be lowercase or uppercase
        if char.isalpha():
            char = random.choice([char.upper(), char.lower()])
        
        result.append(char)
        
        # Randomly insert whitespace, tab, or newline
        if random.random() < whitespace_rate:
            whitespace_type = random.choice([' ', '\t', '\n'])
            result.append(whitespace_type)
    
    return ''.join(result)


def fuzz_case_and_whitespace(sequence: str, compressed: str, text_with_dna: str, text_without_dna: str) -> dict:
    """
    Apply case and whitespace fuzzing to a data entry.
    
    Args:
        sequence: Original DNA sequence
        compressed: Compressed sequence
        text_with_dna: Text with embedded DNA
        text_without_dna: Text without DNA
    
    Returns:
        Dictionary with fuzzed data
    """
    # Fuzz the sequence
    fuzzed_sequence = mix_case_and_whitespace(sequence)
    
    # Fuzz compressed version
    fuzzed_compressed = mix_case_and_whitespace(compressed)
    
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

