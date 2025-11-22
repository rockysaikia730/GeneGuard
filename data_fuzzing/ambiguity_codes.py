"""
Fuzzer that inserts IUPAC ambiguity codes into DNA sequences.
IUPAC codes: R, Y, S, W, K, M, B, D, H, V, N
"""
import random
import re


def insert_ambiguity_codes(sequence: str, insertion_rate: float = 0.15) -> str:
    """
    Insert IUPAC ambiguity codes into a DNA sequence.
    
    Args:
        sequence: Original DNA sequence (ACGT only)
        insertion_rate: Proportion of positions to replace with ambiguity codes
    
    Returns:
        Sequence with IUPAC ambiguity codes inserted
    """
    iupac_codes = ['R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N']
    sequence_list = list(sequence)
    
    # Count how many positions to modify
    num_positions = int(len(sequence_list) * insertion_rate)
    
    # Get random positions (ensure we don't pick the same position twice)
    positions = random.sample(range(len(sequence_list)), min(num_positions, len(sequence_list)))
    
    # Replace characters at selected positions with IUPAC codes
    for pos in positions:
        sequence_list[pos] = random.choice(iupac_codes)
    
    return ''.join(sequence_list)


def fuzz_ambiguity_codes(sequence: str, compressed: str, text_with_dna: str, text_without_dna: str) -> dict:
    """
    Apply ambiguity code fuzzing to a data entry.
    
    Args:
        sequence: Original DNA sequence
        compressed: Compressed sequence
        text_with_dna: Text with embedded DNA
        text_without_dna: Text without DNA
    
    Returns:
        Dictionary with fuzzed data
    """
    # Fuzz the sequence
    fuzzed_sequence = insert_ambiguity_codes(sequence)
    
    # Fuzz compressed version as well
    fuzzed_compressed = insert_ambiguity_codes(compressed, insertion_rate=0.15)
    
    # Update text_with_dna to reflect the new sequences
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

