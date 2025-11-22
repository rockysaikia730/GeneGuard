"""
Fuzzer that performs a full cipher-like character substitution mapping.
"""
import random
import string


def create_substitution_map(dna_alphabet: str = 'ACGTNacgtn') -> dict:
    """
    Create a random substitution mapping for DNA characters.
    
    Args:
        dna_alphabet: Characters to create mapping for
    
    Returns:
        Dictionary mapping original characters to substituted characters
    """
    # Create a shuffled list of characters to map to
    # Use extended alphabet including ambiguity codes and other chars
    target_chars = list('ACGTNRYSWKMBDHVacgtnryswkmbdhv0123456789!@#$%^&*()_+-=[]{}|;:\'",.<>?/~`')
    
    # Shuffle to create random mapping
    random.shuffle(target_chars)
    
    # Create mapping
    substitution_map = {}
    for i, char in enumerate(dna_alphabet):
        substitution_map[char] = target_chars[i % len(target_chars)]
    
    return substitution_map


def substitute_characters(sequence: str, substitution_map: dict) -> str:
    """
    Apply character substitution to a sequence using a mapping.
    
    Args:
        sequence: Original DNA sequence
        substitution_map: Dictionary mapping original to substituted characters
    
    Returns:
        Sequence with substituted characters
    """
    return ''.join(substitution_map.get(char, char) for char in sequence)


def fuzz_character_substitution(sequence: str, compressed: str, text_with_dna: str, text_without_dna: str) -> dict:
    """
    Apply character substitution fuzzing to a data entry.
    
    Args:
        sequence: Original DNA sequence
        compressed: Compressed sequence
        text_with_dna: Text with embedded DNA
        text_without_dna: Text without DNA
    
    Returns:
        Dictionary with fuzzed data
    """
    # Create substitution mapping based on all unique characters in both sequences
    all_chars = set(sequence + compressed)
    dna_alphabet = ''.join(sorted(all_chars))
    
    # Create mapping
    substitution_map = create_substitution_map(dna_alphabet)
    
    # Fuzz the sequence
    fuzzed_sequence = substitute_characters(sequence, substitution_map)
    
    # Fuzz compressed version (using same mapping for consistency)
    fuzzed_compressed = substitute_characters(compressed, substitution_map)
    
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

