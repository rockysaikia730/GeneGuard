"""
Fuzzer that inserts random out-of-alphabet characters into sequences.
Inserts approximately 10% out-of-alphabet characters.
"""
import random
import string


def insert_random_characters(sequence: str, insertion_rate: float = 0.10) -> str:
    """
    Insert random out-of-alphabet characters into a sequence.
    
    Args:
        sequence: Original DNA sequence
        insertion_rate: Proportion of out-of-alphabet characters to insert
    
    Returns:
        Sequence with random out-of-alphabet characters inserted
    """
    # Characters that are NOT in DNA alphabet (ACGT)
    out_of_alphabet = 'EFIJLMOPQRSUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:\'",.<>?/~`'
    
    sequence_list = list(sequence)
    
    # Calculate number of insertions
    num_insertions = int(len(sequence_list) * insertion_rate)
    
    # Get random positions to insert (allowing same position for multiple insertions)
    for _ in range(num_insertions):
        # Random position
        pos = random.randint(0, len(sequence_list))
        # Random out-of-alphabet character
        char = random.choice(out_of_alphabet)
        sequence_list.insert(pos, char)
    
    return ''.join(sequence_list)


def fuzz_random_character_insertion(sequence: str, compressed: str, text_with_dna: str, text_without_dna: str) -> dict:
    """
    Apply random character insertion fuzzing to a data entry.
    
    Args:
        sequence: Original DNA sequence
        compressed: Compressed sequence
        text_with_dna: Text with embedded DNA
        text_without_dna: Text without DNA
    
    Returns:
        Dictionary with fuzzed data
    """
    # Fuzz the sequence
    fuzzed_sequence = insert_random_characters(sequence)
    
    # Fuzz compressed version
    fuzzed_compressed = insert_random_characters(compressed)
    
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

