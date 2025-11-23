import math
from collections import Counter

# O(n) time complexity
def check(text: str) -> bool:
    """
    Check if text contains a DNA sequence.
    
    1. Remove all whitespaces, tabs, and newlines and make everything lowercase
    2. Makes sure the sequence is long enough to be dangerous (100 bp+)
    3. Makes sure 90% of characters are from IUPAC DNA alphabet
    4. Makes sure the sequence has enough entropy (0.5)

    This does not account for character substitutions because it's not a huge problem in non adversarial settings
    and it might lead to false positives for flow cytometry and plate reader data.
    """
    IUPAC_DNA = set('acgtnryswkmbdhv')
    
    cleaned = ''.join(c.lower() for c in text if not c.isspace())
    
    if len(cleaned) < 100:
        return False
    
    window_size = 100
    dna_count = 0
    
    char_counts = {}
    entropy = 0.0
    
    for i in range(window_size):
        char = cleaned[i]
        if char in IUPAC_DNA:
            dna_count += 1
        
        char_counts[char] = char_counts.get(char, 0) + 1
    
    length = window_size
    for count in char_counts.values():
        if count > 0:
            prob = count / length
            entropy -= prob * math.log2(prob)
    
    if dna_count >= window_size * 0.9 and entropy >= 0.5:
        return True
    
    for i in range(window_size, len(cleaned)):
        left_char = cleaned[i - window_size]
        if left_char in IUPAC_DNA:
            dna_count -= 1
        
        old_count = char_counts[left_char]
        old_prob = old_count / length
        if old_prob > 0:
            entropy += old_prob * math.log2(old_prob)
        
        char_counts[left_char] -= 1
        if char_counts[left_char] == 0:
            del char_counts[left_char]
        else:
            new_prob = char_counts[left_char] / length
            if new_prob > 0:
                entropy -= new_prob * math.log2(new_prob)
        
        right_char = cleaned[i]
        if right_char in IUPAC_DNA:
            dna_count += 1
        
        if right_char in char_counts:
            old_count = char_counts[right_char]
            if old_count > 0:
                old_prob = old_count / length
                if old_prob > 0:
                    entropy += old_prob * math.log2(old_prob)
        
        char_counts[right_char] = char_counts.get(right_char, 0) + 1
        new_prob = char_counts[right_char] / length
        if new_prob > 0:
            entropy -= new_prob * math.log2(new_prob)
        
        if dna_count >= window_size * 0.9 and entropy >= 0.5:
            return True
    
    return False
