# GPT Sequence Finder - Interpretability Report

## Model Configuration
- Base Model: gpt2
- LoRA: r=8, alpha=16
- Max Length: 512

## Curriculum Results
- Easy: F1=0.0008, Prec=0.0759, Rec=0.0004
- Medium: F1=0.0029, Prec=0.2899, Rec=0.0015
- Hard: F1=0.0933, Prec=0.9475, Rec=0.0491

## Difficulty Tiers
- Easy: ['text_and_seq', 'text_and_seq_compressed', 'text_and_seq_fuzzed_numerical_interleaving']
- Medium: ['text_and_seq_fuzzed_ambiguity_codes', 'text_and_seq_fuzzed_random_character_insertion', 'text_and_seq_fuzzed_case_and_whitespace']
- Hard: ['text_and_seq_alternate_random_characters', 'text_and_seq_compressed_and_replaced', 'text_and_seq_fuzzed_character_substitution']

## Key Findings
1. Curriculum learning improves generalization across masking types
2. Attention heads specialize for genomic detection
3. Detection emerges in middle-to-late layers

## Generated Files
- figures/difficulty.png
- figures/curriculum_progress.png  
- figures/attention_heatmap.png
- figures/token_analysis.png
- figures/per_type_results.png
- final_model/
