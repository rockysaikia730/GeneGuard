"""
Main script to run all data fuzzers on files from text_and_seq_compressed directory.
"""
import os
import pandas as pd
import argparse
from pathlib import Path

from ambiguity_codes import fuzz_ambiguity_codes
from case_and_whitespace_insertion import fuzz_case_and_whitespace
from numerical_interleaving import fuzz_numerical_interleaving
from random_character_insertion import fuzz_random_character_insertion
from character_substitution import fuzz_character_substitution


def process_directory(input_dir: str, output_dir: str, fuzzer_func, fuzzer_name: str):
    """
    Process all CSV files in input directory using the specified fuzzer.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        fuzzer_func: Function to apply fuzzing
        fuzzer_name: Name of the fuzzer (for logging)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')])
    
    print(f"\n{fuzzer_name}: Processing {len(csv_files)} files...")
    
    for csv_file in csv_files:
        input_path = os.path.join(input_dir, csv_file)
        output_path = os.path.join(output_dir, csv_file)
        
        try:
            # Read input CSV
            df = pd.read_csv(input_path)
            
            # Process each row
            fuzzed_rows = []
            for idx, row in df.iterrows():
                fuzzed_data = fuzzer_func(
                    sequence=str(row['sequence']),
                    compressed=str(row['compressed']),
                    text_with_dna=str(row['text_with_dna']),
                    text_without_dna=str(row['text_without_dna'])
                )
                fuzzed_rows.append(fuzzed_data)
            
            # Create new dataframe
            fuzzed_df = pd.DataFrame(fuzzed_rows)
            
            # Write to output
            fuzzed_df.to_csv(output_path, index=False)
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
    
    print(f"{fuzzer_name}: Completed processing {len(csv_files)} files")


def main():
    """Run all fuzzers on the input directory."""
    parser = argparse.ArgumentParser(description='Run data fuzzers on CSV files')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='text_and_seq_compressed',
        help='Input directory containing CSV files (default: text_and_seq_compressed)'
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='.',
        help='Base directory for output (default: current directory)'
    )
    parser.add_argument(
        '--fuzzer',
        type=str,
        choices=['all', 'ambiguity', 'case', 'numerical', 'random', 'substitution'],
        default='all',
        help='Which fuzzer to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    input_dir = os.path.abspath(args.input_dir)
    output_base = os.path.abspath(args.output_base)
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist")
        return
    
    # Define fuzzers
    fuzzers = []
    
    if args.fuzzer in ['all', 'ambiguity']:
        fuzzers.append((
            fuzz_ambiguity_codes,
            'ambiguity_codes',
            os.path.join(output_base, 'text_and_seq_fuzzed_ambiguity_codes')
        ))
    
    if args.fuzzer in ['all', 'case']:
        fuzzers.append((
            fuzz_case_and_whitespace,
            'case_and_whitespace',
            os.path.join(output_base, 'text_and_seq_fuzzed_case_and_whitespace')
        ))
    
    if args.fuzzer in ['all', 'numerical']:
        fuzzers.append((
            fuzz_numerical_interleaving,
            'numerical_interleaving',
            os.path.join(output_base, 'text_and_seq_fuzzed_numerical_interleaving')
        ))
    
    if args.fuzzer in ['all', 'random']:
        fuzzers.append((
            fuzz_random_character_insertion,
            'random_character_insertion',
            os.path.join(output_base, 'text_and_seq_fuzzed_random_character_insertion')
        ))
    
    if args.fuzzer in ['all', 'substitution']:
        fuzzers.append((
            fuzz_character_substitution,
            'character_substitution',
            os.path.join(output_base, 'text_and_seq_fuzzed_character_substitution')
        ))
    
    # Run fuzzers
    print(f"Processing files from: {input_dir}")
    print(f"Output base directory: {output_base}")
    print(f"Running {len(fuzzers)} fuzzer(s)...")
    
    for fuzzer_func, fuzzer_name, output_dir in fuzzers:
        process_directory(input_dir, output_dir, fuzzer_func, fuzzer_name)
    
    print("\n✅ All fuzzers completed!")


if __name__ == '__main__':
    main()

