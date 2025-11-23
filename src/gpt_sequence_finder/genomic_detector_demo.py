"""
Genomic Sequence Detector - Inference Demo
==========================================
Loads a pre-trained GPT-2 model fine-tuned for genomic sequence detection
and provides inference functionality for demonstration purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2PreTrainedModel, GPT2TokenizerFast
from transformers.modeling_outputs import TokenClassifierOutput
from peft import PeftModel
import numpy as np
from pathlib import Path
import argparse
import json


class GPT2ForTokenClassification(GPT2PreTrainedModel):
    """GPT-2 model for token classification (genomic sequence detection)."""
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            active = labels.view(-1) != -100
            if active.any():
                loss = loss_fn(logits.view(-1, self.num_labels)[active], 
                              labels.view(-1)[active])
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return TokenClassifierOutput(
            loss=loss, 
            logits=logits,
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
        )


class GenomicDetector:
    """Genomic sequence detector using fine-tuned GPT-2."""
    
    def __init__(self, checkpoint_path, model_name='gpt2', device=None):
        """
        Initialize the detector.
        
        Args:
            checkpoint_path: Path to trained model checkpoint (after_hard)
            model_name: Base model name (default: 'gpt2')
            device: Device to run on (auto-detect if None)
        """
        self.model_name = model_name
        self.checkpoint_path = Path(checkpoint_path)
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading model on {self.device}...")
        self._load_model()
        print("✓ Model loaded successfully")
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        # Load tokenizer
        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load config with attentions enabled
        config = GPT2Config.from_pretrained(self.model_name)
        config.num_labels = 2
        config.output_attentions = True
        
        # Load base model
        base_model = GPT2ForTokenClassification.from_pretrained(
            self.model_name, 
            config=config, 
            ignore_mismatched_sizes=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
    
    def detect(self, text, threshold=0.5, max_length=512):
        """
        Detect genomic sequences in text.
        
        Args:
            text: Input text potentially containing genomic sequences
            threshold: Probability threshold for detection (default: 0.5)
            max_length: Maximum sequence length (default: 512)
        
        Returns:
            Dictionary with detection results:
            - regions: List of (start, end) character positions
            - sequences: List of detected genomic sequences
            - probabilities: Token-level probabilities
            - confidence: Overall confidence score
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offsets = encoding['offset_mapping'][0].tolist()
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
        
        # Get predictions
        logits = outputs.logits[0].cpu()
        probs = F.softmax(logits, dim=-1)[:, 1].numpy()  # Prob of genomic class
        preds = (probs >= threshold).astype(int)
        
        # Map predictions to character positions
        genomic_chars = []
        for i, (start, end) in enumerate(offsets):
            if start != end and preds[i] == 1:
                genomic_chars.extend(range(start, end))
        
        # Find contiguous regions
        regions = []
        sequences = []
        
        if genomic_chars:
            genomic_chars = sorted(set(genomic_chars))
            region_start = region_end = genomic_chars[0]
            
            for char_pos in genomic_chars[1:]:
                if char_pos == region_end + 1:
                    region_end = char_pos
                else:
                    # Save completed region
                    regions.append((region_start, region_end + 1))
                    sequences.append(text[region_start:region_end + 1])
                    region_start = region_end = char_pos
            
            # Save last region
            regions.append((region_start, region_end + 1))
            sequences.append(text[region_start:region_end + 1])
        
        # Calculate overall confidence
        if len(preds) > 0:
            genomic_probs = probs[preds == 1]
            confidence = genomic_probs.mean() if len(genomic_probs) > 0 else 0.0
        else:
            confidence = 0.0
        
        return {
            'regions': regions,
            'sequences': sequences,
            'probabilities': probs.tolist(),
            'confidence': float(confidence),
            'n_regions': len(regions),
            'total_genomic_chars': len(genomic_chars) if genomic_chars else 0
        }
    
    def detect_batch(self, texts, threshold=0.5, max_length=512):
        """
        Detect genomic sequences in multiple texts.
        
        Args:
            texts: List of input texts
            threshold: Probability threshold for detection
            max_length: Maximum sequence length
        
        Returns:
            List of detection results (one per input text)
        """
        return [self.detect(text, threshold, max_length) for text in texts]
    
    def format_result(self, text, result, show_probabilities=False):
        """
        Format detection result for display.
        
        Args:
            text: Original input text
            result: Detection result from detect()
            show_probabilities: Include token probabilities (default: False)
        
        Returns:
            Formatted string
        """
        output = []
        output.append("="*80)
        output.append("GENOMIC SEQUENCE DETECTION RESULT")
        output.append("="*80)
        
        if result['n_regions'] == 0:
            output.append("\n✗ No genomic sequences detected")
        else:
            output.append(f"\n✓ Detected {result['n_regions']} genomic region(s)")
            output.append(f"  Overall confidence: {result['confidence']:.2%}")
            output.append(f"  Total genomic characters: {result['total_genomic_chars']}")
            
            for i, (start, end) in enumerate(result['regions'], 1):
                length = end - start
                sequence = result['sequences'][i-1]
                
                output.append(f"\n  Region {i}:")
                output.append(f"    Position: {start}-{end} (length: {length})")
                
                # Show sequence (truncate if long)
                if len(sequence) > 100:
                    seq_display = sequence[:100] + f"... ({len(sequence)-100} more chars)"
                else:
                    seq_display = sequence
                output.append(f"    Sequence: {seq_display}")
        
        output.append("\n" + "="*80)
        
        return "\n".join(output)


def demo_samples():
    """Demo samples for testing."""
    return [
        {
            'name': 'Clean ATGC',
            'text': 'The experiment showed interesting results. The sequence ATGCGATCGATCGATCG was identified. Further analysis pending.'
        },
        {
            'name': 'Compressed notation',
            'text': 'Analysis report: sequence found as TGCAT5ACGC2ATC2GCGC2AG2CGCGATCAGACGAGCT in sample #47.'
        },
        {
            'name': 'Mixed case with whitespace',
            'text': 'Data stream contains: aTgC gAtC gAtC gAtC embedded in the output log.'
        },
        {
            'name': 'No genomic content',
            'text': 'This is a normal document with no special sequences or patterns of interest.'
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='Genomic Sequence Detector - Inference Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (e.g., ./outputs/checkpoints/after_hard)')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Base model name (default: gpt2)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu, auto-detect if not specified)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold (default: 0.5)')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to analyze (if not provided, runs demo samples)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = GenomicDetector(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=args.device
    )
    
    if args.interactive:
        # Interactive mode
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Enter text to analyze (or 'quit' to exit):")
        
        while True:
            text = input("\n> ")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            result = detector.detect(text, threshold=args.threshold)
            print(detector.format_result(text, result))
    
    elif args.text:
        # Single text analysis
        result = detector.detect(args.text, threshold=args.threshold)
        print(detector.format_result(args.text, result))
    
    else:
        # Run demo samples
        print("\n" + "="*80)
        print("RUNNING DEMO SAMPLES")
        print("="*80)
        
        samples = demo_samples()
        
        for sample in samples:
            print(f"\n{'='*80}")
            print(f"Sample: {sample['name']}")
            print(f"{'='*80}")
            print(f"Input: {sample['text']}\n")
            
            result = detector.detect(sample['text'], threshold=args.threshold)
            print(detector.format_result(sample['text'], result))


if __name__ == '__main__':
    main()

### HOW TO USE:

# RUN PRE-DEFINED SAMPLES: python genomic_detector_demo.py --checkpoint ./outputs/checkpoints/after_hard``
# RUN CUSTOM GENOMIC SEQUENCE: python genomic_detector_demo.py \
    #--checkpoint ./outputs/checkpoints/after_hard \
    #--text "The sequence ATGCGATCG was found in sample 42."
# INTERACTIVE MODE: python genomic_detector_demo.py \
    #--checkpoint ./outputs/checkpoints/after_hard \
    #--interactive
# Adjust Threshold: python genomic_detector_demo.py \
    #--checkpoint ./outputs/checkpoints/after_hard \
    #--threshold 0.7 \
    #--text "Your text here"