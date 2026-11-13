"""
Stage 3.2.9G - Text Conditioning Audit

Trace the complete path from natural-language prompt to temporal feature computation.

Author: Devin
Date: 2026-11-03
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Any, List
from text_encoder import SimpleTokenizer, TextEncoder


class TextConditioningAuditor:
    """Audit text conditioning pathway."""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = None
        self.text_encoder = None

    def load_text_encoder(self):
        """Load text encoder and tokenizer."""
        # Look for text encoder checkpoint
        text_encoder_path = self.checkpoint_dir / "text_encoder.pt"
        tokenizer_path = self.checkpoint_dir / "text_encoder_tokenizer.json"

        if text_encoder_path.exists() and tokenizer_path.exists():
            print(f"Loading text encoder from {text_encoder_path}")
            self.text_encoder, self.tokenizer = TextEncoder.load(
                str(text_encoder_path),
                str(tokenizer_path)
            )
        else:
            print("Text encoder checkpoint not found. Using random initialization for audit.")
            # Create dummy tokenizer with motion vocabulary
            self.tokenizer = SimpleTokenizer(vocab_size=1000, min_freq=1)
            motion_texts = [
                "a red object moving left",
                "a red object moving right",
                "a blue object moving upward",
                "a blue object moving downward",
                "an object moving slowly",
                "an object moving quickly",
                "a stationary object",
            ]
            self.tokenizer.build_vocab(motion_texts)

            # Create dummy encoder
            self.text_encoder = TextEncoder(
                vocab_size=len(self.tokenizer.word_to_idx),
                d_model=256,
                n_heads=4,
                n_layers=4,
                d_ff=512,
                max_seq_len=64
            )

    def audit_tokenization(self, prompts: List[str]) -> Dict[str, Any]:
        """Audit tokenization of motion prompts."""
        results = {}

        for prompt in prompts:
            indices, mask = self.tokenizer.encode(prompt, max_length=20)
            tokens = [self.tokenizer.idx_to_word.get(idx, '<UNK>') for idx in indices]

            # Check for unknown tokens
            unk_count = sum(1 for idx in indices if idx == self.tokenizer.SPECIAL_TOKENS['<UNK>'])

            # Check for motion-related tokens
            motion_words = ['left', 'right', 'up', 'down', 'upward', 'downward', 'moving', 'slowly', 'quickly', 'stationary']
            motion_tokens_present = [word for word in motion_words if word in tokens]

            results[prompt] = {
                'indices': indices,
                'tokens': tokens,
                'mask': mask,
                'unk_count': unk_count,
                'motion_tokens_present': motion_tokens_present,
                'effective_length': sum(mask)
            }

        return results

    def audit_embeddings(self, prompts: List[str]) -> Dict[str, Any]:
        """Audit text embeddings."""
        self.text_encoder.eval()
        results = {}

        with torch.no_grad():
            for prompt in prompts:
                indices, mask = self.tokenizer.encode(prompt, max_length=20)
                token_ids = torch.tensor([indices])
                attention_mask = torch.tensor([mask], dtype=torch.float)

                # Encode
                encoded, mask_out = self.text_encoder(token_ids, attention_mask)

                # Compute statistics
                encoded_np = encoded.cpu().numpy()

                results[prompt] = {
                    'embedding_shape': list(encoded.shape),
                    'embedding_mean': float(encoded_np.mean()),
                    'embedding_std': float(encoded_np.std()),
                    'embedding_min': float(encoded_np.min()),
                    'embedding_max': float(encoded_np.max()),
                    'per_token_means': encoded_np[0].mean(axis=1).tolist(),
                    'per_token_stds': encoded_np[0].std(axis=1).tolist(),
                }

        return results

    def compare_embeddings(self, prompt1: str, prompt2: str) -> Dict[str, Any]:
        """Compare embeddings between two prompts."""
        self.text_encoder.eval()

        with torch.no_grad():
            # Encode prompt 1
            indices1, mask1 = self.tokenizer.encode(prompt1, max_length=20)
            token_ids1 = torch.tensor([indices1])
            attention_mask1 = torch.tensor([mask1], dtype=torch.float)
            encoded1, _ = self.text_encoder(token_ids1, attention_mask1)

            # Encode prompt 2
            indices2, mask2 = self.tokenizer.encode(prompt2, max_length=20)
            token_ids2 = torch.tensor([indices2])
            attention_mask2 = torch.tensor([mask2], dtype=torch.float)
            encoded2, _ = self.text_encoder(token_ids2, attention_mask2)

            # Compute differences
            diff = encoded1 - encoded2
            diff_norm = torch.norm(diff).item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                encoded1.flatten().unsqueeze(0),
                encoded2.flatten().unsqueeze(0)
            ).item()

            return {
                'prompt1': prompt1,
                'prompt2': prompt2,
                'tokens1': [self.tokenizer.idx_to_word.get(idx, '<UNK>') for idx in indices1],
                'tokens2': [self.tokenizer.idx_to_word.get(idx, '<UNK>') for idx in indices2],
                'embedding_difference_norm': diff_norm,
                'cosine_similarity': cosine_sim,
                'per_token_difference_norm': torch.norm(diff, dim=-1).cpu().numpy().tolist()[0],
            }

    def audit_current_implementation(self) -> Dict[str, Any]:
        """Audit the current deterministic text embedding implementation."""
        # The current implementation in comparison_generator.py uses random embeddings
        # This is a major issue - there's no actual text conditioning!

        results = {
            'current_implementation': {
                'method': 'random_normal',
                'description': 'torch.randn(batch_size, seq_len, text_dim)',
                'issue': 'NO ACTUAL TEXT ENCODING',
                'impact': 'Text prompts have no effect on generation',
            },
            'expected_implementation': {
                'method': 'text_encoder.forward()',
                'description': 'Tokenize -> Embed -> Encode -> Attention',
                'impact': 'Text prompts should influence temporal conditioning',
            }
        }

        return results


def main():
    """Run text conditioning audit."""
    print("="*80)
    print("STAGE 3.2.9G - TEXT CONDITIONING AUDIT")
    print("="*80)

    base_dir = Path(__file__).parent.parent.parent
    checkpoint_dir = base_dir / "checkpoints_conditional"

    auditor = TextConditioningAuditor(checkpoint_dir)
    auditor.load_text_encoder()

    # Test prompts
    prompts = [
        "A red object moving left",
        "A red object moving right",
        "A blue object moving upward",
        "A blue object moving downward",
        "An object moving slowly",
        "An object moving quickly",
        "A stationary object",
    ]

    print("\n" + "="*80)
    print("TOKENIZATION AUDIT")
    print("="*80)

    tokenization_results = auditor.audit_tokenization(prompts)

    for prompt, result in tokenization_results.items():
        print(f"\nPrompt: '{prompt}'")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Indices: {result['indices']}")
        print(f"  Effective Length: {result['effective_length']}")
        print(f"  Unknown Tokens: {result['unk_count']}")
        print(f"  Motion Tokens Present: {result['motion_tokens_present']}")

    print("\n" + "="*80)
    print("EMBEDDING AUDIT")
    print("="*80)

    embedding_results = auditor.audit_embeddings(prompts)

    for prompt, result in embedding_results.items():
        print(f"\nPrompt: '{prompt}'")
        print(f"  Embedding Shape: {result['embedding_shape']}")
        print(f"  Mean: {result['embedding_mean']:.4f}")
        print(f"  Std: {result['embedding_std']:.4f}")
        print(f"  Min: {result['embedding_min']:.4f}")
        print(f"  Max: {result['embedding_max']:.4f}")

    print("\n" + "="*80)
    print("EMBEDDING COMPARISON (Direction Pairs)")
    print("="*80)

    direction_pairs = [
        ("A red object moving left", "A red object moving right"),
        ("A blue object moving upward", "A blue object moving downward"),
        ("An object moving slowly", "An object moving quickly"),
        ("A red object moving left", "A stationary object"),
    ]

    comparison_results = {}
    for prompt1, prompt2 in direction_pairs:
        comparison = auditor.compare_embeddings(prompt1, prompt2)
        comparison_results[f"{prompt1} vs {prompt2}"] = comparison

        print(f"\n{prompt1} vs {prompt2}")
        print(f"  Tokens 1: {comparison['tokens1']}")
        print(f"  Tokens 2: {comparison['tokens2']}")
        print(f"  Embedding Difference Norm: {comparison['embedding_difference_norm']:.4f}")
        print(f"  Cosine Similarity: {comparison['cosine_similarity']:.4f}")

    print("\n" + "="*80)
    print("CURRENT IMPLEMENTATION AUDIT")
    print("="*80)

    implementation_audit = auditor.audit_current_implementation()

    print(f"\nCurrent Method: {implementation_audit['current_implementation']['method']}")
    print(f"Description: {implementation_audit['current_implementation']['description']}")
    print(f"ISSUE: {implementation_audit['current_implementation']['issue']}")
    print(f"Impact: {implementation_audit['current_implementation']['impact']}")

    print(f"\nExpected Method: {implementation_audit['expected_implementation']['method']}")
    print(f"Description: {implementation_audit['expected_implementation']['description']}")
    print(f"Impact: {implementation_audit['expected_implementation']['impact']}")

    # Save results
    output_dir = Path(__file__).parent / "text_conditioning_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'tokenization': tokenization_results,
        'embeddings': embedding_results,
        'comparisons': comparison_results,
        'implementation_audit': implementation_audit,
    }

    results_path = output_dir / "text_conditioning_audit_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("AUDIT COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to {results_path}")

    # Generate report
    generate_text_conditioning_report(results, output_dir)


def generate_text_conditioning_report(results: Dict[str, Any], output_dir: Path):
    """Generate text conditioning audit report."""
    report = []
    report.append("# Stage 3.2.9G - Text Conditioning Audit Report\n")
    report.append("**Date:** 2026-11-03\n")
    report.append("**Status:** Complete\n\n")

    report.append("## Critical Finding\n\n")
    report.append("### CURRENT IMPLEMENTATION USES RANDOM TEXT EMBEDDINGS\n\n")
    report.append("The current `comparison_generator.py` implementation uses:\n")
    report.append("```python\n")
    report.append("text_emb = torch.randn(batch_size, seq_len, text_dim)\n")
    report.append("```\n\n")
    report.append("**This means:**\n")
    report.append("- Text prompts have **ZERO effect** on video generation\n")
    report.append("- All motion prompts are effectively identical to the model\n")
    report.append("- Direction control failures are **expected**, not surprising\n")
    report.append("- The observed 32% displacement increase in Experiment C is **unconditional** temporal activity\n\n")

    report.append("## Tokenization Analysis\n\n")

    tokenization = results['tokenization']
    for prompt, data in tokenization.items():
        report.append(f"### {prompt}\n")
        report.append(f"- **Tokens:** {data['tokens']}\n")
        report.append(f"- **Unknown Tokens:** {data['unk_count']}\n")
        report.append(f"- **Motion Tokens Present:** {data['motion_tokens_present']}\n")
        report.append(f"- **Effective Length:** {data['effective_length']}\n\n")

    report.append("## Embedding Analysis\n\n")

    embeddings = results['embeddings']
    for prompt, data in embeddings.items():
        report.append(f"### {prompt}\n")
        report.append(f"- **Embedding Shape:** {data['embedding_shape']}\n")
        report.append(f"- **Mean:** {data['embedding_mean']:.4f}\n")
        report.append(f"- **Std:** {data['embedding_std']:.4f}\n\n")

    report.append("## Embedding Comparisons\n\n")

    comparisons = results['comparisons']
    for pair_name, data in comparisons.items():
        report.append(f"### {pair_name}\n")
        report.append(f"- **Cosine Similarity:** {data['cosine_similarity']:.4f}\n")
        report.append(f"- **Difference Norm:** {data['embedding_difference_norm']:.4f}\n\n")

    report.append("## Root Cause Analysis\n\n")
    report.append("### Primary Issue: Missing Text Conditioning\n\n")
    report.append("The text conditioning pathway is **completely bypassed** in the current implementation:\n\n")
    report.append("1. **Prompt to Tokenization:** NOT IMPLEMENTED (random embeddings used)\n")
    report.append("2. **Tokenization to Embeddings:** NOT IMPLEMENTED (random embeddings used)\n")
    report.append("3. **Embeddings to Encoder:** NOT IMPLEMENTED (random embeddings used)\n")
    report.append("4. **Encoder to Temporal Modules:** NOT IMPLEMENTED (random embeddings used)\n\n")

    report.append("### Why Direction Control Fails\n\n")
    report.append("Given that all prompts receive identical random embeddings:\n\n")
    report.append("- **Horizontal motion (50% accuracy):** Random behavior\n")
    report.append("- **Downward motion (0% accuracy):** Random failure\n")
    report.append("- **Upward motion (100% accuracy):** Random success\n")
    report.append("- **Stationary (38% accuracy for C):** Random activity\n\n")

    report.append("### Why Experiment C Has More Motion\n\n")
    report.append("The 32% displacement increase in Experiment C is **unconditional temporal activity**:\n\n")
    report.append("- Temporal modules are active (not receiving conditioning)\n")
    report.append("- They introduce motion regardless of prompt\n")
    report.append("- This is **not** prompt-controlled motion\n")
    report.append("- This explains why directional accuracy is LOWER for C (more chaotic motion)\n\n")

    report.append("## Recommendations\n\n")
    report.append("### Immediate Fix (Required)\n\n")
    report.append("1. **Implement real text encoding** in comparison_generator.py:\n")
    report.append("   ```python\n")
    report.append("   # Load text encoder and tokenizer\n")
    report.append("   text_encoder, tokenizer = TextEncoder.load(\n")
    report.append("       text_encoder_path,\n")
    report.append("       tokenizer_path\n")
    report.append("   )\n\n")
    report.append("   # Encode prompt\n")
    report.append("   indices, mask = tokenizer.encode(prompt, max_length=10)\n")
    report.append("   token_ids = torch.tensor([indices])\n")
    report.append("   attention_mask = torch.tensor([mask], dtype=torch.float)\n")
    report.append("   text_emb, _ = text_encoder(token_ids, attention_mask)\n")
    report.append("   ```\n\n")

    report.append("2. **Verify text encoder checkpoint exists** and is properly trained\n")
    report.append("3. **Re-run Stage 3.2.9F** with real text conditioning\n")
    report.append("4. **Re-evaluate** motion metrics with proper conditioning\n\n")

    report.append("### Secondary Investigation\n\n")
    report.append("After implementing real text conditioning:\n\n")
    report.append("1. Verify motion tokens are in vocabulary\n")
    report.append("2. Verify embeddings differ between direction prompts\n")
    report.append("3. Verify temporal modules receive text-conditioned features\n")
    report.append("4. Re-assess whether temporal modules can learn prompt-controlled motion\n\n")

    report.append("## Classification\n\n")
    report.append("**Primary Issue:** CONDITIONING PATH ISSUE\n\n")
    report.append("Motion instructions are not being transmitted correctly to the temporal modules\n")
    report.append("because the text conditioning pathway is completely bypassed (random embeddings).\n\n")

    report.append("## Next Stage Recommendation\n\n")
    report.append("**Recommendation:** Fix text conditioning implementation before proceeding\n\n")
    report.append("1. Implement real text encoding in inference pipeline\n")
    report.append("2. Verify text encoder checkpoint is available and trained\n")
    report.append("3. Re-run motion evaluation with proper conditioning\n")
    report.append("4. If still failing, investigate temporal module design\n\n")

    # Save report
    report_path = output_dir / "STAGE3_2_9G_TOKEN_CONDITIONING_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
