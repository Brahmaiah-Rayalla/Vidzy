"""
Stage 3.2.9B - Phase 1.2: Verify Real Motion-Token Coverage

Test actual prompts to verify motion words map to intended token IDs.

Author: Devin
Date: 2026-11-02
"""

import sys
import os
import json
from pathlib import Path

# Add paths
parent_dir = Path(__file__).parent.parent
stage3_2_7_dir = parent_dir / "experiments_stage3_2_7"
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(stage3_2_7_dir))

from video_foundation.models.text_encoder import SimpleTokenizer


class MotionTokenCoverageVerifier:
    """Verify motion-token coverage for actual prompts."""

    def __init__(self, tokenizer_path: str):
        self.tokenizer_path = tokenizer_path
        self.tokenizer = SimpleTokenizer.load(tokenizer_path)

        self.appearance_vocab_size = 42
        self.motion_vocab_size = 40
        self.total_vocab_size = self.appearance_vocab_size + self.motion_vocab_size

        self.results = {
            'tokenizer_info': {},
            'prompt_tests': [],
            'coverage_summary': {}
        }

    def log_section(self, title: str):
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)

    def verify_prompt(self, prompt: str) -> dict:
        """Verify a single prompt."""
        token_ids, mask = self.tokenizer.encode(prompt, max_length=64)

        # Decode tokens
        decoded_tokens = []
        for token_id in token_ids:
            if token_id in self.tokenizer.idx_to_word:
                decoded_tokens.append(self.tokenizer.idx_to_word[token_id])
            else:
                decoded_tokens.append(f"<UNK_{token_id}>")

        # Classify tokens
        appearance_tokens = []
        motion_tokens = []
        unk_tokens = []

        for i, token_id in enumerate(token_ids):
            if token_id < self.appearance_vocab_size:
                appearance_tokens.append({
                    'position': i,
                    'token_id': token_id,
                    'word': decoded_tokens[i]
                })
            elif token_id < self.total_vocab_size:
                motion_tokens.append({
                    'position': i,
                    'token_id': token_id,
                    'word': decoded_tokens[i]
                })
            else:
                unk_tokens.append({
                    'position': i,
                    'token_id': token_id,
                    'word': decoded_tokens[i]
                })

        return {
            'prompt': prompt,
            'token_ids': token_ids,
            'decoded_tokens': decoded_tokens,
            'appearance_tokens': appearance_tokens,
            'motion_tokens': motion_tokens,
            'unk_tokens': unk_tokens,
            'has_motion_tokens': len(motion_tokens) > 0,
            'has_unk_tokens': len(unk_tokens) > 0
        }

    def run_verification(self):
        """Run motion-token coverage verification."""
        self.log_section("MOTION-TOKEN COVERAGE VERIFICATION")

        # Tokenizer info
        self.results['tokenizer_info'] = {
            'vocab_size': self.tokenizer.vocab_size,
            'appearance_vocab_size': self.appearance_vocab_size,
            'motion_vocab_size': self.motion_vocab_size,
            'total_vocab_size': self.total_vocab_size
        }

        print(f"Tokenizer Info:")
        print(f"  Total vocab size: {self.tokenizer.vocab_size}")
        print(f"  Appearance vocab size: {self.appearance_vocab_size} (tokens 0-{self.appearance_vocab_size-1})")
        print(f"  Motion vocab size: {self.motion_vocab_size} (tokens {self.appearance_vocab_size}-{self.total_vocab_size-1})")

        # Test prompts
        test_prompts = [
            "A red object moving left.",
            "A blue object moving right.",
            "An object moving upward.",
            "An object moving downward.",
            "An object moving slowly.",
            "An object moving quickly.",
            "An object bouncing.",
            "A stationary object."
        ]

        self.log_section("PROMPT TESTS")

        for prompt in test_prompts:
            result = self.verify_prompt(prompt)
            self.results['prompt_tests'].append(result)

            print(f"\nPrompt: \"{prompt}\"")
            print(f"  Token IDs: {result['token_ids']}")
            print(f"  Decoded: {' '.join(result['decoded_tokens'])}")
            print(f"  Appearance tokens: {len(result['appearance_tokens'])}")
            print(f"  Motion tokens: {len(result['motion_tokens'])}")
            print(f"  UNK tokens: {len(result['unk_tokens'])}")

            if result['motion_tokens']:
                print(f"  Motion token details:")
                for mt in result['motion_tokens']:
                    print(f"    Position {mt['position']}: ID {mt['token_id']} -> '{mt['word']}'")

            if result['unk_tokens']:
                print(f"  WARNING: Found UNK tokens:")
                for ut in result['unk_tokens']:
                    print(f"    Position {ut['position']}: ID {ut['token_id']} -> '{ut['word']}'")

        # Summary
        self.log_section("COVERAGE SUMMARY")

        prompts_with_motion = sum(1 for r in self.results['prompt_tests'] if r['has_motion_tokens'])
        prompts_with_unk = sum(1 for r in self.results['prompt_tests'] if r['has_unk_tokens'])

        self.results['coverage_summary'] = {
            'total_prompts_tested': len(test_prompts),
            'prompts_with_motion_tokens': prompts_with_motion,
            'prompts_with_unk_tokens': prompts_with_unk,
            'motion_token_coverage_rate': prompts_with_motion / len(test_prompts)
        }

        print(f"Total prompts tested: {len(test_prompts)}")
        print(f"Prompts with motion tokens: {prompts_with_motion}")
        print(f"Prompts with UNK tokens: {prompts_with_unk}")
        print(f"Motion token coverage rate: {prompts_with_motion / len(test_prompts):.2%}")

        # Analysis
        self.log_section("ANALYSIS")

        if prompts_with_unk > 0:
            print("WARNING: Some prompts contain UNK tokens.")
            print("This indicates motion words are not in the vocabulary.")
            print("Motion embeddings will not be trained for these words.")
        else:
            print("All prompts tokenized without UNK tokens.")

        if prompts_with_motion == 0:
            print("CRITICAL: No prompts contain motion tokens!")
            print("Motion embeddings will not receive gradients during training.")
            print("This is a blocker for pilot training.")
        elif prompts_with_motion < len(test_prompts):
            print(f"CAUTION: Only {prompts_with_motion}/{len(test_prompts)} prompts contain motion tokens.")
            print("Motion embeddings may receive limited gradients.")
        else:
            print(f"Good: {prompts_with_motion}/{len(test_prompts)} prompts contain motion tokens.")

        return self.results


def main():
    parent_dir = Path(__file__).parent.parent
    tokenizer_path = str(parent_dir / "experiments_stage3_2_7" / "checkpoints" / "tokenizer_motion_expanded.json")

    verifier = MotionTokenCoverageVerifier(tokenizer_path)
    results = verifier.run_verification()

    # Save results
    results_path = Path(__file__).parent / "STAGE3_2_9B_MOTION_TOKEN_COVERAGE.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
