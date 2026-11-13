"""
Test motion tokenization for expanded vocabulary.
"""

import json

new_tokenizer_path = "checkpoints/tokenizer_motion_expanded.json"

# Load tokenizer with proper parsing
with open(new_tokenizer_path, 'r') as f:
    tokenizer_data = json.load(f)

word_to_idx = {}
for key, value in tokenizer_data['word_to_idx'].items():
    if key.startswith('<'):
        word_to_idx[key] = value
    elif key.isdigit():
        # This is a numeric key mapping to a word
        word_to_idx[value] = int(key)
    else:
        # This is a word key mapping to an index
        word_to_idx[key] = value

# Test motion phrases
test_phrases = [
    "moving left",
    "moving right",
    "moving slowly",
    "moving quickly",
    "stationary object",
    "bouncing up and down",
    "rotating clockwise",
    "diagonal movement",
    "accelerating forward",
]

unk_token = word_to_idx.get('<UNK>', 1)

print(f"UNK token ID: {unk_token}")
print(f"Vocabulary size: {len(word_to_idx)}")
print(f"Sample motion tokens: moving={word_to_idx.get('moving')}, left={word_to_idx.get('left')}, right={word_to_idx.get('right')}")

print("\nTokenization results:")
all_passed = True
for phrase in test_phrases:
    tokens = phrase.lower().split()
    indices = []
    for token in tokens:
        if token in word_to_idx:
            indices.append(word_to_idx[token])
        else:
            indices.append(unk_token)
    unk_count = indices.count(unk_token)

    status = "[PASS]" if unk_count == 0 else "[FAIL]"
    print(f"  {status} '{phrase}': {unk_count} UNK tokens, indices: {indices}")

    if unk_count > 0:
        all_passed = False

if all_passed:
    print("\n[SUCCESS] All motion phrases tokenized without UNK tokens")
else:
    print("\n[FAILURE] Some motion phrases still have UNK tokens")
