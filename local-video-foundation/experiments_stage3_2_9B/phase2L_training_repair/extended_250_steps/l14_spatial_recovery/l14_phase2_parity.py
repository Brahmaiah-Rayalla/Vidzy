"""L.14 Phase 2: text-encoder parity audit (original vs split, per-layer)."""
import json, sys, torch, numpy as np
from pathlib import Path
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.text_encoder_split import load_from_standard_encoder

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l14_spatial_recovery')
base = l12_common.base_dir
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()

# split encoder WITH the fix (transformer weights now copied)
split = load_from_standard_encoder(te, motion_vocab_size=40, freeze_transformer=True, original_vocab_size=42).eval()

# also build the OLD buggy version for comparison: random transformer
torch.manual_seed(0)
split_buggy = load_from_standard_encoder.__wrapped__ if hasattr(load_from_standard_encoder,'__wrapped__') else None
# emulate old behavior: rebuild without weight copy
from models.text_encoder_split import TextEncoderSplit
orig_emb = te.token_emb.weight.data[:42].clone()
split_old = TextEncoderSplit(original_vocab_size=42, motion_vocab_size=40, d_model=256, n_heads=4,
    n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0, original_embeddings=orig_emb, freeze_transformer=True).eval()

def stats(a,b):
    d=(a-b).abs()
    return {'mean_abs':float(d.mean()),'max_abs':float(d.max()),'mse':float(((a-b)**2).mean()),
            'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}

prompts = [p['caption'] for p in select_prompts()] + [
 'A red circle on a white background','A blue square on a white background']
report = {'defect': 'load_from_standard_encoder skipped transformer weight copy when freeze_transformer=True -> random transformer',
          'fix': 'weights now always copied (text_encoder_split.py)', 'per_prompt': {}}
for p in prompts:
    ids,m = tok.encode(p, max_length=64)
    it = torch.tensor([ids]); mt = torch.tensor([m], dtype=torch.float)
    with torch.no_grad():
        e1,_ = te(it,mt); e2,_ = split(it,mt); e3,_ = split_old(it,mt)
        # token-embedding level
        emb1 = te.token_emb(it); emb2 = split.token_emb.token_emb(it) if hasattr(split.token_emb,'token_emb') else split.token_emb(it)
    report['per_prompt'][p] = {
        'final_emb_fixed_vs_standard': stats(e1,e2),
        'final_emb_buggy_vs_standard': stats(e1,e3),
    }
# aggregate
agg = {}
for key in ['final_emb_fixed_vs_standard','final_emb_buggy_vs_standard']:
    agg[key] = {k: float(np.mean([v[key][k] for v in report['per_prompt'].values()])) for k in ['mean_abs','max_abs','mse','cosine']}
report['aggregate'] = agg
# regression test artifact
test = '''"""Regression test: split encoder must reproduce standard encoder outputs for original vocab."""
import torch, sys
from pathlib import Path
sys.path.insert(0, r"C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_7/video_foundation/models")
sys.path.insert(0, r"C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation/experiments_stage3_2_7/video_foundation")
from models.text_encoder import TextEncoder, SimpleTokenizer
from models.text_encoder_split import load_from_standard_encoder
base = Path(r"C:/Brahma/AI/AIDirector/Vidzy/local-video-foundation")
tok = SimpleTokenizer.load(base/"experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json")
ck = torch.load(base/"experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt", map_location="cpu")
vocab = ck["text_encoder_state_dict"]["token_emb.weight"].shape[0]
te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
te.load_state_dict(ck["text_encoder_state_dict"]); te.eval()
split = load_from_standard_encoder(te, motion_vocab_size=40, freeze_transformer=True, original_vocab_size=42).eval()
ids, m = tok.encode("A red circle on a white background", max_length=64)
it = torch.tensor([ids]); mt = torch.tensor([m], dtype=torch.float)
with torch.no_grad():
    e1,_ = te(it,mt); e2,_ = split(it,mt)
d = (e1-e2).abs()
assert float(d.max()) < 1e-4, f"parity failed: max diff {float(d.max())}"
print("PARITY TEST PASSED, max diff", float(d.max()))
'''
(OUT/'test_text_encoder_parity.py').write_text(test)
with open(OUT/'L14_TEXT_ENCODER_PARITY_AUDIT.json','w') as f:
    json.dump(report, f, indent=2)
print(json.dumps(agg, indent=1))
