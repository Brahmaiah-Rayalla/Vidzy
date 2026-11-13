"""Regression test: split encoder must reproduce standard encoder outputs for original vocab."""
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
