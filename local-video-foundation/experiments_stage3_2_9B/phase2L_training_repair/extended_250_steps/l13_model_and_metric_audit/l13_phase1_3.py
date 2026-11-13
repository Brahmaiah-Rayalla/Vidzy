"""
L.13 Phase 1 + Phase 3: checkpoint/conditioning audit and bottleneck fusion forensics.
"""
import json, hashlib, sys, torch, numpy as np
from pathlib import Path

L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
from l12_common import *
import l12_common
from models.text_encoder import SimpleTokenizer

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l13_model_and_metric_audit')
base = l12_common.base_dir
report1 = {'checkpoints': {}, 'conditioning': {}, 'forward_signature': 'VideoUNetFixed.forward(x[B,C,T,H,W], timesteps[B], text_emb[B,S,D], text_mask[B,S])'}

# ---- Phase 1: hashes + structure ----
files = {
 'stage2_spatial_ckpt': base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt',
 'stage2_spatial_alt': base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt',
 'C1_250': CKPT_C1, 'C25_250': CKPT_C25,
 'tokenizer': base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json',
 'cond_final': base/'checkpoints_conditional/checkpoint_final.pt',
}
for k, p in files.items():
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        report1['checkpoints'][k] = {'path': str(p), 'sha256': h, 'size_mb': round(p.stat().st_size/1e6,1)}
report1['stage2_reference_sha256_expected'] = 'a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c'
report1['stage2_reference_sha256_match'] = report1['checkpoints'].get('stage2_spatial_ckpt',{}).get('sha256') == 'a7f5142e133c6aeadc656967639eab046a7a8f922e3201a6feda4496a544d57c'

# checkpoint internals (CPU load, keys only)
for k in ['stage2_spatial_ckpt','C1_250','C25_250']:
    ck = torch.load(files[k], map_location='cpu')
    info = {'keys': list(ck.keys())}
    if 'model_state_dict' in ck:
        sd = ck['model_state_dict']
        g = {n: sd[n] for n in sd if 'gate' in n}
        info['gate_params'] = {n: float(v) for n, v in g.items()}
        info['n_state_tensors'] = len(sd)
    if 'text_encoder_state_dict' in ck:
        te = ck['text_encoder_state_dict']
        info['text_encoder_vocab'] = int(te['token_emb.weight'].shape[0]) if 'token_emb.weight' in te else 'n/a'
    info['step'] = ck.get('step'); info['experiment'] = ck.get('experiment'); info['gate_init'] = ck.get('gate_init')
    report1['checkpoints'][k].update(info)

# tokenizer vocab check for candidate spatial prompts
tok = SimpleTokenizer.load(files['tokenizer'])
candidate_prompts = [
 'A red circle on a white background',
 'A blue square on a white background',
 'A yellow triangle on a white background',
 'A green circle at the center',
 'A red square on a white background',
 'A medium blue circle slowly moving from left to right on a white background.',
]
vocab_report = {}
for p in candidate_prompts:
    ids, mask = tok.encode(p, max_length=64)
    unk = ids.count(tok.SPECIAL_TOKENS.get('<UNK>', -1)) if hasattr(tok,'SPECIAL_TOKENS') else sum(1 for i in ids if i==1)
    vocab_report[p] = {'unk_count': unk, 'n_tokens': sum(mask)}
report1['conditioning'] = {'tokenizer': str(files['tokenizer']), 'vocab_prompts': vocab_report,
 'note': 'original_vocab_size=42 for split encoder; motion_vocab_size=40; freeze_transformer=True'}

with open(OUT/'L13_CHECKPOINT_AND_CONDITIONING_AUDIT.json','w') as f:
    json.dump(report1, f, indent=2)
print('Phase1 done. sha match:', report1['stage2_reference_sha256_match'])
print(json.dumps(vocab_report, indent=1))

# ---- Phase 3: bottleneck fusion forensics ----
print('Loading C25 for forensic tests...')
model = load_model(CKPT_C25)
su = model.spatial_unet
prompts = select_prompts()
cap = [p['caption'] for p in prompts if p['motion_type']=='stationary'][0]
te, tm = encode_prompt(model, tok, cap)
g = torch.Generator().manual_seed(1005)
x = torch.randn(1,3,16,64,64,generator=g)
ac, ts = build_diffusion_schedule()
t = ts[10]; tt = torch.tensor([int(t)]).long()

# get feature shapes via return_features call
x_sp = x.permute(0,2,1,3,4).reshape(16,3,64,64)
t_sp = tt.unsqueeze(1).expand(-1,16).reshape(16)
te_sp = te.unsqueeze(1).expand(-1,16,-1,-1).reshape(16,te.shape[1],te.shape[2])
tm_sp = tm.unsqueeze(1).expand(-1,16,-1).reshape(16,tm.shape[1])
with torch.no_grad():
    out0, feats = su(x_sp, t_sp, te_sp, tm_sp, return_features=True)
print('feature shapes:', {k: tuple(v.shape) for k,v in feats.items()})

bn = feats['bottleneck']; lr = feats['low_res']
torch.manual_seed(0)
rand_bn = torch.randn_like(bn); rand_lr = torch.randn_like(lr)

forensic = {'input_shapes': {k: list(v.shape) for k,v in feats.items()}, 'tests': {}}
with torch.no_grad():
    base_out = su(x_sp, t_sp, te_sp, tm_sp)
    # inject LARGE random at bottleneck only
    o1 = su(x_sp, t_sp, te_sp, tm_sp, injected_features={'bottleneck': rand_bn*10})
    forensic['tests']['bottleneck_rand10_maxdiff'] = float((o1-base_out).abs().max())
    forensic['tests']['bottleneck_rand10_meandiff'] = float((o1-base_out).abs().mean())
    # inject at low_res only
    o2 = su(x_sp, t_sp, te_sp, tm_sp, injected_features={'low_res': rand_lr*10})
    forensic['tests']['lowres_rand10_maxdiff'] = float((o2-base_out).abs().max())
    forensic['tests']['lowres_rand10_meandiff'] = float((o2-base_out).abs().mean())
    # both
    o3 = su(x_sp, t_sp, te_sp, tm_sp, injected_features={'bottleneck': rand_bn*10, 'low_res': rand_lr*10})
    forensic['tests']['both_rand10_maxdiff'] = float((o3-base_out).abs().max())

# gate sweep on full model forward at fixed input
set_gate_override(model, 0.25, 0.25)
eps = {}
for bn_g, lr_g in [(0.0,0.25),(0.10,0.25),(0.25,0.25),(0.25,0.10),(0.25,0.0),(0.0,0.0)]:
    set_gate_override(model, bn_g, lr_g)
    with torch.no_grad():
        e = model(x, tt, te, tm)
    eps[(bn_g,lr_g)] = e
sweep = {}
keys = list(eps)
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a,b = keys[i], keys[j]
        sweep[f'{a}_vs_{b}'] = float((eps[a]-eps[b]).abs().max())
forensic['gate_sweep_maxdiff'] = sweep
forensic['gate_sweep_eps_std'] = {str(k): float(v.std()) for k,v in eps.items()}

# restore + verify native
model.fusion_modules['bottleneck'].gate.data.fill_(-1.0409202575683594)
model.fusion_modules['low_res'].gate.data.fill_(-1.0322657823562622)
forensic['restored_gates'] = get_gate_raw_and_activated(model)

# autograd connectivity check: does bottleneck gate get gradient?
model.train()
xt = x.clone().requires_grad_(False)
set_gate_override(model, 0.25, 0.25)
pred = model(x, tt, te, tm)
loss = pred.pow(2).mean()
loss.backward()
for n in ['bottleneck','low_res']:
    p = model.fusion_modules[n].gate
    forensic.setdefault('gate_grads',{})[n] = float(p.grad) if p.grad is not None else None
model.eval()

with open(OUT/'L13_BOTTLENECK_FUSION_AUDIT.json','w') as f:
    json.dump({'forensics': forensic,
               'fusion_equation': 'fused = temporal_features + sigmoid(gate) * (temporal_processed - temporal_features); injected as h = h + fused (FULL fused tensor added, not just gated residual)',
               'conceptual_comparison': 'current = spatial + gate*temporal_residual applied on top of full-tensor injection; interpolation form fused=(1-g)*spatial+g*temporal NOT implemented'}, f, indent=2)
# lowres audit = same forensic data filtered
with open(OUT/'L13_LOWRES_FUSION_AUDIT.json','w') as f:
    json.dump({'note':'See L13_BOTTLENECK_FUSION_AUDIT.json forensics; low_res injection is applied after decoder up_blocks before conv_out and DOES affect output (rand10 injection maxdiff recorded there).',
               'injection_point':'post-decoder, pre-conv_out, 64ch @ 64x64',
               'forensics': forensic}, f, indent=2)
print('Phase3 done.')
print(json.dumps(foreensic['tests'], indent=1))
print(json.dumps(foreensic['gate_sweep_maxdiff'], indent=1))
print('gate grads:', forensic.get('gate_grads'))
