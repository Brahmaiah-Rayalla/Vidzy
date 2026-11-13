"""L.14 Phase 3+4: corrected-inference baseline retest + 200-step spatial probe."""
import json, sys, time, torch, numpy as np
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l14_spatial_recovery')
IMG = OUT/'baseline_images'; IMG.mkdir(exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json'.replace('tokenizer_motion_expanded.json','checkpoint_stage2_motion_expanded.pt'), map_location='cpu')

def build_stage2(trainable=True):
    vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
    su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict'])
    if trainable:
        for p in su.parameters(): p.requires_grad_(True)
        for p in te.parameters(): p.requires_grad_(True)
    else:
        su.eval(); te.eval()
    return su, te

ac, ts = build_diffusion_schedule()

def encode(te, prompt):
    ids, m = tok.encode(prompt, max_length=64)
    with torch.no_grad():
        e, mk = te(torch.tensor([ids]), torch.tensor([m], dtype=torch.float))
    return e, mk

def ddim(su, te, prompt, seed):
    e, mk = encode(te, prompt)
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1,3,64,64,generator=g)
    for i in range(len(ts)-1):
        t, tp = ts[i], ts[i+1]
        with torch.no_grad():
            eps = su(x, torch.tensor([int(t)]).long(), e, mk)
        a_t, a_p = ac[t], ac[tp]
        x = torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*eps
    return (x.clamp(-1,1)+1)/2

def color_stats(img, color):
    rgb = img[0].permute(1,2,0).numpy(); r,g,b = rgb[...,0],rgb[...,1],rgb[...,2]
    rules = {'red':(r>0.5)&(r>g*1.8)&(r>b*1.5),'blue':(b>0.5)&(b>r*1.5)&(b>g*1.2),
             'yellow':(r>0.6)&(g>0.6)&(b<0.4),'green':(g>0.4)&(g>r*1.4)&(g>b*1.2),
             'pink':(r>0.45)&(b>0.3)&(r>g*1.4)&(b>g*1.2),'purple':(r>0.3)&(b>0.3)&(g<r*0.7)&(np.abs(r-b)<0.35),
             'cyan':(g>0.4)&(b>0.4)&(r<g*0.6)&(r<b*0.6),'white':(r>0.75)&(g>0.75)&(b>0.75)}
    m = rules.get(color, lambda *a: np.zeros_like(r,dtype=bool))
    return {'color_frac': float(m.mean()), 'corner_std': float(np.concatenate([rgb[:8,:8].reshape(-1,3),rgb[-8:,-8:].reshape(-1,3)]).std()),
            'mean_rgb': rgb.reshape(-1,3).mean(0).round(3).tolist()}

# ---------- Phase 3: corrected baseline ----------
print('=== PHASE 3: corrected baseline ===', flush=True)
su0, te0 = build_stage2(trainable=False)
PROMPTS = [
 ('simple_red_circle','A red circle on a white background','red'),
 ('simple_blue_square','A blue square on a white background','blue'),
 ('simple_yellow_triangle','A yellow triangle on a white background','yellow'),
 ('train_ltr','A medium blue circle slowly moving from left to right on a white background.','blue'),
 ('train_rtl','A large pink triangle quickly moving from right to left on a white background.','pink'),
 ('train_stat','A medium cyan square on a gray background, remaining stationary.','cyan'),
]
base_res = {}
for name, prompt, col in PROMPTS:
    for seed in [5000, 5001]:
        img = ddim(su0, te0, prompt, seed)
        d = IMG/f'{name}_s{seed}'; d.mkdir(exist_ok=True)
        Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
        np.save(d/'image.npy', img[0].permute(1,2,0).numpy())
        base_res[f'{name}_s{seed}'] = color_stats(img, col)
    print(f'  {name} done', flush=True)
with open(OUT/'L14_SPATIAL_BASELINE_RESULTS.json','w') as f:
    json.dump({'note':'corrected text-encoder path (parity fix applied); Stage2.1 checkpoint unchanged',
               'results': base_res}, f, indent=2)

# ---------- Phase 4: spatial probe (200 steps) ----------
print('=== PHASE 4: spatial probe ===', flush=True)
meta = json.load(open(base/'experiments_stage3_2_7/data_synthetic/train_metadata.json'))
# build image dataset: use first 60 videos for train, rest for val
train_items, val_items = meta[:60], meta[60:80]
def load_frame(item, fi):
    v = np.load(item['video_path'])[fi]  # H,W,3 uint8
    t = torch.from_numpy(v).float().permute(2,0,1)/255.*2-1
    return t
train_set = [(it, fi) for it in train_items for fi in [0,5,10,15]]
val_set = [(it, fi) for it in val_items for fi in [0,8,15]]
print(f'train imgs={len(train_set)} val imgs={len(val_set)}', flush=True)

su, te = build_stage2(trainable=True)
opt = optim.AdamW(list(su.parameters())+list(te.parameters()), lr=0.0002, weight_decay=0.01)
betas, alphas, _ = torch.linspace(0.0001,0.02,1000), None, None
betas = torch.linspace(0.0001,0.02,1000); alphas = 1-betas; abars = torch.cumprod(alphas,0)

def batch_noise_loss(model_su, model_te, item, fi, gen):
    img = load_frame(item, fi)
    t = torch.randint(0,1000,(1,))
    noise = torch.randn(1,3,64,64,generator=gen)
    at = abars[t]; xt = torch.sqrt(at)*img.unsqueeze(0)+torch.sqrt(1-at)*noise
    ids,m = tok.encode(item['caption'], max_length=64)
    with torch.enable_grad():
        e,mk = model_te(torch.tensor([ids]), torch.tensor([m],dtype=torch.float))
        pred = model_su(xt, t, e, mk)
        loss = torch.nn.functional.mse_loss(pred, noise)
    return loss

metrics = {'steps': [], 'val': []}
torch.manual_seed(42); np.random.seed(42)
gen = torch.Generator().manual_seed(42)
su.train(); te.train()
t0=time.time()
N=200
for step in range(N):
    item, fi = train_set[step % len(train_set)]
    loss = batch_noise_loss(su, te, item, fi, gen)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%10==0 or step==0:
        gn = sum(p.grad.norm().item() for p in list(su.parameters())+list(te.parameters()) if p.grad is not None)
        metrics['steps'].append({'step':step+1,'loss':float(loss),'grad_norm':gn,'sec':round(time.time()-t0,1)})
        print(f'step {step+1} loss={float(loss):.4f} {time.time()-t0:.0f}s', flush=True)
    if (step+1) in (50,100,150,200):
        # validation loss on fixed 8 imgs
        su.eval(); te.eval(); vl=[]
        with torch.no_grad():
            for it,fi2 in val_set[:8]:
                img=load_frame(it,fi2); tt=torch.randint(0,1000,(1,))
                nz=torch.randn(1,3,64,64)
                a=abars[tt]; xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
                ids,m=tok.encode(it['caption'],max_length=64)
                e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
                vl.append(float(torch.nn.functional.mse_loss(su(xt,tt,e,mk),nz)))
        metrics['val'].append({'step':step+1,'val_loss':float(np.mean(vl))})
        print('  val_loss',round(float(np.mean(vl)),4), flush=True)
        su.train(); te.train()
# save probe checkpoint (disposable copy location)
torch.save({'model_state_dict':su.state_dict(),'text_encoder_state_dict':te.state_dict(),
            'step':200,'note':'L14 spatial probe, disposable'}, OUT/'probe_step200.pt')
# final samples: eval mode, fixed prompts+seeds
su.eval(); te.eval()
probe_res={}
for name,prompt,col in PROMPTS:
    for seed in [5000,5001]:
        img=ddim(su,te,prompt,seed)
        d=IMG/f'probe_{name}_s{seed}'; d.mkdir(exist_ok=True)
        Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
        probe_res[f'{name}_s{seed}']=color_stats(img,col)
    print('  probe sample',name,flush=True)
with open(OUT/'L14_SPATIAL_PROBE_RESULTS.json','w') as f:
    json.dump({'metrics':metrics,'samples':probe_res,'trainable':'unet+text_encoder','lr':0.0002,'steps':200,'seed':42},f,indent=2)
with open(OUT/'L14_SPATIAL_TRAINING_METRICS.json','w') as f:
    json.dump(metrics,f,indent=2)
with open(OUT/'L14_SPATIAL_QUALITY_EVALUATION.json','w') as f:
    json.dump({'baseline':base_res,'probe':probe_res},f,indent=2)
print(f'DONE in {(time.time()-t0)/60:.1f} min')
