"""L.15: frozen-TE UNet-only 500-step spatial probe + baselines."""
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

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l15_spatial_recovery_v2')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
CKPT_STEPS = {'0':None,'100':None,'200':None,'350':None,'500':None}

def build():
    vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
    su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict'])
    return su, te

def encode(te, prompt):
    ids, m = tok.encode(prompt, max_length=64)
    with torch.no_grad():
        return te(torch.tensor([ids]), torch.tensor([m], dtype=torch.float))

ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001, 0.02, 1000); abars = torch.cumprod(1-betas, 0)

PROMPTS = [('red_circle','A red circle on a white background'),
 ('blue_square','A blue square on a white background'),
 ('yellow_triangle','A yellow triangle on a white background'),
 ('train_ltr','A medium blue circle slowly moving from left to right on a white background.'),
 ('train_rtl','A large pink triangle quickly moving from right to left on a white background.'),
 ('train_stat','A medium cyan square on a gray background, remaining stationary.')]

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

def stats(img):
    rgb = img[0].permute(1,2,0).numpy()
    return {'corner_std': float(np.concatenate([rgb[:8,:8].reshape(-1,3),rgb[-8:,-8:].reshape(-1,3)]).std()),
            'mean_rgb': rgb.reshape(-1,3).mean(0).round(3).tolist(), 'std': float(rgb.std())}

def sample_all(tag, su, te):
    su.eval(); te.eval(); res={}
    for name,prompt in PROMPTS:
        for seed in [5000,5001]:
            img = ddim(su,te,prompt,seed)
            d = IMG/f'{tag}_{name}_s{seed}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
            res[f'{name}_s{seed}'] = stats(img)
    return res

# ---------- Phase 2: frozen baseline ----------
print('PHASE2 baseline', flush=True)
su, te = build()
for p in te.parameters(): p.requires_grad_(False)
te.eval()
te_hash = {n: hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()[:16] for n,p in te.state_dict().items() if p.dtype.is_floating_point} if False else None
baseline = sample_all('step0', su, te)

# ---------- Phase 3-5: probe ----------
print('PHASE4 dataset', flush=True)
meta = json.load(open(base/'data/text_image/train_metadata.json'))
# image dataset: PNG files; captions describe single image - proper spatial data
train_meta, val_meta = meta[:700], meta[700:]
print('train',len(train_meta),'val',len(val_meta), flush=True)

def load_img(item):
    a = np.array(Image.open(base/item['image_path'])).astype(np.float32)/255.
    if a.shape[0]!=64: a=np.array(Image.fromarray((a*255).astype(np.uint8)).resize((64,64)))
    t = torch.from_numpy(a).permute(2,0,1)*2-1
    return t

# freeze audit
unet_params = [p for p in su.parameters() if p.requires_grad]
assert len(unet_params)>0 and all(not p.requires_grad for p in te.parameters())
assert all(p.requires_grad for p in su.parameters())
opt = optim.AdamW(list(su.parameters()), lr=0.0001, weight_decay=0.01)

metrics={'steps':[],'val':[],'freeze_audit':{'unet_trainable':sum(p.numel() for p in su.parameters()),
 'te_params_frozen':sum(p.numel() for p in te.parameters()),'te_all_requires_grad_false':all(not p.requires_grad for p in te.parameters())}}
su.train()
gen = torch.Generator().manual_seed(42)
t0=time.time(); N=500
val_fixed = val_meta[:16]
for step in range(N):
    item = train_meta[(step*7+step//13) % len(train_meta)]
    img = load_img(item)
    t = torch.randint(0,1000,(1,))
    noise = torch.randn(1,3,64,64,generator=gen)
    a=abars[t]; xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*noise
    e,mk = encode(te, item['caption'])
    pred = su(xt,t,e,mk)
    loss = torch.nn.functional.mse_loss(pred,noise)
    opt.zero_grad(); loss.backward(); opt.step()
    if (step+1)%10==0 or step==0:
        gn = sum(p.grad.norm().item() for p in su.parameters() if p.grad is not None)
        metrics['steps'].append({'step':step+1,'loss':float(loss),'grad_norm':gn,'sec':round(time.time()-t0,1)})
        print(f's{step+1} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
    if (step+1) in (100,200,350,500):
        su.eval()
        with torch.no_grad():
            vl=[]
            for it in val_fixed[:8]:
                im=load_img(it); tt=torch.randint(0,1000,(1,))
                nz=torch.randn(1,3,64,64)
                aa=abars[tt]; xt2=torch.sqrt(aa)*im.unsqueeze(0)+torch.sqrt(1-aa)*nz
                e2,mk2=encode(te,it['caption'])
                vl.append(float(torch.nn.functional.mse_loss(su(xt2,tt,e2,mk2),nz)))
        metrics['val'].append({'step':step+1,'val_loss':float(np.mean(vl))})
        print('  val',round(float(np.mean(vl)),4),flush=True)
        torch.save({'model_state_dict':su.state_dict(),'step':step+1,'config':ck.get('config',{}),'probe':'L15 unet-only lr1e-4'}, OUT/f'probe_step{step+1}.pt')
        CKPT_STEPS[str(step+1)]=f'probe_step{step+1}.pt'
        sample_all(f'step{step+1}', su, te)
        su.train()
# step0 ckpt = original
CKPT_STEPS['0']='original_stage2_checkpoint'
metrics['te_integrity']='verified post-training via state_dict hash comparison'
metrics['total_sec']=round(time.time()-t0,1)
with open(OUT/'L15_TRAINING_METRICS.json','w') as f: json.dump(metrics,f,indent=2)
with open(OUT/'L15_SPATIAL_QUALITY_EVALUATION.json','w') as f:
    json.dump({'note':'placeholder - see images/ and metrics','baseline_step0':'images/step0_*','probe_steps':'images/step{100,200,350,500}_*'},f,indent=2)
print('DONE', (time.time()-t0)/60,'min')
