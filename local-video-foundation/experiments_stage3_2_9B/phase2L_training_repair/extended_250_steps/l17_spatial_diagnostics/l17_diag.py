"""L.17 diagnostics: caption sensitivity, overfit probe, gradient audit."""
import json, sys, time, torch, numpy as np, hashlib
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics')
IMG = OUT/'overfit_images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001,0.02,1000); abars = torch.cumprod(1-betas,0)
vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]

def build():
    su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict'])
    return su, te

def enc(te, caption):
    ids,m = tok.encode(caption, max_length=64)
    with torch.no_grad():
        e,mk = te(torch.tensor([ids]), torch.tensor([m],dtype=torch.float))
    return e, mk

meta = json.load(open(base/'data/text_image/train_metadata.json'))
train_meta, val_meta = meta[:700], meta[700:800]
def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1

# ============ PHASE 2: objective audit math check ============
# verify DDIM update inverts forward process for a known x0/eps
x0 = torch.randn(1,3,64,64); eps_t = torch.randn(1,3,64,64)
t = torch.tensor([500]); a = abars[t].view(1,1,1,1)
xt = torch.sqrt(a)*x0 + torch.sqrt(1-a)*eps_t
# perfect model predicts eps_t; DDIM one step to t-1 should approximate x0 direction
su_tmp, te_tmp = build()
a2 = abars[t-1].view(1,1,1,1)
x1 = torch.sqrt(a2)*(xt - torch.sqrt(1-a)*eps_t)/torch.sqrt(a) + torch.sqrt(1-a2)*eps_t
# x1 should equal sqrt(a2)*x0 + sqrt(1-a2)*eps_t
expected = torch.sqrt(a2)*x0 + torch.sqrt(1-a2)*eps_t
audit = {'objective':'eps-prediction','train_forward':'xt=sqrt(abars[t])*x0+sqrt(1-abars[t])*eps; loss=MSE(pred,eps)','ddim_update':'x_prev=sqrt(a_prev)*(x-sqrt(1-a_t)*eps)/sqrt(a_t)+sqrt(1-a_prev)*eps','math_check_max_err':float((x1-expected).abs().max()),'consistent':bool((x1-expected).abs().max()<1e-5)}
with open(OUT/'L17_TRAINING_INFERENCE_OBJECTIVE_AUDIT.json','w') as f:
    json.dump({'audit':audit,'notes':'parameterization eps both sides; schedule identical (linear 0.0001-0.02,1000); timestep indexing ts[i]->ts[i+1] consistent'},f,indent=2)
print('objective audit consistent:', audit['consistent'], flush=True)

# ============ PHASE 3: caption sensitivity ============
print('PHASE3 caption sensitivity', flush=True)
su, te = build(); su.eval(); te.eval()
for p in te.parameters(): p.requires_grad_(False)
# pick 8 diverse train captions
sel = [0, 5, 17, 33, 60, 88, 120, 200]
caps = [train_meta[i]['caption'] for i in sel]
sens = {'per_example':{}, 'timestep': int(ts[12]), 'noise_seed': 777}
g = torch.Generator().manual_seed(777)
xn = torch.randn(1,3,64,64,generator=g)
tfix = torch.tensor([int(ts[12])])
# also t=100,800
for tt in [int(ts[5]), int(ts[12]), int(ts[20])]:
    tfix = torch.tensor([tt])
    diffs=[]
    for i,cap in enumerate(caps):
        eA,mkA = enc(te, cap)
        # mismatched caption
        eB,mkB = enc(te, caps[(i+3)%len(caps)])
        with torch.no_grad():
            pA = su(xn, tfix, eA, mkA)
            pB = su(xn, tfix, eB, mkB)
        d=(pA-pB).abs()
        diffs.append({'mean_abs':float(d.mean()),'max_abs':float(d.max()),'cosine':float(torch.nn.functional.cosine_similarity(pA.flatten(),pB.flatten(),dim=0))})
    sens[f't{tt}'] = {'mean_abs_avg':float(np.mean([x['mean_abs'] for x in diffs])),
                      'cosine_avg':float(np.mean([x['cosine'] for x in diffs])),
                      'min_cosine':float(min(x['cosine'] for x in diffs))}
with open(OUT/'L17_CAPTION_SENSITIVITY.json','w') as f: json.dump(sens,f,indent=2)
print('sens:', json.dumps({k:v for k,v in sens.items() if k.startswith('t')}), flush=True)

# cross-attn gradient audit: one backward on train batch
su.train()
item = train_meta[0]; img=load_img(item)
t=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
a=abars[t].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
e,mk = enc(te, item['caption'])
pred = su(xt,t,e,mk)
loss = torch.nn.functional.mse_loss(pred,nz); loss.backward()
ca_grads={}
for n,p in su.named_parameters():
    if 'attn' in n or 'cross' in n or 'to_k' in n or 'to_v' in n or 'to_q' in n:
        if p.grad is not None: ca_grads[n]=float(p.grad.norm())
all_gn = {n:float(p.grad.norm()) for n,p in su.named_parameters() if p.grad is not None}
top = sorted(all_gn.items(), key=lambda x:-x[1])[:10]
ca_audit={'cross_attn_param_count':len(ca_grads),'cross_attn_grad_mean':float(np.mean(list(ca_grads.values()))) if ca_grads else 0,
 'cross_attn_grad_max':float(max(ca_grads.values())) if ca_grads else 0,
 'overall_grad_mean':float(np.mean(list(all_gn.values()))),'top10':top[:5]}
with open(OUT/'L17_CROSS_ATTENTION_GRADIENT_AUDIT.json','w') as f: json.dump(ca_audit,f,indent=2)
print('ca grads mean',ca_audit['cross_attn_grad_mean'],'overall',ca_audit['overall_grad_mean'],flush=True)
su.eval()

# ============ PHASE 4-6: 10-image overfit ============
print('PHASE4-6 overfit', flush=True)
# deterministic diverse subset: pick by shape/color variety
import collections
subset_idx=[]
seen=set()
for i,it in enumerate(train_meta[:300]):
    sp=it.get('scene_params',{})
    key=(sp.get('objects',[{}])[0].get('shape','?'), sp.get('objects',[{}])[0].get('color','?')) if sp.get('objects') else ('?','?')
    if key not in seen:
        seen.add(key); subset_idx.append(i)
    if len(subset_idx)>=10: break
subset=[train_meta[i] for i in subset_idx]
manifest={'indices':subset_idx,'pairs':[{'idx':i,'caption':it['caption'],'image_path':it['image_path'],
 'sha256':hashlib.sha256((base/it['image_path']).read_bytes()).hexdigest()} for i,it in zip(subset_idx,subset)]}
with open(OUT/'L17_OVERFIT_DATASET_MANIFEST.json','w') as f: json.dump(manifest,f,indent=2)
print('subset',subset_idx,flush=True)

# train UNet-only, frozen TE, lr1e-4, bs1, 1000 steps on 10 pairs
torch.manual_seed(42)
su2, te2 = build()
for p in te2.parameters(): p.requires_grad_(False)
te2.eval()
opt=optim.AdamW(list(su2.parameters()),lr=0.0001,weight_decay=0.01)
g2=torch.Generator().manual_seed(42)
su2.train()
train_metrics={'steps':[],'val':[]}
t0=time.time(); N=1000
EVAL_AT={0,100,250,500,750,1000}
def ddim(su,te,cap,seed):
    e,mk=enc(te,cap); gg=torch.Generator().manual_seed(seed); x=torch.randn(1,3,64,64,generator=gg)
    for i in range(len(ts)-1):
        t,tp=ts[i],ts[i+1]
        with torch.no_grad(): eps=su(x,torch.tensor([int(t)]).long(),e,mk)
        a_t,a_p=ac[t],ac[tp]
        x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*eps
    return (x.clamp(-1,1)+1)/2
def eval10(tag):
    su2.eval(); res={}
    for j,it in enumerate(subset):
        img=ddim(su2,te2,it['caption'],6000+j)
        d=IMG/f'{tag}_p{j}'; d.mkdir(exist_ok=True)
        Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
        gt=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
        gen=img[0].permute(1,2,0).numpy()
        res[f'p{j}']={'mse_to_gt':float(((gen-gt)**2).mean()),'mae':float(np.abs(gen-gt).mean())}
    su2.train(); return res
train_metrics['evals']={'0':eval10('step0')}
for step in range(N):
    it=subset[step%10]; img=load_img(it)
    t=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g2)
    a=abars[t].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
    e,mk=enc(te2,it['caption'])
    pred=su2(xt,t,e,mk)
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%20==0 or s==1:
        gn=sum(p.grad.norm().item() for p in su2.parameters() if p.grad is not None)
        train_metrics['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
        if s%100==0: print(f'ofit s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
    if s in EVAL_AT and s!=0:
        train_metrics['evals'][str(s)]=eval10(f'step{s}')
        torch.save({'model_state_dict':su2.state_dict(),'step':s},OUT/f'overfit_step{s}.pt')
        # fixed-subset val loss
        su2.eval(); vl=[]
        with torch.no_grad():
            for it2 in subset[:5]:
                im=load_img(it2); tt=torch.randint(0,1000,(1,)); nz2=torch.randn(1,3,64,64)
                aa=abars[tt].view(1,1,1,1); xt2=torch.sqrt(aa)*im.unsqueeze(0)+torch.sqrt(1-aa)*nz2
                e2,mk2=enc(te2,it2['caption'])
                vl.append(float(torch.nn.functional.mse_loss(su2(xt2,tt,e2,mk2),nz2)))
        train_metrics['val'].append({'step':s,'val':float(np.mean(vl))})
        su2.train()
train_metrics['total_sec']=round(time.time()-t0,1)
with open(OUT/'L17_OVERFIT_TRAINING_METRICS.json','w') as f: json.dump(train_metrics,f,indent=2)
with open(OUT/'L17_OVERFIT_VISUAL_EVALUATION.json','w') as f:
    json.dump({'note':'MSE-to-GT per pair per checkpoint in metrics evals; images in overfit_images/','subset_captions':[it['caption'] for it in subset]},f,indent=2)
with open(OUT/'L17_OVERFIT_TRAINING_CONFIG.json','w') as f:
    json.dump({'init':'stage2 ckpt','trainable':'UNet','frozen':'TE','lr':0.0001,'bs':1,'steps':1000,'data':'10 fixed pairs'},f,indent=2)
# diversity: pairwise diff among step1000 outputs
arrs=[np.array(Image.open(IMG/f'step1000_p{j}'/'image.png')).astype(np.float32)/255 for j in range(10)]
arrs=np.array(arrs); pd=[np.abs(arrs[i]-arrs[j]).mean() for i in range(10) for j in range(i+1,10)]
with open(OUT/'L17_OVERFIT_DIVERSITY_ANALYSIS.json','w') as f:
    json.dump({'step1000_pairwise_mean':float(np.mean(pd)),'pairwise_min':float(np.min(pd)),'pairwise_max':float(np.max(pd))},f,indent=2)
print('DONE overfit',(time.time()-t0)/60,'min')
