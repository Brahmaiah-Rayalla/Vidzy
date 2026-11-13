"""L.20: learned-query + FiLM and pooled-FiLM conditioning prototypes."""
import json, sys, math, time, torch, numpy as np, hashlib
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet, ConditionalResidualBlock
import torch.optim as optim
import torch.nn.functional as F

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l20_conditioning_architecture')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001,0.02,1000); abars = torch.cumprod(1-betas,0)
vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
meta = json.load(open(base/'data/text_image/train_metadata.json'))
train_meta = meta[:700]

def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1

def build_unet():
    su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict']); su.eval()
    return su

def build_te():
    te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
    for p in te.parameters(): p.requires_grad_(False)
    return te

def enc(te,cap):
    ids,m=tok.encode(cap,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
    return e,mk

# ---------- FiLM conditioning modules ----------
class PooledFiLM(torch.nn.Module):
    """Architecture B: masked-mean text -> MLP -> per-block gamma/beta."""
    def __init__(self, block_channels):
        super().__init__()
        self.block_channels=block_channels  # list[int]
        self.mlp=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,256),torch.nn.SiLU())
        self.heads=torch.nn.ModuleList([torch.nn.Linear(256,2*c) for c in block_channels])
        for h in self.heads: torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
    def forward(self,text_emb,mask):
        w=mask.unsqueeze(-1).float()
        pooled=(text_emb*w).sum(1)/w.sum(1).clamp_min(1)
        ctx=self.mlp(pooled)
        outs=[h(ctx) for h in self.heads]
        return outs  # list of [B,2C]

class LearnedQueryFiLM(torch.nn.Module):
    """Architecture A: learned queries attend to text tokens -> context -> FiLM."""
    def __init__(self, block_channels, n_queries=8, d=256):
        super().__init__()
        self.nq=n_queries; self.d=d
        self.queries=torch.nn.Parameter(torch.randn(n_queries,d)*0.02)
        self.k_proj=torch.nn.Linear(d,d); self.v_proj=torch.nn.Linear(d,d)
        self.attn_out=torch.nn.Linear(d,d)
        self.mlp=torch.nn.Sequential(torch.nn.Linear(n_queries*d,d),torch.nn.SiLU(),torch.nn.Linear(d,d),torch.nn.SiLU())
        self.heads=torch.nn.ModuleList([torch.nn.Linear(d,2*c) for c in block_channels])
        for h in self.heads: torch.nn.init.zeros_(h.weight); torch.nn.init.zeros_(h.bias)
    def forward(self,text_emb,mask):
        B,S,D=text_emb.shape
        K=self.k_proj(text_emb); V=self.v_proj(text_emb)         # [B,S,D]
        Q=self.queries.unsqueeze(0).expand(B,-1,-1)              # [B,nq,D]
        # single-head scaled dot product
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.d)  # [B,nq,S]
        mk=mask.unsqueeze(1)                                     # [B,1,S]
        scores=scores.masked_fill(mk==0,float('-inf'))
        w=torch.softmax(scores,-1)                               # [B,nq,S]
        ctx=torch.matmul(w,V)                                    # [B,nq,D]
        ctx=ctx.reshape(B,-1)                                    # [B,nq*D]
        self.last_attn=w
        h=self.mlp(ctx)
        outs=[hd(h) for hd in self.heads]
        return outs

def attach_film(su, film, tag):
    """Post-hooks on ConditionalResidualBlock outputs applying per-channel FiLM."""
    blocks=[m for m in su.modules() if isinstance(m,ConditionalResidualBlock)]
    assert len(blocks)==len(film.heads),(len(blocks),len(film.heads))
    handles=[]
    def make(i,b):
        def hook(mod,inp,out):
            gb=film.current[i]  # [B,2C]
            g,beta=gb.chunk(2,-1)
            g=g.view(1,-1,1,1); beta=beta.view(1,-1,1,1)
            return out*(1+g)+beta
        return hook
    for i,b in enumerate(blocks):
        handles.append(b.register_forward_hook(make(i,b)))
    # wrap film forward to stash per-block params
    orig=film.forward
    def fwd2(te_,mk_):
        outs=orig(te_,mk_)
        film.current=outs
        return outs
    film.forward=fwd2
    return handles

def forward_eps(su,film,te,x,t,cap):
    e,mk=enc(te,cap)
    film(e,mk)  # populates film.current
    return su(x,t,e,mk)

# ---------- diagnostics on 8 captions ----------
print('building models',flush=True)
suA=build_unet(); teA=build_te()
blocks=[m for m in suA.modules() if isinstance(m,ConditionalResidualBlock)]
chs=[b.conv2.out_channels for b in blocks]
filmA=LearnedQueryFiLM(chs); filmB=PooledFiLM(chs)
suB=build_unet()
# NOTE: same unet weights for both arms (separate instances)

diag_caps=[train_meta[i]['caption'] for i in [0,1,2,5,17,33,60,88]]
diag={'A':{},'B':{}}
for cap in diag_caps:
    e,mk=enc(teA,cap)
    with torch.no_grad():
        outsA=filmA(e,mk); outsB=filmB(e,mk)
    # entropy of learned queries
    B,S,D=e.shape; K=filmA.k_proj(e); V=filmA.v_proj(e)
    Q=filmA.queries.unsqueeze(0).expand(B,-1,-1)
    sc=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(256)
    sc=sc.masked_fill(mk.unsqueeze(1)==0,float('-inf'))
    w=torch.softmax(sc,-1)
    nv=int(mk.sum())
    ent=-(w*torch.log(w.clamp_min(1e-12))).sum(-1)/math.log(nv)
    diag['A'][cap[:40]]={'norm_entropy':float(ent.mean()),'pmax':float(w.max(-1).values.mean()),
        'film_gamma_std':float(torch.cat([o[:, :c].std() for o,c in zip(outsA,chs)]).mean()) if False else float(torch.stack([o[:, :c].std() for o,c in zip(outsA,chs)]).mean()),
        'film_param_rms':float(torch.stack([o.pow(2).mean().sqrt() for o in outsA]).mean())}
    diag['B'][cap[:40]]={'film_param_rms':float(torch.stack([o.pow(2).mean().sqrt() for o in outsB]).mean()),
        'pooled_rms':float(((e*mk.unsqueeze(-1)).sum(1)/mk.sum(1,keepdim=True).clamp_min(1)).pow(2).mean().sqrt())}
with open(OUT/'L20_CONDITIONING_FORWARD_AUDIT.json','w') as f: json.dump(diag,f,indent=2)
entA=[v['norm_entropy'] for v in diag['A'].values()]
print('A init norm_entropy mean',float(np.mean(entA)),'min',min(entA),flush=True)

# caption sensitivity of FiLM params (A vs B)
sens={}
for name,film,su in [('A',filmA,suA),('B',filmB,suB)]:
    su.eval()
    handles=attach_film(su,film,'x')
    g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); t=torch.tensor([int(ts[12])])
    preds=[]
    for cap in diag_caps[:4]:
        with torch.no_grad(): preds.append(forward_eps(su,film,teA,xn,t,cap))
    d=(preds[0]-preds[1]).abs()
    sens[name]={'swap_mean_abs':float(d.mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(preds[0].flatten(),preds[1].flatten(),dim=0))}
    for h in handles: h.remove()
with open(OUT/'L20_INITIAL_ATTENTION_SELECTIVITY.json','w') as f: json.dump({'entropy_A':entA,'sensitivity':sens},f,indent=2)
print('sensitivity:',sens,flush=True)

# ---------- Phase 6: two-arm training on 10-pair subset ----------
man=json.load(open(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics\L17_OVERFIT_DATASET_MANIFEST.json'))
subset=[train_meta[i] for i in man['indices']]

def run_arm(arm):
    su=build_unet(); te=build_te()
    film=LearnedQueryFiLM(chs) if arm=='A' else PooledFiLM(chs)
    handles=attach_film(su,film,arm)
    params=list(su.parameters())+list(film.parameters())
    opt=optim.AdamW(params,lr=0.0001,weight_decay=0.01)
    su.train(); film.train()
    g=torch.Generator().manual_seed(42)
    mets={'steps':[],'evals':{}}
    EVAL_AT={0,100,250,500,750}
    def evalall(tag):
        su.eval(); film.eval(); res={}
        for j,it in enumerate(subset):
            gg=torch.Generator().manual_seed(6000+j); x=torch.randn(1,3,64,64,generator=gg)
            with torch.no_grad():
                ep=forward_eps(su,film,te,x,torch.tensor([int(ts[12])]),it['caption'])
            # quick 8-step DDIM for speed? use full for fidelity on final only; use 12 steps here
            xx=x.clone()
            sub=ts[::2]  # 13 steps approx
            for i in range(len(sub)-1):
                t,tp=sub[i],sub[i+1]
                with torch.no_grad(): ep2=forward_eps(su,film,te,xx,torch.tensor([int(t)]),it['caption'])
                a_t,a_p=ac[t],ac[tp]
                xx=torch.sqrt(a_p)*(xx-torch.sqrt(1-a_t)*ep2)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep2
            img=(xx.clamp(-1,1)+1)/2
            d=IMG/f'{arm}_{tag}_p{j}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
            gt=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
            res[f'p{j}']={'mse_to_gt':float(((img[0].permute(1,2,0).numpy()-gt)**2).mean())}
        su.train(); film.train(); return res
    mets['evals']['0']=evalall('s0')
    t0=time.time(); N=750
    for step in range(N):
        it=subset[step%10]; img=load_img(it)
        t=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
        a=abars[t].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
        pred=forward_eps(su,film,te,xt,t,it['caption'])
        loss=torch.nn.functional.mse_loss(pred,nz)
        opt.zero_grad(); loss.backward(); opt.step()
        s=step+1
        if s%20==0 or s==1:
            gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
            mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
            if s%100==0: print(f'  {arm} s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
        if s in EVAL_AT and s!=0:
            mets['evals'][str(s)]=evalall(f's{s}')
            torch.save({'su':su.state_dict(),'film':film.state_dict(),'step':s},OUT/f'{arm}_step{s}.pt')
    mets['total_sec']=round(time.time()-t0,1)
    return mets,su,film,te

mA,suA2,filmA2,te2=run_arm('A')
with open(OUT/'L20_ARM_A_TRAINING_METRICS.json','w') as f: json.dump(mA,f,indent=2)
print('ARM A done',mA['total_sec'],flush=True)
mB,_,_,_=run_arm('B')
with open(OUT/'L20_ARM_B_TRAINING_METRICS.json','w') as f: json.dump(mB,f,indent=2)
print('ARM B done',mB['total_sec'],flush=True)

# post-training attention selectivity for arm A
suA2.eval(); filmA2.eval()
handles=attach_film(suA2,filmA2,'post')
post_ent=[]
for cap in diag_caps:
    e,mk=enc(te2,cap)
    with torch.no_grad():
        filmA2(e,mk)
        K=filmA2.k_proj(e); V=filmA2.v_proj(e)
        Q=filmA2.queries.unsqueeze(0)
        sc=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(256)
        sc=sc.masked_fill(mk.unsqueeze(1)==0,float('-inf'))
        w=torch.softmax(sc,-1); nv=int(mk.sum())
        ent=-(w*torch.log(w.clamp_min(1e-12))).sum(-1)/math.log(nv)
        post_ent.append(float(ent.mean()))
comp={'armA_post_entropy':post_ent,'armA_entropy_mean':float(np.mean(post_ent))}
# caption binding eval summary
for arm,m in [('A',mA),('B',mB)]:
    for tag,ev in m['evals'].items():
        ms=[v['mse_to_gt'] for v in ev.values()]
        comp.setdefault(arm,{})[tag]={'mse_mean':float(np.mean(ms)),'mse_min':float(np.min(ms))}
with open(OUT/'L20_CAPTION_BINDING_EVALUATION.json','w') as f: json.dump({'comp':comp,'note':'mse_to_gt on 10 overfit pairs; entropy for arm A post-training'},f,indent=2)
with open(OUT/'L20_SPATIAL_QUALITY_COMPARISON.json','w') as f: json.dump(comp,f,indent=2)
with open(OUT/'L20_DIVERSITY_ANALYSIS.json','w') as f:
    import glob
    div={}
    for arm in ['A','B']:
        for tag in ['s250','s750']:
            arrs=[np.array(Image.open(p)).astype(np.float32)/255 for p in sorted(IMG.glob(f'{arm}_{tag}_p*/image.png'))]
            if len(arrs)>1:
                arrs=np.array(arrs); pd=[np.abs(arrs[i]-arrs[j]).mean() for i in range(len(arrs)) for j in range(i+1,len(arrs))]
                div[f'{arm}_{tag}']=float(np.mean(pd))
    json.dump(div,f,indent=2)
print('DONE ALL')
