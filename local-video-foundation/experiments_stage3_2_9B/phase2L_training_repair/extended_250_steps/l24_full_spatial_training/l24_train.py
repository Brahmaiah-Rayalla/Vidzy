"""L.24: full end-to-end spatial training, direct text channels, 10k steps bs4."""
import json, sys, time, torch, numpy as np, hashlib, random
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L21 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l21_direct_text_channel_experiment')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet, CrossAttentionBlock
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l24_full_spatial_training')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
SEQ,FEAT,NCH=64,8,512
meta=json.load(open(base/'data/text_image/train_metadata.json'))
train_meta=meta[:700]; val_meta=meta[700:800]
def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1
def build_te():
    te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
    for p in te.parameters(): p.requires_grad_(False)
    return te
class SplitConv(torch.nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,w.shape[0],3,padding=1)
        self.conv_text=torch.nn.Conv2d(NCH,w.shape[0],3,padding=1,bias=False)
        with torch.no_grad():
            self.conv_rgb.weight.copy_(w[:,:3]); self.conv_rgb.bias.copy_(b); self.conv_text.weight.copy_(w[:,3:])
    def forward(self,xc): return self.conv_rgb(xc[:,:3])+self.conv_text(xc[:,3:])

def build_model():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    old=su.conv_in
    new=torch.nn.Conv2d(3+NCH,old.out_channels,3,padding=1)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias); new.weight[:,:3].copy_(old.weight); new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new
    sd=torch.load(L21/'A_step750.pt',map_location='cpu')
    su.load_state_dict(sd['su'])   # transfers learned UNet+conv state? NO - spec: Stage2.1 UNet + L21 cond modules only
    return su
# CORRECT init per spec: Stage2.1 UNet weights, L21 only for tokproj + conv_text slice
def build_model2():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    old=su.conv_in
    new=torch.nn.Conv2d(3+NCH,old.out_channels,3,padding=1)
    sd21=torch.load(L21/'A_step750.pt',map_location='cpu')
    w21=sd21['su']['conv_in.weight']; b21=sd21['su']['conv_in.bias']
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias)  # Stage2.1 bias
        new.weight[:,:3].copy_(old.weight)            # Stage2.1 RGB
        new.weight[:,3:].copy_(w21[:,3:])             # L21 text slice
    su.conv_in=new
    tp=torch.nn.Linear(256,FEAT); tp.load_state_dict(sd21['tokproj'])
    sc=SplitConv(new.weight,new.bias); su.conv_in=sc
    return su,tp

def batch_text_channels(te,tp,caps):
    B=len(caps); tc=torch.zeros(B,NCH,64,64); es=[];mks=[]
    for i,cap in enumerate(caps):
        ids,m=tok.encode(cap,max_length=64)
        with torch.no_grad():
            e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        tpp=tp(e)*mk.unsqueeze(-1).float()
        tc[i]=tpp.reshape(-1,1,1).expand(NCH,64,64) if False else tpp.reshape(1,-1).expand(64*64,NCH).T.reshape(NCH,64,64)
        es.append(e);mks.append(mk)
    # legacy cross-attn bypass: zeros emb, ones mask
    ze=torch.zeros(B,64,256); zm=torch.ones(B,64)
    return tc,ze,zm

def fwd_batch(su,tp,te,x,t,caps):
    tc,ze,zm=batch_text_channels(te,tp,caps)
    return su(torch.cat([x,tc],1),t,ze,zm)

# eval prompts
EP=["a red circle on a plain white background","a blue square on a plain white background",
    "a yellow triangle on a plain white background","a green circle at the center",
    "a red square on a plain white background","a small blue circle on the left",
    "a large red triangle on the right","a purple square in the middle",
    train_meta[100]['caption'],train_meta[200]['caption'],val_meta[0]['caption'],val_meta[1]['caption']]

def ddim_eval(su,tp,te,prompt,seed=1234,steps=20):
    su.eval()
    g=torch.Generator().manual_seed(seed); x=torch.randn(1,3,64,64,generator=g)
    tc,ze,zm=batch_text_channels(te,tp,[prompt])
    sub=ts[::max(1,len(ts)//steps)]
    for i in range(len(sub)-1):
        t,tp_=sub[i],sub[i+1]
        with torch.no_grad(): ep=su(torch.cat([x,tc],1),torch.tensor([int(t)]),ze,zm)
        a_t,a_p=ac[t],ac[tp_]
        x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
    su.train(); return (x.clamp(-1,1)+1)/2

def eval_checkpoint(tag):
    su.eval(); tp.eval(); te=build_te()
    for pi,p in enumerate(EP):
        img=ddim_eval(su,tp,te,p)
        d=IMG/f'{tag}_p{pi}'; d.mkdir(exist_ok=True)
        Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
    su.train()

# ---------- build & preflight ----------
print('building',flush=True)
su,tp=build_model2(); te=build_te()
# freeze legacy cross-attn params + TE; train everything else
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
    else: p.requires_grad_(True)
for p in tp.parameters(): p.requires_grad_(True)
params=[p for p in su.parameters() if p.requires_grad]+list(tp.parameters())
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)
n_train=sum(p.numel() for p in params)
print('trainable params:',n_train,flush=True)
# quick pilot 30 steps
su.train(); tp.train()
g=torch.Generator().manual_seed(42)
t0=time.time()
for step in range(30):
    idx=[random.Random(42+step*4+k).randrange(700) for k in range(4)]
    batch=[train_meta[i] for i in idx]
    imgs=torch.stack([load_img(it) for it in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd_batch(su,tp,te,xt,tt,[it['caption'] for it in batch])
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    if step%10==0: print(f'pilot s{step} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
pt=(time.time()-t0)/30
print('pilot step time',round(pt,2),'s -> est',round(pt*10000/3600,1),'h',flush=True)
with open(OUT/'L24_CPU_PILOT_BENCHMARK.json','w') as f: json.dump({'step_s':pt,'est_hours':pt*10000/3600,'bs':4},f,indent=2)

# ---------- main run ----------
print('MAIN RUN 10000 steps',flush=True)
# resume from latest checkpoint if present
existing=sorted(OUT.glob('ckpt_*.pt'))
start_step=0
if existing:
    latest=existing[-1]; sd=torch.load(latest,map_location='cpu')
    su.load_state_dict(sd['su']); tp.load_state_dict(sd['tokproj']); opt.load_state_dict(sd['opt'])
    start_step=sd['step']
    print('RESUMED from',latest.name,'step',start_step,flush=True)
    mets={'steps':[],'evals':{},'val':[{'step':start_step,'val':-1}]}
else:
    mets={'steps':[],'evals':{}}
EVAL_AT={0,1000,2000,4000,6000,8000,10000}
CKPT_AT=set(range(0,10001,500))
def val_loss():
    su.eval(); tp.eval(); te=build_te(); losses=[]
    g2=torch.Generator().manual_seed(999)
    for j in range(8):
        it=val_meta[j]; img=load_img(it).unsqueeze(0)
        tt=torch.tensor([500])
        nz=torch.randn(1,3,64,64,generator=g2)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
        with torch.no_grad(): pred=fwd_batch(su,tp,te,xt,tt,[it['caption']])
        losses.append(float(torch.nn.functional.mse_loss(pred,nz)))
    su.train(); tp.train(); return float(np.mean(losses))
if start_step==0:
    mets['val']=[{'step':0,'val':val_loss()}]
    eval_checkpoint('s0')
t0=time.time(); N=10000
for step in range(start_step,N):
    idx=[random.Random(42+step*4+k).randrange(700) for k in range(4)]
    batch=[train_meta[i] for i in idx]
    imgs=torch.stack([load_img(it) for it in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd_batch(su,tp,te,xt,tt,[it['caption'] for it in batch])
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%50==0:
        gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
        mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
        if s%500==0:
            vl=val_loss(); mets['val'].append({'step':s,'val':vl})
            print(f'  s{s} loss={float(loss):.4f} val={vl:.4f} {time.time()-t0:.0f}s',flush=True)
            torch.save({'su':su.state_dict(),'tokproj':tp.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
    if s in EVAL_AT:
        eval_checkpoint(f's{s}')
# final caption binding eval
su.eval(); tp.eval(); te=build_te()
bind={}
it=subset0=train_meta[0]
gg=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=gg); tt=torch.tensor([int(ts[12])])
with torch.no_grad():
    for capname,cap in [('correct',it['caption']),('wrong',train_meta[5]['caption'])]:
        tc,ze,zm=batch_text_channels(te,tp,[cap])
        ep=su(torch.cat([xn,tc],1),tt,ze,zm)
        bind[capname]={'pred_std':float(ep.std())}
    tc0,_,_=batch_text_channels(te,tp,[it['caption']]); tc0=torch.zeros_like(tc0)
    ez=torch.zeros(1,64,256); zm=torch.ones(1,64)
    epz=su(torch.cat([xn,tc0],1),tt,ez,zm)
    bind['zero_text']={'pred_std':float(epz.std())}
mets['total_sec']=round(time.time()-t0,1); mets['binding']=bind
with open(OUT/'L24_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L24_VALIDATION_COMPARISON.json','w') as f: json.dump({'val':mets['val']},f,indent=2)
with open(OUT/'L24_CAPTION_BINDING_EVALUATION.json','w') as f: json.dump(bind,f,indent=2)
print('DONE',mets['total_sec']/3600,'h')
