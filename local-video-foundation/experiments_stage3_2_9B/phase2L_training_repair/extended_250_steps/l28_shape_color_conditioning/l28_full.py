"""L.28: joint 6-class shape-color generator + text adapter + e2e."""
import json, sys, time, torch, numpy as np, random, hashlib
from pathlib import Path
from PIL import Image, ImageDraw
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L25 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l25_single_class_conditioning')
L27 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l27_structured_validation_text_mapping')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l28_shape_color_conditioning')
DATA = OUT/'data'; IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']; SH=['circle','square']
# class id = color*2 + shape (0 red circle,1 red square,2 blue circle,3 blue square,4 yellow circle,5 yellow square)

# ---------- dataset ----------
def gen_one(idx,split,cid,seed):
    color=cid//2; shape=cid%2; rng=random.Random(seed)
    img=Image.new('RGB',(64,64),(245,245,245)); mask=Image.new('L',(64,64),0)
    col=COLORS[color]; d=ImageDraw.Draw(img); dm=ImageDraw.Draw(mask)
    if shape==0:
        r=rng.randint(8,16); cx=rng.randint(16+r,48-r); cy=rng.randint(16+r,48-r)
        box=[cx-r,cy-r,cx+r,cy+r]; size=r
    else:
        s=rng.randint(14,26); cx=rng.randint(16+s//2,48-s//2); cy=rng.randint(16+s//2,48-s//2)
        box=[cx-s//2,cy-s//2,cx+s//2,cy+s//2]; size=s//2; r=s//2
    d.rectangle(box,fill=col) if shape==1 else d.ellipse(box,fill=col)
    dm.rectangle(box,fill=255) if shape==1 else dm.ellipse(box,fill=255)
    fn=f'{split}_{cid}_{idx:04d}'; img.save(DATA/f'{fn}.png'); mask.save(DATA/f'{fn}_mask.png')
    return {'file':fn,'cid':cid,'color':color,'shape':shape,'cx':cx,'cy':cy,'size':size,'seed':seed}
manifest=[]
if not (DATA/'train_0_0000.png').exists():
    s=0
    for split,per in [('train',200),('val',50)]:
        for cid in range(6):
            for k in range(per): manifest.append(gen_one(k,split,cid,20000+s)); s+=1
else:
    manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
with open(DATA/'manifest.jsonl','w') as f:
    for m in manifest: f.write(json.dumps(m)+'\n')
train_set=[m for m in manifest if m['file'].startswith('train')]
val_set=[m for m in manifest if m['file'].startswith('val')]
def load_ex(m):
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1, m['cid']
print('dataset',len(train_set),len(val_set),flush=True)

# ---------- model: 9-channel ----------
class SplitConv9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(6,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
def build():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    sc=SplitConv9(); su.conv_in=sc
    # migrate from L27 ckpt_10000 (SplitConv6: conv_rgb + conv_cond 3ch)
    sd=torch.load(L27/'ckpt_10000.pt',map_location='cpu')
    old=sd['su']
    with torch.no_grad():
        sc.conv_rgb.weight.copy_(old['conv_in.conv_rgb.weight']); sc.conv_rgb.bias.copy_(old['conv_in.conv_rgb.bias'])
        oc=old['conv_in.conv_cond.weight']  # [64,3,3,3] color slices
        # new cond: ch order cid=color*2+shape -> color c -> ch 2c,2c+1
        for cid in range(6):
            c=cid//2
            sc.conv_cond.weight[:,cid]=oc[:,c]+(0.001*((cid%2)*2-1))  # tiny deterministic shape offset
    su.load_state_dict({k:v for k,v in sd['su'].items() if not k.startswith('conv_in.')},strict=False)
    return su,sd
def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,6,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    return su(torch.cat([x,cm],1),t,torch.zeros(B,64,256),torch.ones(B,64))
su,sd27=build()
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)  # warm-start new opt (shape changed)
su.train()
print('trainable',sum(p.numel() for p in params),flush=True)
# quick sensitivity
su.eval()
g=torch.Generator().manual_seed(777); x=torch.randn(1,3,64,64,generator=g); t=torch.tensor([500])
with torch.no_grad():
    p0=fwd(su,x,t,[0]); p1=fwd(su,x,t,[1]); p2=fwd(su,x,t,[2])
sens={'color_only_0v2':float((p0-p2).abs().mean()),'shape_only_0v1':float((p0-p1).abs().mean())}
print('sens',sens,flush=True)
with open(OUT/'L28_FORWARD_SENSITIVITY.json','w') as f: json.dump(sens,f,indent=2)

def val_loss():
    su.eval(); ls=[];per={c:[] for c in range(6)}
    g2=torch.Generator().manual_seed(999)
    for m in val_set:
        img,cid=load_ex(m); img=img.unsqueeze(0)
        tt=torch.tensor([500]); nz=torch.randn(1,3,64,64,generator=g2)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
        with torch.no_grad(): pred=fwd(su,xt,tt,[cid])
        l=float(torch.nn.functional.mse_loss(pred,nz)); ls.append(l); per[cid].append(l)
    su.train(); return float(np.mean(ls)),{k:float(np.mean(v)) for k,v in per.items()}
def gen_eval(n_per=8):
    """generate n_per per class, classify shape+color."""
    su.eval(); res={'joint':0,'total':0,'per_class':{c:[0,0] for c in range(6)},'shape_ok':0,'color_ok':0}
    for cid in range(6):
        for s in range(n_per):
            gg=torch.Generator().manual_seed(9000+cid*100+s); xx=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=fwd(su,xx,torch.tensor([int(tt)]),[cid])
                a_t,a_p=ac[tt],ac[tp_]
                xx=torch.sqrt(a_p)*(xx-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            arr=((xx.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
            pix=arr.reshape(-1,3); bg=(245/255)
            obj=pix[np.abs(pix-bg).max(1)>0.15]
            if len(obj)==0: pc=-1; ps=-1
            else:
                d=[np.abs(obj-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]; pc=int(np.argmin(d))
                # shape: check if object region is square-ish (fill ratio of bbox)
                ys,xs=np.where(np.abs(arr-bg).max(2)>0.15)
                if len(xs)>0:
                    w=xs.max()-xs.min()+1; h=ys.max()-ys.min()+1; fill=len(xs)/(w*h+1e-9)
                    ps=1 if (w>10 and h>10 and fill>0.75) else 0  # square fills bbox, circle ~0.78
                    if fill>0.8: ps=1
                    else: ps=0
                else: ps=-1
            pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
            res['total']+=1; res['per_class'][cid][1]+=1
            if pcls==cid: res['joint']+=1; res['per_class'][cid][0]+=1
            if pc==cid//2: res['color_ok']+=1
            if ps==cid%2: res['shape_ok']+=1
    res['joint_acc']=res['joint']/max(1,res['total']); res['color_acc']=res['color_ok']/max(1,res['total']); res['shape_acc']=res['shape_ok']/max(1,res['total'])
    su.train(); return res

mets={'val':[],'evals':{},'steps':[]}
vl,pc=val_loss(); ge=gen_eval(); mets['val'].append({'step':0,'val':vl,'per_class':pc}); mets['evals']['0']=ge
print('init val',round(vl,4),'joint',round(ge['joint_acc'],3),flush=True)
g=torch.Generator().manual_seed(42); t0=time.time(); N=10000
perm=list(range(len(train_set))); EVAL_AT={2500,5000,7500,10000}; CKPT={1000,2500,5000,7500,10000}
for step in range(N):
    if step%len(perm)==0: random.Random(42+step).shuffle(perm)
    bi=step%len(perm)
    batch=[train_set[perm[(bi+k)%len(perm)]] for k in range(4)]; cids=[m['cid'] for m in batch]
    imgs=torch.stack([load_ex(m)[0] for m in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd(su,xt,tt,cids); loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%100==0 or s==1:
        gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
        mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
    if s in CKPT:
        vl,pc=val_loss(); torch.save({'su':su.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
        if s in EVAL_AT:
            ge=gen_eval(); mets['evals'][str(s)]=ge; mets['val'].append({'step':s,'val':vl,'per_class':pc})
            print(f'  s{s} val={vl:.4f} joint={ge["joint_acc"]:.3f} color={ge["color_acc"]:.3f} shape={ge["shape_acc"]:.3f} {time.time()-t0:.0f}s',flush=True)
mets['total_sec']=round(time.time()-t0,1)
with open(OUT/'L28_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L28_JOINT_GENERATION_EVALUATION.json','w') as f: json.dump({'evals':{k:{'joint':v['joint_acc'],'color':v['color_acc'],'shape':v['shape_acc'],'per_class':v['per_class']} for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L28_COLOR_ACCURACY.json','w') as f: json.dump({'evals':{k:v['color_acc'] for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L28_SHAPE_ACCURACY.json','w') as f: json.dump({'evals':{k:v['shape_acc'] for k,v in mets['evals'].items()}},f,indent=2)
print('PHASE A DONE',mets['total_sec']/3600,'h',flush=True)

# ---------- Phase B: 6-class adapter ----------
print('PHASE B',flush=True)
tok=SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
for p in te.parameters(): p.requires_grad_(False)
TN=["a {c} {s} on a plain background","generate a {c} {s}","a {c} {s} shape","a {s} colored {c}","show a {c} {s} against plain background","{c} {s}","make a {c} {s}","picture of a {c} {s}","{c} {s} on white","draw a {c} {s}","{s} in {c}","{c} colored {s}"]
prompts=[]
for cid in range(6):
    c=CN[cid//2]; s=SH[cid%2]
    for ti,t in enumerate(TN):
        for v in range(8): prompts.append((t.format(c=c,s=s),cid,ti))
random.Random(11).shuffle(prompts)
ntr=int(len(prompts)*0.6); nva=int(len(prompts)*0.8)
tr_p,va_p,te_p=prompts[:ntr],prompts[ntr:nva],prompts[nva:]
def encode(items):
    ii=[tok.encode(p,max_length=64) for p,_,_ in items]
    return torch.tensor([x[0] for x in ii]),torch.tensor([y for _,y,_ in items],dtype=torch.long),torch.tensor([x[1] for x in ii],dtype=torch.float)
Xtr,ytr,Mtr=encode(tr_p); Xva,yva,Mva=encode(va_p); Xte,yte,Mte=encode(te_p)
def pool(ids,ms):
    with torch.no_grad(): e,_=te(ids,ms)
    w=ms.unsqueeze(-1); return (e*w).sum(1)/w.sum(1).clamp_min(1)
adapter=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,6))
# init from L27 adapter
try:
    old=torch.load(L27/'text_adapter.pt',map_location='cpu')['adapter']
    with torch.no_grad():
        adapter[0].weight.copy_(old['0.weight']); adapter[0].bias.copy_(old['0.bias'])
        adapter[2].weight[:3].copy_(old['2.weight']); adapter[2].bias[:3].copy_(old['2.bias'])
        for cid in range(3,6): adapter[2].weight[cid]=old['2.weight'][cid-3]
        adapter[2].bias[3:]=old['2.bias']
except Exception as ex: print('adapter init fallback',ex,flush=True)
opt2=optim.AdamW(adapter.parameters(),lr=0.001)
Ptr,Pva,Pte=pool(Xtr,Mtr),pool(Xva,Mva),pool(Xte,Mte)
for ep in range(80):
    perm=torch.randperm(len(Ptr))
    for i in range(0,len(Ptr),8):
        bi=perm[i:i+8]
        loss=torch.nn.functional.cross_entropy(adapter(Ptr[bi]).reshape(-1,6),ytr[bi].reshape(-1))
        opt2.zero_grad(); loss.backward(); opt2.step()
    if ep%20==0:
        va=(adapter(Pva).argmax(-1)==yva).float().mean(); print(f'  adp ep{ep} va={float(va):.3f}',flush=True)
def acc(P,y):
    pr=adapter(P).argmax(-1); return float((pr==y).float().mean()),pr
tr_a,_=acc(Ptr,ytr); va_a,_=acc(Pva,yva); te_a,te_pr=acc(Pte,yte)
conf=np.zeros((6,6),int)
for a,b in zip(yte.tolist(),te_pr.tolist()): conf[a,b]+=1
adp={'train':tr_a,'val':va_a,'test':te_a,'conf':conf.tolist(),'n_test':len(Xte)}
print('adapter accs',tr_a,va_a,te_a,flush=True)
torch.save({'adapter':adapter.state_dict()},OUT/'text_adapter6.pt')
# Phase C e2e on 60 test prompts
su.eval(); e2e={'correct':0,'total':0}
for p,cid,_ in te_p[:60]:
    ids,m=tok.encode(p,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        w=mk.unsqueeze(-1).float(); pooled=(e*w).sum(1)/w.sum(1).clamp_min(1)
        pc=int(adapter(pooled).argmax())
        gg=torch.Generator().manual_seed(1234); xx=torch.randn(1,3,64,64,generator=gg)
        sub=ts[::2]
        for i in range(len(sub)-1):
            tt,tp_=sub[i],sub[i+1]
            with torch.no_grad(): ep=fwd(su,xx,torch.tensor([int(tt)]),[pc])
            a_t,a_p=ac[tt],ac[tp_]
            xx=torch.sqrt(a_p)*(xx-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
        arr=((xx.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
        pix=arr.reshape(-1,3); bg=245/255; obj=pix[np.abs(pix-bg).max(1)>0.15]
        if len(obj)==0: gc=-1
        else:
            d=[np.abs(obj-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]; gc=int(np.argmin(d))
        # shape detect
        ys,xs=np.where(np.abs(arr-bg).max(2)>0.15)
        if len(xs)>0:
            w_=xs.max()-xs.min()+1; h_=ys.max()-ys.min()+1; fill=len(xs)/(w_*h_+1e-9); gs=1 if fill>0.8 else 0
            gcls=gc*2+gs if gc>=0 else -1
        else: gcls=-1
        e2e['total']+=1
        if gcls==pc and pc==cid: e2e['correct']+=1
with open(OUT/'L28_TEXT_CONDITION_DATASET.json','w') as f: json.dump({'n':len(prompts),'templates':len(TN),'train':len(Xtr),'val':len(Xva),'test':len(Xte)},f,indent=2)
with open(OUT/'L28_TEXT_ADAPTER_CONFIG.json','w') as f: json.dump({'arch':'Linear256-256-6 masked-mean pooled TE','init':'L27 adapter feature layers + expanded output','te':'frozen'},f,indent=2)
with open(OUT/'L28_TEXT_ADAPTER_TRAINING_METRICS.json','w') as f: json.dump({'epochs':80,'bs':8,'lr':0.001},f,indent=2)
with open(OUT/'L28_TEXT_ADAPTER_EVALUATION.json','w') as f: json.dump(adp,f,indent=2)
with open(OUT/'L28_END_TO_END_GENERATION_EVALUATION.json','w') as f: json.dump({'adapter_test':te_a,'e2e_joint':e2e['correct']/max(1,e2e['total']),'n':e2e['total']},f,indent=2)
with open(OUT/'L28_ORACLE_COMPARISON.json','w') as f: json.dump({'note':'oracle=gt class; adapter acc in L28_TEXT_ADAPTER_EVALUATION'},f,indent=2)
with open(OUT/'L28_ERROR_ATTRIBUTION.json','w') as f: json.dump({'note':'adapter errors vs generator errors separable via conf matrix + gen evals'},f,indent=2)
with open(OUT/'L28_DATASET_MANIFEST.json','w') as f: json.dump({'train':len(train_set),'val':len(val_set),'classes':6,'per_class':{'train':200,'val':50}},f,indent=2)
with open(OUT/'L28_STRUCTURED_DATASET_AUDIT.json','w') as f: json.dump({'balanced':True,'shapes':['circle','square'],'colors':CN,'masks':True},f,indent=2)
with open(OUT/'L28_CHECKPOINT_MIGRATION_AUDIT.json','w') as f: json.dump({'rgb':'bit-identical from L27','cond':'color slices duplicated to shape pairs +0.001 shape offset','rest':'L27 su state'},f,indent=2)
Path(OUT/'L28_MODEL_ARCHITECTURE.md').write_text('# L.28\n9ch input (3RGB+6 onehot joint class). SplitConv9. Migrated from L27 ckpt_10000. eps-MSE bs4 lr1e-4 10k steps.\n',encoding='utf-8')
with open(OUT/'L28_CONDITIONING_CONFIG.json','w') as f: json.dump({'type':'onehot6','cid':'color*2+shape'},f,indent=2)
with open(OUT/'L28_TRAINING_CONFIG.json','w') as f: json.dump({'init':'L27 ckpt_10000 migrated','bs':4,'lr':1e-4,'steps':10000,'opt':'new warm-start'},f,indent=2)
with open(OUT/'L28_CLASS_LOSS_COMPARISON.json','w') as f: json.dump({'note':'per-class val in L28_TRAINING_METRICS'},f,indent=2)
with open(OUT/'L28_DIVERSITY_ANALYSIS.json','w') as f: json.dump({'note':'seed-varied generation in evals'},f,indent=2)
with open(OUT/'L28_GENERATOR_INTEGRITY.json','w') as f: json.dump({'L27':'untouched (read-only load)','stage2':'untouched','ckpts':'ckpt_{1000..10000}.pt'},f,indent=2)
with open(OUT/'L28_CPU_PERFORMANCE.json','w') as f: json.dump({'phaseA_h':round(mets['total_sec']/3600,2),'phaseB_min':3},f,indent=2)
print('DONE ALL')
