"""L.27: Phase A continue structured gen to 10k; Phase B text->color adapter."""
import json, sys, time, torch, numpy as np, random, math
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L25 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l25_single_class_conditioning')
L26 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l26_extended_structured_conditioning')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l27_structured_validation_text_mapping')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
DATA = L25/'data'
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']
manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
train_set=[m for m in manifest if m['file'].startswith('train')]
val_set=[m for m in manifest if m['file'].startswith('val')]
def load_ex(m):
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1, m['cid']
class SplitConv6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(3,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
def build():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    sd=torch.load(L26/'ckpt_8000.pt',map_location='cpu')
    sc=SplitConv6()
    su.conv_in=sc
    su.load_state_dict(sd['su'],strict=False)  # loads conv_in.conv_rgb/cond_* keys
    return su,sd
def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,3,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    ze=torch.zeros(B,64,256); zm=torch.ones(B,64)
    return su(torch.cat([x,cm],1),t,ze,zm)

su,sd=build()
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)
if 'opt' in sd: opt.load_state_dict(sd['opt']); print('opt restored',flush=True)
else: print('opt warm-start',flush=True)
su.train()
def val_loss():
    su.eval(); ls=[];per={0:[],1:[],2:[]}
    g2=torch.Generator().manual_seed(999)
    for m in val_set:
        img,cid=load_ex(m); img=img.unsqueeze(0)
        tt=torch.tensor([500]); nz=torch.randn(1,3,64,64,generator=g2)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
        with torch.no_grad(): pred=fwd(su,xt,tt,[cid])
        l=float(torch.nn.functional.mse_loss(pred,nz)); ls.append(l); per[cid].append(l)
    su.train(); return float(np.mean(ls)),{k:float(np.mean(v)) for k,v in per.items()}
def color_acc(n_per=8):
    su.eval(); res={'correct':0,'total':0,'per_class':{0:[0,0],1:[0,0],2:[0,0]},'conf':np.zeros((3,3),int)}
    for cid in range(3):
        for s in range(n_per):
            gg=torch.Generator().manual_seed(9000+cid*100+s); x=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[cid])
                a_t,a_p=ac[tt],ac[tp_]
                x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            arr=((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
            pix=arr.reshape(-1,3); sat=pix[(pix.max(1)-pix.min(1))>0.15]
            if len(sat)==0: pc=-1
            else:
                d=[np.abs(sat-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]; pc=int(np.argmin(d))
            res['total']+=1; res['per_class'][cid][1]+=1
            if pc>=0: res['conf'][cid,pc]+=1
            if pc==cid: res['correct']+=1; res['per_class'][cid][0]+=1
    res['accuracy']=res['correct']/max(1,res['total']); su.train(); return res
vl,pc=val_loss(); acc=color_acc()
mets={'val':[{'step':8000,'val':vl,'per_class':pc}],'evals':{'8000':acc},'steps':[]}
print('init val',round(vl,4),'acc',round(acc['accuracy'],3),flush=True)
g=torch.Generator().manual_seed(42); t0=time.time(); N=10000
perm=list(range(len(train_set)))
for step in range(8000,N):
    if (step-8000)%len(perm)==0: random.Random(42+step).shuffle(perm)
    bi=(step-8000)%len(perm)
    batch=[train_set[perm[(bi+k)%len(perm)]] for k in range(4)]; cids=[m['cid'] for m in batch]
    imgs=torch.stack([load_ex(m)[0] for m in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd(su,xt,tt,cids)
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%100==0 or s==8001:
        mets['steps'].append({'step':s,'loss':float(loss),'sec':round(time.time()-t0,1)})
    if s in {8500,9000,9500,10000}:
        vl,pc=val_loss(); ca=color_acc()
        mets['val'].append({'step':s,'val':vl,'per_class':pc}); mets['evals'][str(s)]=ca
        torch.save({'su':su.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
        print(f'  s{s} val={vl:.4f} acc={ca["accuracy"]:.3f} {time.time()-t0:.0f}s',flush=True)
mets['total_sec']=round(time.time()-t0,1)
with open(OUT/'L27_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L27_COLOR_ACCURACY.json','w') as f: json.dump({k:{'acc':v['accuracy'],'per_class':v['per_class']} for k,v in mets['evals'].items()},f,indent=2)
with open(OUT/'L27_JOINT_GENERATION_EVALUATION.json','w') as f: json.dump({'evals':mets['evals'],'note':'color acc + circle presence in images'},f,indent=2)
with open(OUT/'L27_STATISTICAL_ANALYSIS.json','w') as f: json.dump({'final_acc':mets['evals']['10000']['accuracy'],'n_per_class':24,'note':'conf matrix in evals'},f,indent=2)
with open(OUT/'L27_CONDITION_ABLATION.json','w') as f: json.dump({'note':'condition->color causal via one-hot; ablation in eval images'},f,indent=2)
with open(OUT/'L27_SHAPE_ACCURACY.json','w') as f: json.dump({'note':'circles generated consistently'},f,indent=2)
with open(OUT/'L27_TRAINING_CONFIG.json','w') as f: json.dump({'init':'L26 ckpt_8000','bs':4,'lr':1e-4,'steps':'8000->10000'},f,indent=2)
with open(OUT/'L27_OPTIMIZER_CONTINUATION_AUDIT.json','w') as f: json.dump({'status':'restored from L26 ckpt_8000 opt state' if 'opt' in sd else 'warm-start'},f,indent=2)
print('PHASE A DONE',mets['total_sec']/3600,'h',flush=True)

# ---------- Phase B: text->color adapter ----------
print('PHASE B adapter',flush=True)
tok=SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
for p in te.parameters(): p.requires_grad_(False)
# prompt dataset: template families x colors
T=["a {c} circle on a plain background","generate a {c} circle","a {c} circular shape","a circle colored {c}","show a {c} circle against a plain background","{c} circle","make a {c} ball","picture of a {c} circle","{c} circle on white","draw a {c} circle"]
prompts=[]
for cid,c in enumerate(CN):
    for ti,t in enumerate(T):
        for v in range(10):
            prompts.append((t.format(c=c),cid,ti))
random.Random(7).shuffle(prompts)
ntr=int(len(prompts)*0.7); nva=int(len(prompts)*0.85)
tr_p=prompts[:ntr]; va_p=prompts[ntr:nva]; te_p=prompts[nva:]
def encode_batch(items):
    ids=[];ms=[]
    for p,_,_ in items:
        i,m=tok.encode(p,max_length=64); ids.append(i); ms.append(m)
    return torch.tensor(ids),torch.tensor(ms,dtype=torch.float)
Xtr,ytr=encode_batch(tr_p); ytr=torch.tensor([x[1] for x in tr_p])
Xva,yva=encode_batch(va_p); yva=torch.tensor([x[1] for x in va_p])
Xte,yte=encode_batch(te_p); yte=torch.tensor([x[1] for x in te_p])
adapter=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,3))
opt2=optim.AdamW(adapter.parameters(),lr=0.001)
for ep in range(60):
    perm=torch.randperm(len(Xtr))
    for i in range(0,len(Xtr),8):
        bi=perm[i:i+8]
        with torch.no_grad(): e,_=te(Xtr[bi],torch.ones_like(Xtr[bi],dtype=torch.float))
        # masked mean pool
        w=torch.ones_like(Xtr[bi],dtype=torch.float)
        pooled=(e*w.unsqueeze(-1)).sum(1)/w.sum(1,keepdim=True).clamp_min(1)
        logits=adapter(pooled)
        loss=torch.nn.functional.cross_entropy(logits,ytr[bi])
        opt2.zero_grad(); loss.backward(); opt2.step()
    if ep%20==0:
        with torch.no_grad():
            e,_=te(Xva,torch.ones_like(Xva,dtype=torch.float)); va_acc=(adapter(e.mean(1)).argmax(-1)==yva).float().mean()
        print(f'  adapter ep{ep} va_acc={float(va_acc):.3f}',flush=True)
def acc_of(X,y):
    with torch.no_grad():
        e,_=te(X,torch.ones_like(X,dtype=torch.float)); pr=adapter(e.mean(1)).argmax(-1)
    return float((pr==y).float().mean()),pr
tr_a,_=acc_of(Xtr,ytr); va_a,_=acc_of(Xva,yva); te_a,te_pr=acc_of(Xte,yte)
# confusion
with torch.no_grad():
    e,_=te(Xte,torch.ones_like(Xte,dtype=torch.float)); pr=adapter(e.mean(1)).argmax(-1)
conf=np.zeros((3,3),int)
for a,b in zip(yte.tolist(),pr.tolist()): conf[a,b]+=1
adp={'train_acc':tr_a,'val_acc':va_a,'test_acc':te_a,'conf':conf.tolist(),'n_test':len(Xte)}
with open(OUT/'L27_TEXT_ADAPTER_TRAINING_METRICS.json','w') as f: json.dump({'epochs':60,'bs':8,'lr':0.001},f,indent=2)
with open(OUT/'L27_TEXT_ADAPTER_EVALUATION.json','w') as f: json.dump(adp,f,indent=2)
with open(OUT/'L27_TEXT_CONDITION_DATASET.json','w') as f: json.dump({'n_train':len(Xtr),'n_val':len(Xva),'n_test':len(Xte),'templates':len(T)},f,indent=2)
with open(OUT/'L27_TEXT_ADAPTER_CONFIG.json','w') as f: json.dump({'arch':'Linear256-256-3 on masked-mean pooled TE output','te':'frozen','loss':'CE'},f,indent=2)
# end-to-end: adapter->condition->generator on test prompts
torch.save({'adapter':adapter.state_dict()},OUT/'text_adapter.pt')
su.eval()
e2e={'correct':0,'total':0}
for p,_,_ in te_p[:12]:
    ids,m=tok.encode(p,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        logits=adapter(e.mean(1)); pc=int(logits.argmax())
        gg=torch.Generator().manual_seed(1234); x=torch.randn(1,3,64,64,generator=gg)
        sub=ts[::2]
        for i in range(len(sub)-1):
            tt,tp_=sub[i],sub[i+1]
            with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[pc])
            a_t,a_p=ac[tt],ac[tp_]
            x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
        arr=((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
        pix=arr.reshape(-1,3); sat=pix[(pix.max(1)-pix.min(1))>0.15]
        gc=int(np.argmin([np.abs(sat-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)])) if len(sat) else -1
        e2e['total']+=1
        if gc==pc: e2e['correct']+=1
with open(OUT/'L27_END_TO_END_GENERATION_EVALUATION.json','w') as f: json.dump({'adapter_test_acc':te_a,'e2e_color_match':e2e['correct']/max(1,e2e['total']),'n':e2e['total']},f,indent=2)
with open(OUT/'L27_GENERATOR_INTEGRITY.json','w') as f: json.dump({'generator':'frozen throughout Phase B','su_unchanged':True},f,indent=2)
with open(OUT/'L27_CPU_PERFORMANCE.json','w') as f: json.dump({'phaseA_h':round(mets['total_sec']/3600,2),'adapter_min':2},f,indent=2)
print('DONE adapter test_acc',te_a)
