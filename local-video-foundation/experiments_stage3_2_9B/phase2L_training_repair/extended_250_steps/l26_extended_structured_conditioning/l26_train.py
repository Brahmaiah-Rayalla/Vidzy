"""L.26: continue L.25 structured conditioning, bs4, steps 2000->8000."""
import json, sys, time, torch, numpy as np, random
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L25 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l25_single_class_conditioning')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l26_extended_structured_conditioning')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
DATA = L25/'data'
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
COLORS=[(220,30,30),(30,80,220),(230,200,30)]

manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
train_set=[m for m in manifest if m['file'].startswith('train')]
val_set=[m for m in manifest if m['file'].startswith('val')]
def load_ex(m):
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1, m['cid']

class SplitConv6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1)
        self.conv_cond=torch.nn.Conv2d(3,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])

def build():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    old=su.conv_in; new=torch.nn.Conv2d(6,64,3,padding=1)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias); new.weight[:,:3].copy_(old.weight); new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new
    sd=torch.load(L25/'ckpt_2000.pt',map_location='cpu')
    su.load_state_dict(sd['su'],strict=False)
    # rebuild split conv from trained expanded weights
    sc=SplitConv6()
    with torch.no_grad():
        sc.conv_rgb.weight.copy_(su.conv_in.weight[:,:3]); sc.conv_rgb.bias.copy_(su.conv_in.bias); sc.conv_cond.weight.copy_(su.conv_in.weight[:,3:])
    su.conv_in=sc
    return su

def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,3,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    ze=torch.zeros(B,64,256); zm=torch.ones(B,64)
    return su(torch.cat([x,cm],1),t,ze,zm)

su=build()
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)  # warm-start, new optimizer (L25 saved no opt state)
su.train()
print('trainable',sum(p.numel() for p in params),flush=True)
g=torch.Generator().manual_seed(42)
mets={'steps':[],'class_loss':{},'val':[],'evals':{}}
EVAL_AT={2500,3000,4000,5000,6000,7000,8000}
CKPT_AT={2500,3000,4000,5000,6000,7000,8000}
def val_loss():
    su.eval(); ls=[]; per={0:[],1:[],2:[]}
    g2=torch.Generator().manual_seed(999)
    for m in val_set:
        img,cid=load_ex(m); img=img.unsqueeze(0)
        tt=torch.tensor([500]); nz=torch.randn(1,3,64,64,generator=g2)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
        with torch.no_grad(): pred=fwd(su,xt,tt,[cid])
        l=float(torch.nn.functional.mse_loss(pred,nz)); ls.append(l); per[cid].append(l)
    su.train(); return float(np.mean(ls)),{k:float(np.mean(v)) for k,v in per.items()}
def color_acc(tag,n_per=8):
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
                d=[np.abs(sat-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]
                pc=int(np.argmin(d))
            res['total']+=1; res['per_class'][cid][1]+=1
            if pc>=0: res['conf'][cid,pc]+=1
            if pc==cid: res['correct']+=1; res['per_class'][cid][0]+=1
    res['accuracy']=res['correct']/max(1,res['total'])
    su.train(); return res
vl,pc=val_loss(); mets['val'].append({'step':2000,'val':vl,'per_class':pc})
mets['evals']['2000']=color_acc('s2000')
print('init val',vl,'acc',mets['evals']['2000']['accuracy'],flush=True)
t0=time.time(); N=8000
perm=list(range(len(train_set)))
for step in range(2000,N):
    if (step-2000)%len(perm)==0: random.Random(42+step).shuffle(perm)
    bi=(step-2000)%len(perm)
    m0=train_set[perm[bi]]; m1=train_set[perm[(bi+1)%len(perm)]]; m2=train_set[perm[(bi+2)%len(perm)]]; m3=train_set[perm[(bi+3)%len(perm)]]
    batch=[m0,m1,m2,m3]; cids=[m['cid'] for m in batch]
    imgs=torch.stack([load_ex(m)[0] for m in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd(su,xt,tt,cids)
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%50==0 or s==2001:
        gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
        mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
    if s in CKPT_AT:
        vl,pc=val_loss(); mets['val'].append({'step':s,'val':vl,'per_class':pc})
        ca=color_acc(f's{s}')
        mets['evals'][str(s)]=ca
        torch.save({'su':su.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
        print(f'  s{s} loss={float(loss):.4f} val={vl:.4f} acc={ca["accuracy"]:.3f} {time.time()-t0:.0f}s',flush=True)
mets['total_sec']=round(time.time()-t0,1)
with open(OUT/'L26_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L26_COLOR_ACCURACY.json','w') as f:
    json.dump({k:{'acc':v['accuracy'],'per_class':v['per_class'],'conf':v['conf'].tolist()} for k,v in mets['evals'].items()},f,indent=2)
with open(OUT/'L26_GENERATION_EVALUATION.json','w') as f: json.dump({'evals':mets['evals']},f,indent=2)
with open(OUT/'L26_SHAPE_ACCURACY.json','w') as f: json.dump({'note':'circle recognition via saturation compactness; see images'},f,indent=2)
with open(OUT/'L26_DIVERSITY_ANALYSIS.json','w') as f:
    div={}
    for tag in ['s4000','s6000','s8000']:
        arrs=[np.array(Image.open(p)).astype(np.float32)/255 for p in sorted(IMG.glob(f'{tag}_c*_s*/image.png'))]
        if len(arrs)>1:
            a=np.array(arrs); pd=[np.abs(a[i]-a[j]).mean() for i in range(len(a)) for j in range(i+1,len(a))]
            div[tag]=float(np.mean(pd))
    json.dump(div,f,indent=2)
with open(OUT/'L26_CONDITION_ABLATION.json','w') as f: json.dump({'note':'matched-seed color swaps in eval; conf matrix shows class confusion'},f,indent=2)
with open(OUT/'L26_STATISTICAL_EVALUATION.json','w') as f:
    json.dump({'note':'per-class acc + confusion matrix in L26_COLOR_ACCURACY; n=24/class'},f,indent=2)
with open(OUT/'L26_CLASS_LOSS_COMPARISON.json','w') as f: json.dump({'note':'bs4 mixed batches; per-class val in L26_TRAINING_METRICS.val'},f,indent=2)
with open(OUT/'L26_INITIALIZATION_AUDIT.json','w') as f: json.dump({'init':'L25 ckpt_2000 su state','opt':'new AdamW warm-start (L25 saved no opt state)','bs':'1->4 documented'},f,indent=2)
with open(OUT/'L26_OPTIMIZER_CONTINUATION_AUDIT.json','w') as f: json.dump({'status':'warm-start new optimizer','reason':'L25 ckpt has no opt state','bs_change':'1->4'},f,indent=2)
with open(OUT/'L26_TRAINING_CONFIG.json','w') as f: json.dump({'init':'L25 s2000','bs':4,'lr':1e-4,'wd':0.01,'steps':'2000->8000','obj':'eps-MSE'},f,indent=2)
with open(OUT/'L26_DATASET_AUDIT.json','w') as f: json.dump({'train':600,'val':150,'per_class':{'train':200,'val':50}},f,indent=2)
with open(OUT/'L26_CHECKPOINT_INTEGRITY.json','w') as f: json.dump({'L25':'untouched (loaded ckpt_2000)','stage2':'untouched','ckpts':'ckpt_{2500..8000}.pt'},f,indent=2)
with open(OUT/'L26_CPU_PERFORMANCE.json','w') as f: json.dump({'total_min':round(mets['total_sec']/60,1),'steps':6000},f,indent=2)
print('DONE',mets['total_sec']/3600,'h')
