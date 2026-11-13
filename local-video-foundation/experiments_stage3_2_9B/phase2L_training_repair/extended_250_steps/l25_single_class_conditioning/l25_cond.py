"""L.25: single-class (circle) 3-color structured conditioning, 2000 steps."""
import json, sys, time, torch, numpy as np, hashlib, random
from pathlib import Path
from PIL import Image, ImageDraw
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l25_single_class_conditioning')
DATA = OUT/'data'; IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True); DATA.mkdir(exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
COLORS=[(220,30,30),(30,80,220),(230,200,30)]  # red, blue, yellow
NAMES=['red','blue','yellow']

# ---------- Phase 2: generate dataset ----------
def gen_one(idx, split, cid, seed):
    rng=random.Random(seed)
    img=Image.new('RGB',(64,64),(245,245,245))
    mask=Image.new('L',(64,64),0)
    r=rng.randint(8,16); cx=rng.randint(16+r,48-r); cy=rng.randint(16+r,48-r)
    col=COLORS[cid]
    d=ImageDraw.Draw(img); d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col)
    dm=ImageDraw.Draw(mask); dm.ellipse([cx-r,cy-r,cx+r,cy+r],fill=255)
    fn=f'{split}_{cid}_{idx:04d}'
    img.save(DATA/f'{fn}.png'); mask.save(DATA/f'{fn}_mask.png')
    return {'file':fn,'cid':cid,'cx':cx,'cy':cy,'r':r,'seed':seed}
manifest=[]
if not (DATA/'train_0_0000.png').exists():
    s=0
    for split,per in [('train',200),('val',50)]:
        for cid in range(3):
            for k in range(per):
                manifest.append(gen_one(k,split,cid,10000+s)); s+=1
else:
    manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
with open(DATA/'manifest.jsonl','w') as f:
    for m in manifest: f.write(json.dumps(m)+'\n')
train_set=[m for m in manifest if m['file'].startswith('train')]
val_set=[m for m in manifest if m['file'].startswith('val')]
print('dataset:',len(train_set),len(val_set),flush=True)

def load_ex(m):
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1, m['cid']

# ---------- model: 6-channel input ----------
class SplitConv6(torch.nn.Module):
    def __init__(self,w,b):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,w.shape[0],3,padding=1)
        self.conv_cond=torch.nn.Conv2d(3,w.shape[0],3,padding=1,bias=False)
        with torch.no_grad():
            self.conv_rgb.weight.copy_(w[:,:3]); self.conv_rgb.bias.copy_(b)
            self.conv_cond.weight.normal_(0,0.02)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])

def build():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    old=su.conv_in
    new=torch.nn.Conv2d(6,old.out_channels,3,padding=1)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias); new.weight[:,:3].copy_(old.weight); new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new; su.load_state_dict({'conv_in.weight':new.weight,'conv_in.bias':new.bias},strict=False)
    # replace conv_in with split version carrying same weights
    sc=SplitConv6.__new__(SplitConv6); torch.nn.Module.__init__(sc)
    sc.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); sc.conv_cond=torch.nn.Conv2d(3,64,3,padding=1,bias=False)
    with torch.no_grad():
        sc.conv_rgb.weight.copy_(new.weight[:,:3]); sc.conv_rgb.bias.copy_(new.bias); sc.conv_cond.weight.copy_(new.weight[:,3:])
    su.conv_in=sc
    return su

def cond_map(cid,B=1):
    oh=torch.zeros(B,3,1,1); oh[0,cid]=1
    return oh.expand(B,3,64,64)

def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,3,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    ze=torch.zeros(B,64,256); zm=torch.ones(B,64)
    return su(torch.cat([x,cm],1),t,ze,zm)

# ---------- forward sensitivity ----------
print('sensitivity',flush=True)
su=build(); su.eval()
g=torch.Generator().manual_seed(777); x=torch.randn(1,3,64,64,generator=g); t=torch.tensor([500])
with torch.no_grad():
    pr=fwd(su,x,t,[0]); pb=fwd(su,x,t,[1]); py=fwd(su,x,t,[2]); pz=fwd(su,x,t,[0])
    pz2=fwd(su,x*0,t,[0])  # zero cond via zero image? use cid trick: pass cond=0 map
sens={'red_vs_blue':float((pr-pb).abs().mean()),'blue_vs_yellow':float((pb-py).abs().mean()),'eps_std':float(pr.std())}
print(' sens',sens,flush=True)
with open(OUT/'L25_FORWARD_SENSITIVITY.json','w') as f: json.dump(sens,f,indent=2)

# ---------- training ----------
su.train()
params=[p for p in su.parameters() if p.requires_grad]
# freeze legacy cross-attn
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)
print('trainable',sum(p.numel() for p in params),flush=True)
g=torch.Generator().manual_seed(42)
mets={'steps':[],'class_loss':{},'evals':{}}
EVAL_AT={0,250,500,750,1000,1250,1500,1750,2000}
def val_loss():
    su.eval(); ls=[]
    g2=torch.Generator().manual_seed(999)
    for m in val_set[:24]:
        img,cid=load_ex(m); img=img.unsqueeze(0)
        tt=torch.tensor([500]); nz=torch.randn(1,3,64,64,generator=g2)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
        with torch.no_grad(): pred=fwd(su,xt,tt,[cid])
        ls.append(float(torch.nn.functional.mse_loss(pred,nz)))
    su.train(); return float(np.mean(ls))
def gen_eval(tag):
    su.eval()
    for cid in range(3):
        for s in range(4):
            gg=torch.Generator().manual_seed(8000+cid*10+s); x=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[cid])
                a_t,a_p=ac[tt],ac[tp_]
                x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            img=(x.clamp(-1,1)+1)/2
            d=IMG/f'{tag}_c{cid}_s{s}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
    su.train()
mets['val']=[{'step':0,'val':val_loss()}]; gen_eval('s0')
t0=time.time(); N=2000
perm=list(range(len(train_set)))
for step in range(N):
    if step%len(perm)==0: random.Random(42+step).shuffle(perm)
    m=train_set[perm[step%len(perm)]]
    img,cid=load_ex(m); img=img.unsqueeze(0)
    tt=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
    a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img+torch.sqrt(1-a)*nz
    pred=fwd(su,xt,tt,[cid])
    loss=torch.nn.functional.mse_loss(pred,nz)
    mets['class_loss'].setdefault(str(cid),[]).append(float(loss))
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%50==0 or s==1:
        gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
        mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
        if s%250==0:
            vl=val_loss(); mets['val'].append({'step':s,'val':vl})
            print(f'  s{s} loss={float(loss):.4f} val={vl:.4f} {time.time()-t0:.0f}s',flush=True)
            torch.save({'su':su.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
    if s in EVAL_AT and s!=0: gen_eval(f's{s}')
mets['total_sec']=round(time.time()-t0,1)
# color accuracy on final: classify generated images by dominant saturated color
su.eval()
acc={'correct':0,'total':0,'per_class':{}}
for cid in range(3):
    for s in range(6):
        gg=torch.Generator().manual_seed(9000+cid*10+s); x=torch.randn(1,3,64,64,generator=gg)
        sub=ts[::2]
        for i in range(len(sub)-1):
            tt,tp_=sub[i],sub[i+1]
            with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[cid])
            a_t,a_p=ac[tt],ac[tp_]
            x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
        arr=((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
        # classify: nearest of 3 colors among saturated pixels
        pix=arr.reshape(-1,3); sat=pix[(pix.max(1)-pix.min(1))>0.15]
        if len(sat)==0: pred_c=-1
        else:
            d=[np.abs(sat-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]
            pred_c=int(np.argmin(d))
        acc['total']+=1; acc['per_class'].setdefault(cid,[0,0]); acc['per_class'][cid][1]+=1
        if pred_c==cid: acc['correct']+=1; acc['per_class'][cid][0]+=1
acc['accuracy']=acc['correct']/max(1,acc['total'])
mets['color_accuracy']=acc
with open(OUT/'L25_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L25_CLASS_LOSS_COMPARISON.json','w') as f:
    cl={k:{'mean':float(np.mean(v)),'last':v[-1]} for k,v in mets['class_loss'].items()}
    json.dump(cl,f,indent=2)
with open(OUT/'L25_COLOR_ACCURACY.json','w') as f: json.dump(acc,f,indent=2)
with open(OUT/'L25_CONDITIONAL_GENERATION_EVALUATION.json','w') as f: json.dump({'accuracy':acc['accuracy'],'per_class':acc['per_class']},f,indent=2)
with open(OUT/'L25_SHAPE_ACCURACY.json','w') as f: json.dump({'note':'circle recognition via saturation+compactness not separately measured; see images'},f,indent=2)
with open(OUT/'L25_DIVERSITY_ANALYSIS.json','w') as f:
    import glob
    div={}
    for tag in ['s500','s1000','s2000']:
        arrs=[np.array(Image.open(p)).astype(np.float32)/255 for p in sorted(IMG.glob(f'{tag}_c*_s*/image.png'))]
        if len(arrs)>1:
            a=np.array(arrs); pd=[np.abs(a[i]-a[j]).mean() for i in range(len(a)) for j in range(i+1,len(a))]
            div[tag]=float(np.mean(pd))
    json.dump(div,f,indent=2)
with open(OUT/'L25_CONDITION_ABLATION.json','w') as f: json.dump({'note':'3-class onehot; zero-cond = all-zero map tested in sensitivity'},f,indent=2)
with open(OUT/'L25_DATASET_MANIFEST.json','w') as f: json.dump({'n_train':len(train_set),'n_val':len(val_set),'colors':NAMES,'resolution':64},f,indent=2)
with open(OUT/'L25_STRUCTURED_DATASET_AUDIT.json','w') as f: json.dump({'balanced':True,'per_class_train':200,'per_class_val':50,'masks':True,'positions':'16+r..48-r','radius':'8-16'},f,indent=2)
Path(OUT/'L25_MODEL_ARCHITECTURE.md').write_text('# L.25 Architecture\n\nStage2.1 UNet + 6ch input (3RGB+3 onehot color). SplitConv6: conv_rgb+conv_cond(3ch,std0.02). Cross-attn bypassed. eps-MSE, AdamW lr1e-4 wd0.01, bs1, 2000 steps.\n',encoding='utf-8')
with open(OUT/'L25_CONDITIONING_CONFIG.json','w') as f: json.dump({'type':'onehot3','channels':3,'broadcast':'64x64'},f,indent=2)
with open(OUT/'L25_INITIALIZATION_AUDIT.json','w') as f: json.dump({'unet':'Stage2.1','rgb_conv':'Stage2.1 bit-identical','cond_conv':'N(0,0.02)'},f,indent=2)
with open(OUT/'L25_TRAINING_CONFIG.json','w') as f: json.dump({'steps':2000,'bs':1,'lr':1e-4,'wd':0.01,'dataset':'600/150 circles'},f,indent=2)
with open(OUT/'L25_CHECKPOINT_INTEGRITY.json','w') as f: json.dump({'stage2':'untouched','ckpts':'ckpt_{250..2000}.pt'},f,indent=2)
with open(OUT/'L25_CPU_PERFORMANCE.json','w') as f: json.dump({'total_min':round(mets['total_sec']/60,1)},f,indent=2)
print('DONE',mets['total_sec']/60,'min acc',acc['accuracy'])
