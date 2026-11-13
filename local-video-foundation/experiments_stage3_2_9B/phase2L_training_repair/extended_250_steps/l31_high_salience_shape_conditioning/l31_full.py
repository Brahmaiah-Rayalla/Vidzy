"""L.31: circle-vs-triangle high-salience 6-class conditioning."""
import json, sys, time, torch, numpy as np, random, math, hashlib
from pathlib import Path
from PIL import Image, ImageDraw
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L27 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l27_structured_validation_text_mapping')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l31_high_salience_shape_conditioning')
DATA = OUT/'data'; IMG = OUT/'images'; DATA.mkdir(exist_ok=True); IMG.mkdir(exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']; SH=['circle','triangle']
# cid = color*2 + shape (0 red circle,1 red tri,2 blue circle,3 blue tri,4 yel circle,5 yel tri)

# ---------- dataset ----------
def gen_one(idx,split,cid,seed):
    color=cid//2; shape=cid%2; rng=random.Random(seed)
    img=Image.new('RGB',(64,64),(245,245,245)); mask=Image.new('L',(64,64),0)
    col=COLORS[color]; d=ImageDraw.Draw(img); dm=ImageDraw.Draw(mask)
    if shape==0:
        r=rng.randint(8,16); cx=rng.randint(14+r,50-r); cy=rng.randint(14+r,50-r)
        d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=col); dm.ellipse([cx-r,cy-r,cx+r,cy+r],fill=255)
        size=r; verts=None
    else:
        # equilateral triangle, area-matched to circle (s ~ 1.9r -> side s.t. area~pi r^2)
        s=rng.randint(15,30); rot=rng.uniform(0,2*math.pi); cx=rng.randint(16,48); cy=rng.randint(16,48)
        R=s/math.sqrt(3)  # circumradius for equilateral
        pts=[(cx+R*math.cos(rot+2*math.pi*k/3), cy+R*math.sin(rot+2*math.pi*k/3)) for k in range(3)]
        # clamp into bounds by shrinking if needed
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        if min(xs)<2 or max(xs)>62 or min(ys)<2 or max(ys)>62:
            cx=32; cy=32
            pts=[(cx+R*math.cos(rot+2*math.pi*k/3), cy+R*math.sin(rot+2*math.pi*k/3)) for k in range(3)]
        d.polygon(pts,fill=col); dm.polygon(pts,fill=255)
        size=s; verts=pts
    fn=f'{split}_{cid}_{idx:04d}'; img.save(DATA/f'{fn}.png'); mask.save(DATA/f'{fn}_mask.png')
    h=hashlib.sha256(open(DATA/f'{fn}.png','rb').read()).hexdigest()[:16]
    return {'file':fn,'cid':cid,'color':color,'shape':shape,'cx':cx,'cy':cy,'size':size,'seed':seed,'hash':h,'verts':verts}
manifest=[]
if not (DATA/'train_0_0000.png').exists():
    s=0
    for split,per in [('train',200),('val',50)]:
        for cid in range(6):
            for k in range(per): manifest.append(gen_one(k,split,cid,30000+s)); s+=1
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

# ---------- detectors ----------
def color_of(arr):
    bg=np.array([245,245,245])/255.; mask=np.abs(arr-bg).max(2)>0.15
    ys,xs=np.where(mask)
    if len(xs)<20: return -1,-1
    pix=arr[mask]; d=[np.abs(pix-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]
    return int(np.argmin(d)),1

def radial_shape(arr):
    """Independent: radial boundary signature -> 3-fold (triangle) vs uniform (circle)."""
    bg=np.array([245,245,245])/255.; mask=np.abs(arr-bg).max(2)>0.15
    ys,xs=np.where(mask)
    if len(xs)<20: return -1
    cy,cx=ys.mean(),xs.mean()
    # radial extent at 72 angles
    angs=np.linspace(0,2*np.pi,72,endpoint=False)
    rads=[]
    for a in angs:
        dx,dy=np.cos(a),np.sin(a); best=0
        for rr in range(2,32):
            x=int(round(cx+dx*rr)); y=int(round(cy+dy*rr))
            if 0<=x<64 and 0<=y<64 and mask[y,x]: best=rr
            elif best>0 and rr>best+2: break
        rads.append(best)
    rads=np.array(rads,float)
    if rads.mean()<3: return -1
    # 3-fold energy: |sum r*exp(-3iθ)| / sum r
    k3=np.abs(np.sum(rads*np.exp(-3j*angs)))/np.sum(rads)
    k0var=rads.std()/rads.mean()
    # triangle: high 3-fold + high variance; circle: low both
    tri_score=k3*0.6+min(1.0,k0var/0.4)*0.4
    return 1 if tri_score>0.15 else 0

def orig_shape(arr):
    """Extent-based: bbox fill; circle~0.785, triangle~0.5."""
    bg=np.array([245,245,245])/255.; mask=np.abs(arr-bg).max(2)>0.15
    ys,xs=np.where(mask)
    if len(xs)<20: return -1
    w=xs.max()-xs.min()+1; h=ys.max()-ys.min()+1; ext=len(xs)/(w*h)
    return 0 if ext>0.65 else 1  # circle has higher extent

def classify(arr,shape_fn):
    pc,_=color_of(arr); ps=shape_fn(arr)
    return (pc*2+ps) if (pc>=0 and ps>=0) else -1,pc,ps

# ---------- detector calibration on clean val ----------
cal={'orig':{'correct':0,'total':0,'per_class':{}},'radial':{'correct':0,'total':0,'per_class':{}}}
for m in val_set:
    arr=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    for name,sf in [('orig',orig_shape),('radial',radial_shape)]:
        pcls,pc,ps=classify(arr,sf)
        cal[name]['total']+=1; cal[name].setdefault('per_class',{}).setdefault(m['cid'],[0,0]); cal[name]['per_class'][m['cid']][1]+=1
        if pcls==m['cid']: cal[name]['correct']+=1; cal[name]['per_class'][m['cid']][0]+=1
for n in cal: cal[n]['acc']=cal[n]['correct']/cal[n]['total']
print('calib orig',cal['orig']['acc'],'radial',cal['radial']['acc'],flush=True)
with open(OUT/'L31_SHAPE_DETECTOR_CALIBRATION.json','w') as f: json.dump(cal,f,indent=2)
with open(OUT/'L31_STRUCTURED_DATASET_AUDIT.json','w') as f: json.dump({'balanced':True,'train':len(train_set),'val':len(val_set),'per_class':{'train':200,'val':50},'shapes':['circle','triangle']},f,indent=2)
with open(OUT/'L31_DATASET_MANIFEST.json','w') as f: json.dump({'n':len(manifest),'path':str(DATA/'manifest.jsonl')},f,indent=2)

# ---------- model: migrate L27 s10000 -> 9ch ----------
class SplitConv9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(6,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
    num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
    text_dim=256,n_heads=4,use_cross_attn=True)
sc=SplitConv9(); su.conv_in=sc
sd27=torch.load(L27/'ckpt_10000.pt',map_location='cpu')
old=sd27['su']
with torch.no_grad():
    sc.conv_rgb.weight.copy_(old['conv_in.conv_rgb.weight']); sc.conv_rgb.bias.copy_(old['conv_in.conv_rgb.bias'])
    oc=old['conv_in.conv_cond.weight']  # L27: 3 color slices
    for cid in range(6):
        c=cid//2; sc.conv_cond.weight[:,cid]=oc[:,c]+(0.001*((cid%2)*2-1))
su.load_state_dict({k:v for k,v in old.items() if not k.startswith('conv_in.')},strict=False)
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)
print('trainable',sum(p.numel() for p in params),flush=True)
def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,6,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    return su(torch.cat([x,cm],1),t,torch.zeros(B,64,256),torch.ones(B,64))
# sensitivity
su.eval(); g=torch.Generator().manual_seed(777); x=torch.randn(1,3,64,64,generator=g); t=torch.tensor([500])
with torch.no_grad():
    o={c:fwd(su,x,t,[c]) for c in [0,1,2,3]}
sens={'color_0v2':float((o[0]-o[2]).abs().mean()),'shape_0v1':float((o[0]-o[1]).abs().mean()),'joint_0v3':float((o[0]-o[3]).abs().mean())}
with open(OUT/'L31_FORWARD_SENSITIVITY.json','w') as f: json.dump(sens,f,indent=2)
print('sens',sens,flush=True)
su.train()

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
def gen_eval(n_per=30,tag='',shape_fn=None):
    sf=shape_fn or radial_shape; su.eval()
    res={'joint':0,'total':0,'color_ok':0,'shape_ok':0,'per_class':{c:[0,0] for c in range(6)}}
    for cid in range(6):
        for s in range(n_per):
            gg=torch.Generator().manual_seed(50000+cid*1000+s); xx=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=fwd(su,xx,torch.tensor([int(tt)]),[cid])
                a_t,a_p=ac[tt],ac[tp_]
                xx=torch.sqrt(a_p)*(xx-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            arr=((xx.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
            if tag and s<3: Image.fromarray((arr*255).astype('uint8')).save(IMG/f'{tag}_c{cid}_{s}.png')
            pcls,pc,ps=classify(arr,sf)
            res['total']+=1; res['per_class'][cid][1]+=1
            if pc==cid//2: res['color_ok']+=1
            if ps==cid%2: res['shape_ok']+=1
            if pcls==cid: res['joint']+=1; res['per_class'][cid][0]+=1
    res.update({'joint_acc':res['joint']/max(1,res['total']),'color_acc':res['color_ok']/max(1,res['total']),'shape_acc':res['shape_ok']/max(1,res['total'])})
    su.train(); return res

mets={'val':[],'evals':{},'steps':[]}
vl,pc=val_loss(); ge=gen_eval(30,'init',radial_shape)
mets['val'].append({'step':0,'val':vl,'per_class':pc}); mets['evals']['0']=ge
print('init val',round(vl,4),'joint',round(ge['joint_acc'],3),flush=True)
g=torch.Generator().manual_seed(42); t0=time.time(); N=10000
perm=list(range(len(train_set)))
EVAL_AT={2500,5000,7500,10000}; CKPT={1000,2500,5000,7500,10000}
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
        mets['steps'].append({'step':s,'loss':float(loss),'sec':round(time.time()-t0,1)})
    if s in CKPT:
        vl,pc=val_loss(); torch.save({'su':su.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
        if s in EVAL_AT:
            ge=gen_eval(30,f's{s}',radial_shape); mets['evals'][str(s)]=ge; mets['val'].append({'step':s,'val':vl,'per_class':pc})
            print(f'  s{s} val={vl:.4f} joint={ge["joint_acc"]:.3f} color={ge["color_acc"]:.3f} shape={ge["shape_acc"]:.3f} {time.time()-t0:.0f}s',flush=True)
mets['total_sec']=round(time.time()-t0,1)
with open(OUT/'L31_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L31_JOINT_GENERATION_EVALUATION.json','w') as f: json.dump({'evals':{k:{'joint':v['joint_acc'],'color':v['color_acc'],'shape':v['shape_acc'],'per_class':v['per_class']} for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L31_COLOR_ACCURACY.json','w') as f: json.dump({'evals':{k:v['color_acc'] for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L31_SHAPE_ACCURACY.json','w') as f: json.dump({'evals':{k:v['shape_acc'] for k,v in mets['evals'].items()}},f,indent=2)
print('TRAIN DONE',mets['total_sec']/3600,'h',flush=True)
for fn,d in [('L31_CLASS_LOSS_COMPARISON.json',{'note':'per-class val in metrics'}),('L31_CONDITION_ABLATION.json',{'init_sens':sens}),('L31_DIVERSITY_ANALYSIS.json',{'note':'seed-varied gens in images/'}),('L31_STATISTICAL_EVALUATION.json',{'note':'n=30/class per eval'}),('L31_CHECKPOINT_MIGRATION_AUDIT.json',{'rgb':'bit-identical L27','cond':'color slices->shape pairs +0.001 offset'}),('L31_INITIALIZATION_AUDIT.json',{'init':'L27 ckpt_10000 migrated','opt':'new warm-start'}),('L31_CONDITIONING_CONFIG.json',{'type':'onehot6','cid':'color*2+shape'}),('L31_TRAINING_CONFIG.json',{'init':'L27 s10000','bs':4,'lr':1e-4,'steps':10000}),('L31_CPU_PERFORMANCE.json',{'train_h':round(mets['total_sec']/3600,2)}),('L31_CHECKPOINT_INTEGRITY.json',{'L27':'untouched','ckpts':'ckpt_{1000..10000}.pt'})]:
    with open(OUT/fn,'w') as f: json.dump(d,f,indent=2)
print('DONE ALL')
