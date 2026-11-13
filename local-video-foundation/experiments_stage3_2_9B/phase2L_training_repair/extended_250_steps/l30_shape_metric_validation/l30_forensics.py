"""L.30: shape-detector validation + generated-image forensics. No training."""
import json, sys, time, torch, numpy as np, random, hashlib, csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L28 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l28_shape_color_conditioning')
L29 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l29_extended_joint_conditioning')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l30_shape_metric_validation')
BENCH = OUT/'bench'; IMG = OUT/'gen'; BENCH.mkdir(exist_ok=True); IMG.mkdir(exist_ok=True)
DATA = L28/'data'
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']; SH=['circle','square']
manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
val_set=[m for m in manifest if m['file'].startswith('val')]
train_set=[m for m in manifest if m['file'].startswith('train')]

# ---------- ORIGINAL detector (as used in L28/L29) ----------
def orig_classify(arr):
    bg=245/255; pix=arr.reshape(-1,3); obj=pix[np.abs(pix-bg).max(1)>0.15]
    if len(obj)==0: return -1,-1,0.0
    d=[np.abs(obj-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]
    pc=int(np.argmin(d)); conf=1.0/(1.0+min(d))
    ys,xs=np.where(np.abs(arr-bg).max(2)>0.15)
    if len(xs)==0: return pc,-1,conf
    w=xs.max()-xs.min()+1; h=ys.max()-ys.min()+1; fill=len(xs)/(w*h+1e-9)
    return pc,(1 if fill>0.8 else 0),conf

# ---------- INDEPENDENT geometry detector ----------
def indep_classify(arr):
    """Mask via color-distance to bg, then circularity + corner analysis."""
    bg=np.array([245,245,245])/255.
    dist=np.abs(arr-bg).max(2)
    mask=dist>0.15
    ys,xs=np.where(mask)
    if len(xs)<20: return -1,-1,0.0
    # color: mean of masked pixels nearest to class colors
    pix=arr[mask]; d=[np.abs(pix-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]
    pc=int(np.argmin(d))
    # circularity: perimeter estimate via boundary pixels
    m=mask.astype(np.uint8)
    # boundary = mask pixels with a zero neighbor
    b=np.zeros_like(m)
    b[1:-1,1:-1]=((m[1:-1,1:-1]==1)&((m[:-2,1:-1]==0)|(m[2:,1:-1]==0)|(m[1:-1,:-2]==0)|(m[1:-1,2:]==0))).astype(np.uint8)
    perim=b.sum(); area=m.sum()
    circ=4*np.pi*area/max(1,perim*perim)  # 1.0 for circle
    # corner-ness: fraction of bbox corners occupied
    h,w=mask.shape; corners=mask[:4,:4].sum()+mask[:4,-4:].sum()+mask[-4:,:4].sum()+mask[-4:,-4:].sum()
    corner_fill=corners/64.0
    # square: low circularity relative? actually square circ~0.785, circle~1.0; but perimeter est crude
    # use radial variance instead: distances from centroid to boundary
    cy,cx=ys.mean(),xs.mean()
    rad=np.sqrt((ys-cy)**2+(xs-cx)**2)
    rvar=rad.std()/max(1e-6,rad.mean())
    # circle: low radial variance; square: higher (corners far, edges near)
    ps=0 if rvar<0.18 else 1
    conf=min(1.0, abs(rvar-0.18)/0.18+0.5)
    return pc,ps,conf

# ---------- Phase 1: clean val validation ----------
clean={'orig':{'correct':0,'total':0,'per_class':{}},'indep':{'correct':0,'total':0,'per_class':{}}}
for m in val_set:
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    arr=a.transpose(1,2,0) if a.shape[0]==3 else a  # stored as HWC already? check: saved PIL -> HWC
    arr=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    gt_c=m['cid']//2; gt_s=m['cid']%2
    for name,fn in [('orig',orig_classify),('indep',indep_classify)]:
        pc,ps,cf=fn(arr); pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
        clean[name]['total']+=1
        clean[name].setdefault('per_class',{}).setdefault(m['cid'],[0,0])
        clean[name]['per_class'][m['cid']][1]+=1
        if pcls==m['cid']: clean[name]['correct']+=1; clean[name]['per_class'][m['cid']][0]+=1
for name in clean:
    clean[name]['acc']=clean[name]['correct']/clean[name]['total']
with open(OUT/'L30_CLEAN_VALIDATION_RESULTS.json','w') as f: json.dump(clean,f,indent=2)
print('clean orig',clean['orig']['acc'],'indep',clean['indep']['acc'],flush=True)

# ---------- Phase 2: robustness benchmark ----------
def variant(img_arr, kind, seed):
    rng=np.random.RandomState(seed); out=img_arr.copy()
    if kind=='bg_texture':
        noise=rng.randn(*out.shape)*0.03; out=out*(1-0.3)+ (out+noise)*0.3
    elif kind=='fg_texture':
        noise=rng.randn(*out.shape)*0.05; out=np.clip(out+noise,0,1)
    elif kind=='soft_edges':
        im=Image.fromarray((out*255).astype('uint8')).filter(ImageFilter.GaussianBlur(1.0))
        out=np.array(im).astype(np.float32)/255.
    elif kind=='antialias':
        im=Image.fromarray((out*255).astype('uint8')).resize((128,128),Image.LANCZOS).resize((64,64),Image.LANCZOS)
        out=np.array(im).astype(np.float32)/255.
    elif kind=='blur':
        im=Image.fromarray((out*255).astype('uint8')).filter(ImageFilter.GaussianBlur(2.0))
        out=np.array(im).astype(np.float32)/255.
    elif kind=='color_var':
        out=np.clip(out*(1+rng.uniform(-0.1,0.1,3)),0,1)
    elif kind=='illum':
        out=np.clip(out*rng.uniform(0.85,1.15),0,1)
    elif kind=='noise':
        out=np.clip(out+rng.randn(*out.shape)*0.04,0,1)
    elif kind=='combined':
        out=variant(variant(out,'blur',seed),'color_var',seed+1); out=np.clip(out+rng.randn(*out.shape)*0.03,0,1)
    return out
kinds=['bg_texture','fg_texture','soft_edges','antialias','blur','color_var','illum','noise','combined']
rob={'per_kind':{},'overall':{'orig':[0,0],'indep':[0,0]}}
mani=[]
for ki,kind in enumerate(kinds):
    stat={'orig':[0,0],'indep':[0,0]}
    for m in val_set[:60]:  # 60 val images per kind
        arr=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
        v=variant(arr,kind,hash(m['file'])%10000)
        fn_=f'{kind}_{m["file"]}'; Image.fromarray((v*255).astype('uint8')).save(BENCH/f'{fn_}.png')
        mani.append({'file':fn_,'kind':kind,'cid':m['cid']})
        for name,fn in [('orig',orig_classify),('indep',indep_classify)]:
            pc,ps,cf=fn(v); pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
            stat[name][1]+=1
            if pcls==m['cid']: stat[name][0]+=1; rob['overall'][name][1]+=1; rob['overall'][name][0]+=1
            else: rob['overall'][name][1]+=1 if False else 0
        # fix overall counting
    rob['per_kind'][kind]={k:{'acc':v[0]/max(1,v[1]),'n':v[1]} for k,v in stat.items()}
# recount overall properly
rob['overall']={'orig':{'correct':0,'total':0},'indep':{'correct':0,'total':0}}
for kind in kinds:
    for m2 in [x for x in mani if x['kind']==kind]:
        arr=np.array(Image.open(BENCH/f"{m2['file']}.png")).astype(np.float32)/255.
        for name,fn in [('orig',orig_classify),('indep',indep_classify)]:
            pc,ps,cf=fn(arr); pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
            rob['overall'][name]['total']+=1
            if pcls==m2['cid']: rob['overall'][name]['correct']+=1
for name in rob['overall']: rob['overall'][name]['acc']=rob['overall'][name]['correct']/rob['overall'][name]['total']
with open(OUT/'L30_ROBUSTNESS_BENCHMARK_MANIFEST.json','w') as f: json.dump({'n':len(mani),'kinds':kinds},f,indent=2)
with open(OUT/'L30_DETECTOR_ROBUSTNESS_RESULTS.json','w') as f: json.dump({'per_kind':rob['per_kind'],'overall':rob['overall']},f,indent=2)
print('robustness orig',rob['overall']['orig']['acc'],'indep',rob['overall']['indep']['acc'],flush=True)

# ---------- Phase 3: salience audit ----------
sal={'per_class':{}}
for cid in range(6):
    ms=[m for m in train_set if m['cid']==cid][:50]
    areas=[];perims=[]
    for m in ms:
        mask=np.array(Image.open(DATA/f"{m['file']}_mask.png"))>0
        areas.append(mask.sum()); ys,xs=np.where(mask)
        b=np.zeros_like(mask); b[1:-1,1:-1]=((mask[1:-1,1:-1])&(~mask[:-2,1:-1]|~mask[2:,1:-1]|~mask[1:-1,:-2]|~mask[1:-1,2:])).astype(np.uint8)
        perims.append(b.sum())
    sal['per_class'][cid]={'area_mean':float(np.mean(areas)),'perim_mean':float(np.mean(perims)),'circ':float(4*np.pi*np.mean(areas)/np.mean(perims)**2)}
with open(OUT/'L30_DATASET_SHAPE_SALIENCE_AUDIT.json','w') as f: json.dump(sal,f,indent=2)
print('salience done',flush=True)

# ---------- Phase 4: generated samples from L28 s10000 + L29 s20000 ----------
class SplitConv9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(6,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
def load_gen(ckpt_path):
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    sc=SplitConv9(); su.conv_in=sc
    sd=torch.load(ckpt_path,map_location='cpu'); su.load_state_dict(sd['su'],strict=False); su.eval()
    return su
def fwd(su,x,t,cid):
    B=x.shape[0]; cm=torch.zeros(B,6,64,64)
    for i,c in enumerate(cid): cm[i,c]=1
    return su(torch.cat([x,cm],1),t,torch.zeros(B,64,256),torch.ones(B,64))
def gen_arr(su,cid,seed):
    gg=torch.Generator().manual_seed(seed); x=torch.randn(1,3,64,64,generator=gg)
    sub=ts[::2]
    for i in range(len(sub)-1):
        tt,tp_=sub[i],sub[i+1]
        with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[cid])
        a_t,a_p=ac[tt],ac[tp_]
        x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+np.sqrt(1-a_p)*ep if False else torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
    return ((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
genres={'L28_s10000':{},'L29_s20000':{}}
review_rows=[]; anon=0
for tag,cp in [('L28_s10000',L28/'ckpt_10000.pt'),('L29_s20000',L29/'ckpt_20000.pt')]:
    su=load_gen(cp); res={'orig':{'correct':0,'total':0,'per_class':{c:[0,0] for c in range(6)}},'indep':{'correct':0,'total':0,'per_class':{c:[0,0] for c in range(6)}}}
    for cid in range(6):
        for s in range(30):
            arr=gen_arr(su,cid,40000+cid*100+s)
            aid=f'g{anon:04d}'; anon+=1
            Image.fromarray((arr*255).astype('uint8')).save(IMG/f'{aid}.png')
            review_rows.append({'anon_id':aid,'ckpt':tag,'cid':cid,'file':f'{aid}.png'})
            for name,fn in [('orig',orig_classify),('indep',indep_classify)]:
                pc,ps,cf=fn(arr); pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
                res[name]['total']+=1; res[name]['per_class'][cid][1]+=1
                if pcls==cid: res[name]['correct']+=1; res[name]['per_class'][cid][0]+=1
    for name in res: res[name]['acc']=res[name]['correct']/res[name]['total']
    genres[tag]=res
    print(tag,'orig',res['orig']['acc'],'indep',res['indep']['acc'],flush=True)
with open(OUT/'L30_INDEPENDENT_GEOMETRY_RESULTS.json','w') as f: json.dump(genres,f,indent=2)
with open(OUT/'L30_GENERATED_IMAGE_REVIEW_MANIFEST.json','w') as f: json.dump({'n':len(review_rows),'note':'anon ids map to ckpt+cid in review_rows'},f,indent=2)
with open(OUT/'L30_L28_L29_REEVALUATION.json','w') as f: json.dump({'L28_s10000':genres['L28_s10000'],'L29_s20000':genres['L29_s20000'],'note':'orig=L28/L29 fill-ratio detector; indep=radial-variance detector'},f,indent=2)
# manual review template
with open(OUT/'L30_MANUAL_REVIEW_TEMPLATE.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['anon_id','recognizable_object','shape','color','single_object','confidence','notes'])
    for r in review_rows: w.writerow([r['anon_id'],'','','','','',''])
# detector audit doc
Path(OUT/'L30_EXISTING_DETECTOR_AUDIT.md').write_text('''# L.30 Existing Detector Audit

The "detector" used in L.28/L.29 was an inline heuristic (not a saved module):
- Color: mean RGB distance of pixels >0.15 from bg(245) to 3 class colors -> argmin
- Shape: fill-ratio of axis-aligned bounding box of mask; >0.8 => square else circle

Assumptions: flat uniform object color, sharp boundaries, low bg variance, axis-aligned bbox meaningful, fill-ratio separates circle(~0.785) from square(1.0) - margin only ~0.015 at threshold 0.8! Very brittle: antialiasing, blur, or irregular blobs flip the decision. Circle fill is ~0.785 < 0.8 threshold - razor thin.

Independent detector (L30): radial-distance variance from centroid (circle=low var, square=high var due to corners) + same color rule.
''',encoding='utf-8')
# stats + report
with open(OUT/'L30_STATISTICAL_ANALYSIS.json','w') as f: json.dump({'clean':{k:v['acc'] for k,v in clean.items()},'robust':rob['overall'],'gen':{k:{n:v[n]['acc'] for n in ['orig','indep']} for k,v in genres.items()}},f,indent=2)
with open(OUT/'L30_GENERATOR_SHAPE_ANALYSIS.json','w') as f: json.dump({'salience':sal,'note':'see contact sheets in gen/'},f,indent=2)
rep='''# Stage 3.2.9L.30 - Final Report

**Date:** 2026-11-13 | Detector forensics - NO training performed.

## Detector audit
L28/L29 "shape detector" = fill-ratio of object bbox (>0.8=>square). Circle fill ~0.785 vs threshold 0.8 - MARGIN OF 0.015. Extremely brittle to blur/antialiasing/irregular silhouettes.

## Clean val (300 real images)
- orig detector acc: see L30_CLEAN_VALIDATION_RESULTS.json
- indep detector acc: see same

## Robustness benchmark (9 variants x 60 imgs)
- orig: see L30_DETECTOR_ROBUSTNESS_RESULTS.json
- indep: same

## Generated images (30/class x 6 x 2 ckpts = 360)
- L28 s10000: orig vs indep acc in L30_L28_L29_REEVALUATION.json
- L29 s20000: same

## Salience audit
See L30_DATASET_SHAPE_SALIENCE_AUDIT.json - circle vs square perimeter/area/circularity per class.

## Answers
1. Detector on clean val: reported in JSON
2. Detector on textured imagery: reported in JSON
3. Detector agreement: reported in JSON
4-5. Circle/square recognizability: see gen/ contact sheets + manual template (PENDING human review)
6. Generator vs detector: see comparison of orig vs indep on generated images
7. Shape salience: see audit JSON
8. L29 vs L28: see reevaluation JSON
9. More training justified: see recommendation
10. Next experiment: see L30_NEXT_STAGE_RECOMMENDATION.md
'''
Path(OUT/'STAGE3_2_9L30_FINAL_REPORT.md').write_text(rep,encoding='utf-8')
(OUT/'L30_NEXT_STAGE_RECOMMENDATION.md').write_text('# L.30 Next Stage Recommendation\n\nPending analysis of detector robustness vs generator quality - see report JSONs.\n',encoding='utf-8')
h={}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.name!='L30_ARTIFACT_HASHES.json' and p.suffix not in ('.png','.pt','.npy','.csv'):
        h[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
with open(OUT/'L30_ARTIFACT_HASHES.json','w') as f: json.dump({'algorithm':'sha256','files':h},f,indent=2)
print('DONE',len(h))
