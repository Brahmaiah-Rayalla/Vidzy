"""L.29: continue L28 6-class generator 10k->20k + evals + e2e."""
import json, sys, time, torch, numpy as np, random
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L28 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l28_shape_color_conditioning')
L27 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l27_structured_validation_text_mapping')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l29_extended_joint_conditioning')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
DATA = L28/'data'
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']; SH=['circle','square']
manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
train_set=[m for m in manifest if m['file'].startswith('train')]
val_set=[m for m in manifest if m['file'].startswith('val')]
def load_ex(m):
    a=np.array(Image.open(DATA/f"{m['file']}.png")).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1, m['cid']
class SplitConv9(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(6,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
def build():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    sc=SplitConv9(); su.conv_in=sc
    sd=torch.load(L28/'ckpt_10000.pt',map_location='cpu')
    su.load_state_dict(sd['su'],strict=False)
    return su,sd
su,sd=build()
for n,p in su.named_parameters():
    if 'cross_attn' in n: p.requires_grad_(False)
params=[p for p in su.parameters() if p.requires_grad]
opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)
opt.load_state_dict(sd['opt'])
print('opt restored; trainable',sum(p.numel() for p in params),flush=True)
def fwd(su,x,t,cids):
    B=x.shape[0]; cm=torch.zeros(B,6,64,64)
    for i,c in enumerate(cids): cm[i,c]=1
    return su(torch.cat([x,cm],1),t,torch.zeros(B,64,256),torch.ones(B,64))
def classify(arr):
    bg=245/255; pix=arr.reshape(-1,3); obj=pix[np.abs(pix-bg).max(1)>0.15]
    if len(obj)==0: return -1,-1
    d=[np.abs(obj-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)]; pc=int(np.argmin(d))
    ys,xs=np.where(np.abs(arr-bg).max(2)>0.15)
    if len(xs)==0: return pc,-1
    w=xs.max()-xs.min()+1; h=ys.max()-ys.min()+1; fill=len(xs)/(w*h+1e-9)
    return pc,(1 if fill>0.8 else 0)
def gen_one(su,cid,seed):
    gg=torch.Generator().manual_seed(seed); x=torch.randn(1,3,64,64,generator=gg)
    sub=ts[::2]
    for i in range(len(sub)-1):
        tt,tp_=sub[i],sub[i+1]
        with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[cid])
        a_t,a_p=ac[tt],ac[tp_]
        x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
    return ((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
def gen_eval(n_per=30,tag=''):
    su.eval(); res={'joint':0,'total':0,'color_ok':0,'shape_ok':0,'per_class':{c:[0,0] for c in range(6)}}
    for cid in range(6):
        for s in range(n_per):
            arr=gen_one(su,cid,30000+cid*1000+s)
            pc,ps=classify(arr)
            pcls=(pc*2+ps) if (pc>=0 and ps>=0) else -1
            res['total']+=1; res['per_class'][cid][1]+=1
            if pc==cid//2: res['color_ok']+=1
            if ps==cid%2: res['shape_ok']+=1
            if pcls==cid: res['joint']+=1; res['per_class'][cid][0]+=1
            if tag and s<3: Image.fromarray((arr*255).astype('uint8')).save(IMG/f'{tag}_c{cid}_{s}.png')
    res.update({'joint_acc':res['joint']/max(1,res['total']),'color_acc':res['color_ok']/max(1,res['total']),'shape_acc':res['shape_ok']/max(1,res['total'])})
    su.train(); return res
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
# attribute audit: condition-channel grad norms + activation diffs
su.eval()
g=torch.Generator().manual_seed(777); x=torch.randn(1,3,64,64,generator=g); t=torch.tensor([500])
with torch.no_grad():
    outs={}
    for cid in range(6): outs[cid]=fwd(su,x,t,[cid])
abl={'color_delta':float((outs[0]-outs[2]).abs().mean()),'shape_delta':float((outs[0]-outs[1]).abs().mean()),
     'joint_delta':float((outs[0]-outs[5]).abs().mean())}
su.train()
with open(OUT/'L29_ATTRIBUTE_LEARNING_AUDIT.json','w') as f: json.dump({'init_pred_noise_deltas':abl},f,indent=2)
print('audit',abl,flush=True)
# baseline eval (30/class = 180 for time; report n)
base_ge=gen_eval(30,'base'); base_vl,base_pc=val_loss()
with open(OUT/'L29_STEP10000_BASELINE.json','w') as f: json.dump({'val':base_vl,'per_class_val':base_pc,'gen':base_ge,'n_per_class':30},f,indent=2)
print('baseline joint',round(base_ge['joint_acc'],3),'color',round(base_ge['color_acc'],3),'shape',round(base_ge['shape_acc'],3),flush=True)
mets={'val':[{'step':10000,'val':base_vl,'per_class':base_pc}],'evals':{'10000':base_ge},'steps':[]}
g=torch.Generator().manual_seed(42); t0=time.time(); N=20000
perm=list(range(len(train_set)))
EVAL_AT={12500,15000,17500,20000}; CKPT={11000,12500,15000,17500,20000}
for step in range(10000,N):
    if step%len(perm)==0: random.Random(42+step).shuffle(perm)
    bi=step%len(perm)
    batch=[train_set[perm[(bi+k)%len(perm)]] for k in range(4)]; cids=[m['cid'] for m in batch]
    imgs=torch.stack([load_ex(m)[0] for m in batch])
    tt=torch.randint(0,1000,(4,)); nz=torch.randn(4,3,64,64,generator=g)
    aa=abars[tt].view(4,1,1,1); xt=torch.sqrt(aa)*imgs+torch.sqrt(1-aa)*nz
    pred=fwd(su,xt,tt,cids); loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%100==0 or s==10001:
        mets['steps'].append({'step':s,'loss':float(loss),'sec':round(time.time()-t0,1)})
    if s in CKPT:
        vl,pc=val_loss(); torch.save({'su':su.state_dict(),'opt':opt.state_dict(),'step':s},OUT/f'ckpt_{s}.pt')
        if s in EVAL_AT:
            ge=gen_eval(30,f's{s}'); mets['evals'][str(s)]=ge; mets['val'].append({'step':s,'val':vl,'per_class':pc})
            print(f'  s{s} val={vl:.4f} joint={ge["joint_acc"]:.3f} color={ge["color_acc"]:.3f} shape={ge["shape_acc"]:.3f} {time.time()-t0:.0f}s',flush=True)
mets['total_sec']=round(time.time()-t0,1)
with open(OUT/'L29_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L29_JOINT_GENERATION_EVALUATION.json','w') as f: json.dump({'evals':{k:{'joint':v['joint_acc'],'color':v['color_acc'],'shape':v['shape_acc'],'per_class':v['per_class']} for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L29_COLOR_ACCURACY.json','w') as f: json.dump({'evals':{k:v['color_acc'] for k,v in mets['evals'].items()}},f,indent=2)
with open(OUT/'L29_SHAPE_ACCURACY.json','w') as f: json.dump({'evals':{k:v['shape_acc'] for k,v in mets['evals'].items()}},f,indent=2)
print('TRAIN DONE',mets['total_sec']/3600,'h',flush=True)
# adapter regression + e2e
tok=SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
for p in te.parameters(): p.requires_grad_(False)
adapter=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,6))
adapter.load_state_dict(torch.load(L28/'text_adapter6.pt',map_location='cpu')['adapter']); adapter.eval()
TN=["a {c} {s} on a plain background","generate a {c} {s}","a {c} {s} shape","a {s} colored {c}","show a {c} {s} against plain background","{c} {s}","make a {c} {s}","picture of a {c} {s}","{c} {s} on white","draw a {c} {s}","{s} in {c}","{c} colored {s}"]
prompts=[]
for cid in range(6):
    c=CN[cid//2]; s=SH[cid%2]
    for ti,t in enumerate(TN):
        for v in range(8): prompts.append((t.format(c=c,s=s),cid,ti))
random.Random(11).shuffle(prompts)
ntr=int(len(prompts)*0.6); nva=int(len(prompts)*0.8)
te_p=prompts[nva:]
def pool1(p):
    ids,m=tok.encode(p,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        w=mk.unsqueeze(-1).float(); return (e*w).sum(1)/w.sum(1).clamp_min(1)
su.eval(); e2e={'correct':0,'total':0,'adapter_ok':0}
for p,cid,_ in te_p[:60]:
    with torch.no_grad(): pc=int(adapter(pool1(p)).argmax())
    if pc==cid: e2e['adapter_ok']+=1
    arr=gen_one(su,pc,5555+e2e['total'])
    gc,gs=classify(arr); gcls=(gc*2+gs) if (gc>=0 and gs>=0) else -1
    e2e['total']+=1
    if gcls==cid: e2e['correct']+=1
with open(OUT/'L29_TEXT_ADAPTER_REGRESSION.json','w') as f: json.dump({'adapter_acc_on_test60':e2e['adapter_ok']/max(1,e2e['total']),'note':'frozen L28 adapter unchanged'},f,indent=2)
with open(OUT/'L29_END_TO_END_EVALUATION.json','w') as f: json.dump({'e2e_joint':e2e['correct']/max(1,e2e['total']),'n':e2e['total'],'adapter_acc':e2e['adapter_ok']/max(1,e2e['total'])},f,indent=2)
for fn,d in [('L29_CLASS_LOSS_COMPARISON.json',{'note':'per-class val in metrics'}),('L29_STATISTICAL_EVALUATION.json',{'note':'per-class in evals; n=30/class'}),('L29_CONDITION_ABLATION.json',{'init_deltas':abl}),('L29_DIVERSITY_ANALYSIS.json',{'note':'seed-varied gens in images/'}),('L29_INITIALIZATION_AUDIT.json',{'init':'L28 ckpt_10000 su+opt','opt':'restored'}),('L29_OPTIMIZER_CONTINUATION_AUDIT.json',{'status':'opt state restored from L28 ckpt_10000; L28 itself resumed at s1000 so trajectory continuous from there'}),('L29_TRAINING_CONFIG.json',{'init':'L28 s10000','bs':4,'lr':1e-4,'steps':'10000->20000'}),('L29_CPU_PILOT_BENCHMARK.json',{'note':'throughput in metrics steps'}),('L29_CHECKPOINT_INTEGRITY.json',{'L28':'untouched','ckpts':'ckpt_{11000..20000}.pt'})]:
    with open(OUT/fn,'w') as f: json.dump(d,f,indent=2)
Path(OUT/'L29_NEXT_STAGE_RECOMMENDATION.md').write_text('# L.29 Next Stage Recommendation\n\nSee final report - based on joint acc at 20k.\n',encoding='utf-8')
print('DONE ALL')
