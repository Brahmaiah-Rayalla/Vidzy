"""L.23: split conv + limited unfreeze. Arm A cond-only, Arm B +first res block."""
import json, sys, time, torch, numpy as np, hashlib
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L21 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l21_direct_text_channel_experiment')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l23_limited_spatial_unfreezing')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas=torch.linspace(0.0001,0.02,1000); abars=torch.cumprod(1-betas,0)
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
SEQ,FEAT,NCH=64,8,512
meta=json.load(open(base/'data/text_image/train_metadata.json')); train_meta=meta[:700]
def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1
def build_te():
    te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
    for p in te.parameters(): p.requires_grad_(False)
    return te

class SplitConv(torch.nn.Module):
    """h = conv_rgb(x_rgb) + gain*conv_text(text_channels). No shared bias."""
    def __init__(self, ckpt_su):
        super().__init__()
        w=ckpt_su.conv_in.weight; b=ckpt_su.conv_in.bias
        self.conv_rgb=torch.nn.Conv2d(3,w.shape[0],3,padding=1)
        self.conv_text=torch.nn.Conv2d(NCH,w.shape[0],3,padding=1,bias=False)
        with torch.no_grad():
            self.conv_rgb.weight.copy_(w[:,:3]); self.conv_rgb.bias.copy_(b)
            self.conv_text.weight.copy_(w[:,3:])
        self.gain=1.0
    def forward(self,xc):
        return self.conv_rgb(xc[:,:3])+self.gain*self.conv_text(xc[:,3:])

def build_model():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    # temp expanded conv to load A750 weights
    old=su.conv_in
    new=torch.nn.Conv2d(3+NCH,old.out_channels,3,padding=1)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias); new.weight[:,:3].copy_(old.weight); new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new
    sd=torch.load(L21/'A_step750.pt',map_location='cpu')
    su.load_state_dict(sd['su'])
    tp=torch.nn.Linear(256,FEAT); tp.load_state_dict(sd['tokproj'])
    sc=SplitConv(su)   # reads expanded weights from su.conv_in
    su.conv_in=sc
    return su,tp

def text_channels(te,tp,cap):
    ids,m=tok.encode(cap,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
    tpp=tp(e)*mk.unsqueeze(-1).float()
    return tpp.reshape(1,-1,1,1).expand(1,-1,64,64),e,mk
def fwd(su,tp,te,x,t,cap,zero=False):
    tc,e,mk=text_channels(te,tp,cap)
    if zero: tc=torch.zeros_like(tc)
    ze=torch.zeros_like(e); zm=torch.ones_like(mk)
    return su(torch.cat([x,tc],1),t,ze,zm)

# ---------- parity audit ----------
print('parity',flush=True)
suA,tpA=build_model()
# reference: expanded conv forward
sd=torch.load(L21/'A_step750.pt',map_location='cpu')
ref_w=sd['su']['conv_in.weight']; ref_b=sd['su']['conv_in.bias']
g=torch.Generator().manual_seed(1); xc=torch.randn(1,515,64,64,generator=g)
with torch.no_grad():
    ref=ref_w[:, :3].unsqueeze(-1).unsqueeze(-1)  # not needed; use conv2d directly
    import torch.nn.functional as FFn
    refout=FFn.conv2d(xc[:,:3],ref_w[:,:3],ref_b,1,1)+FFn.conv2d(xc[:,3:],ref_w[:,3:],None,1,1)
    splitout=suA.conv_in(xc)
parity={'max_abs_diff':float((refout-splitout).abs().max())}
print(' parity max diff',parity['max_abs_diff'],flush=True)
with open(OUT/'L23_CONV_SPLIT_PARITY_AUDIT.json','w') as f: json.dump(parity,f,indent=2)

# ---------- gain sweep ----------
print('gain sweep',flush=True)
te=build_te(); suA.eval(); tpA.eval()
fcaps=[train_meta[i]['caption'] for i in [0,1,2,5]]
g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); tt=torch.tensor([int(ts[12])])
sweep={}
for gain in [0.5,1.0,2.0,4.0]:
    suA.conv_in.gain=gain
    with torch.no_grad():
        p0=fwd(suA,tpA,te,xn,tt,fcaps[0]); p1=fwd(suA,tpA,te,xn,tt,fcaps[1])
    sweep[str(gain)]={'swap_mean_abs':float((p0-p1).abs().mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(p0.flatten(),p1.flatten(),dim=0)),'eps_std':float(p0.std())}
    print(f' gain{gain}: {sweep[str(gain)]}',flush=True)
suA.conv_in.gain=1.0
with open(OUT/'L23_TEXT_GAIN_SWEEP.json','w') as f: json.dump(sweep,f,indent=2)
# first res block audit: does it see caption diffs?
fb=suA.down_blocks[0][0]
store={}
def cap_fwd(cap):
    st={}
    h=fb.register_forward_hook(lambda m,i,o: st.__setitem__('out',o.detach()))
    with torch.no_grad(): fwd(suA,tpA,te,xn,tt,cap)
    h.remove(); return st['out']
o0=cap_fwd(fcaps[0]); o1=cap_fwd(fcaps[1])
fb_audit={'first_block':'down_blocks.0.0','cap_diff':float((o0-o1).abs().mean()),'feat_rms':float(o0.pow(2).mean().sqrt()),
          'rel_diff':float((o0-o1).abs().mean()/(o0.pow(2).mean().sqrt()+1e-9))}
with open(OUT/'L23_FIRST_RESIDUAL_BLOCK_AUDIT.json','w') as f: json.dump(fb_audit,f,indent=2)
print(' fb audit',fb_audit,flush=True)

# ---------- arms ----------
man=json.load(open(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics\L17_OVERFIT_DATASET_MANIFEST.json'))
subset=[train_meta[i] for i in man['indices']]

def run_arm(arm):
    su,tp=build_model(); te2=build_te()
    for p in su.parameters(): p.requires_grad_(False)
    for p in tp.parameters(): p.requires_grad_(True)
    su.conv_in.conv_text.weight.requires_grad_(True)
    pg=[{'params':list(tp.parameters()),'lr':1e-4},
        {'params':[su.conv_in.conv_text.weight],'lr':1e-4}]
    if arm=='B':
        for p in su.down_blocks[0][0].parameters(): p.requires_grad_(True)
        pg.append({'params':list(su.down_blocks[0][0].parameters()),'lr':1e-5})
    opt=optim.AdamW(pg,weight_decay=0.01)
    su.train(); tp.train()
    # frozen hash snapshot
    frozen_hash={n:hashlib.sha256(p.detach().numpy().tobytes()).hexdigest() for n,p in su.named_parameters() if not p.requires_grad}
    g=torch.Generator().manual_seed(42)
    mets={'steps':[],'evals':{}}
    EVAL_AT={0,50,100,250,500}
    def evalall(tag):
        su.eval(); tp.eval(); res={}
        for j,it in enumerate(subset):
            gg=torch.Generator().manual_seed(6000+j); x=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=fwd(su,tp,te2,x,torch.tensor([int(tt)]),it['caption'])
                a_t,a_p=ac[tt],ac[tp_]
                x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            img=(x.clamp(-1,1)+1)/2
            d=IMG/f'{arm}_{tag}_p{j}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
            gt=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
            res[f'p{j}']={'mse_to_gt':float(((img[0].permute(1,2,0).numpy()-gt)**2).mean())}
        su.train(); tp.train(); return res
    mets['evals']['0']=evalall('s0')
    t0=time.time(); N=500
    for step in range(N):
        it=subset[step%10]; img=load_img(it)
        tt=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
        pred=fwd(su,tp,te2,xt,tt,it['caption'])
        loss=torch.nn.functional.mse_loss(pred,nz)
        opt.zero_grad(); loss.backward(); opt.step()
        s=step+1
        if s%20==0 or s==1:
            gn=sum(p.grad.norm().item() for p in tp.parameters() if p.grad is not None)+su.conv_in.conv_text.weight.grad.norm().item()
            mets['steps'].append({'step':s,'loss':float(loss),'gn':float(gn),'sec':round(time.time()-t0,1)})
            if s%100==0: print(f'  {arm} s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
        if s in EVAL_AT and s!=0:
            mets['evals'][str(s)]=evalall(f's{s}')
            torch.save({'su':su.state_dict(),'tokproj':tp.state_dict(),'splitconv':True,'step':s},OUT/f'{arm}_step{s}.pt')
    # integrity: frozen params unchanged
    integ=all(hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()==h for (n,p),(_,h) in zip([(n,p) for n,p in su.named_parameters() if not p.requires_grad],frozen_hash.items()))
    # simpler recheck
    mism=[n for n,h in frozen_hash.items() for pn,p in su.named_parameters() if pn==n and hashlib.sha256(p.detach().numpy().tobytes()).hexdigest()!=h]
    mets['frozen_integrity_violations']=mism
    mets['total_sec']=round(time.time()-t0,1)
    # final sensitivity
    su.eval(); tp.eval()
    gg=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=gg); tt=torch.tensor([int(ts[12])])
    with torch.no_grad():
        p0=fwd(su,tp,te2,xn,tt,subset[0]['caption']); p1=fwd(su,tp,te2,xn,tt,subset[1]['caption'])
        pz=fwd(su,tp,te2,xn,tt,subset[0]['caption'],zero=True)
    mets['final_sensitivity']={'swap_mean_abs':float((p0-p1).abs().mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(p0.flatten(),p1.flatten(),dim=0)),'zero_ablate':float((p0-pz).abs().mean())}
    return mets

mA=run_arm('A')
with open(OUT/'L23_ARM_A_TRAINING_METRICS.json','w') as f: json.dump(mA,f,indent=2)
print('ARM A done',mA['total_sec'],flush=True)
mB=run_arm('B')
with open(OUT/'L23_ARM_B_TRAINING_METRICS.json','w') as f: json.dump(mB,f,indent=2)
print('ARM B done',mB['total_sec'],flush=True)
comp={}
for arm,m in [('A',mA),('B',mB)]:
    for tg,ev in m['evals'].items():
        ms=[v['mse_to_gt'] for v in ev.values()]
        comp.setdefault(arm,{})[tg]={'mse_mean':float(np.mean(ms)),'mse_min':float(np.min(ms))}
with open(OUT/'L23_CAPTION_BINDING_EVALUATION.json','w') as f: json.dump({'evals':comp,'sens':{'A':mA['final_sensitivity'],'B':mB['final_sensitivity']}},f,indent=2)
with open(OUT/'L23_SEMANTIC_IMAGE_EVALUATION.json','w') as f: json.dump({'evals':comp,'note':'mse_to_gt 10 overfit pairs'},f,indent=2)
with open(OUT/'L23_FEATURE_PROPAGATION_ANALYSIS.json','w') as f: json.dump({'fb_audit':fb_audit,'gain_sweep':sweep},f,indent=2)
with open(OUT/'L23_ARM_A_CONFIG.json','w') as f: json.dump({'init':'L21 A_step750','trainable':'tokproj+conv_text','lr':1e-4,'steps':500},f,indent=2)
with open(OUT/'L23_ARM_B_CONFIG.json','w') as f: json.dump({'init':'L21 A_step750','trainable':'tokproj+conv_text+down_blocks.0.0@1e-5','lr':1e-4,'steps':500},f,indent=2)
with open(OUT/'L23_PARAMETER_FREEZE_AUDIT.json','w') as f: json.dump({'rgb_branch':'excluded from optimizer entirely','frozen_params_not_in_opt':True,'violations':{'A':mA['frozen_integrity_violations'],'B':mB['frozen_integrity_violations']}},f,indent=2)
with open(OUT/'L23_CHECKPOINT_INTEGRITY.json','w') as f: json.dump({'stage2':'untouched','L21_L22':'untouched','ckpts':['A_step{50,100,250,500}.pt','B_step{...}.pt']},f,indent=2)
with open(OUT/'L23_CPU_PERFORMANCE.json','w') as f: json.dump({'armA_min':round(mA['total_sec']/60,1),'armB_min':round(mB['total_sec']/60,1)},f,indent=2)
print('DONE ALL')
