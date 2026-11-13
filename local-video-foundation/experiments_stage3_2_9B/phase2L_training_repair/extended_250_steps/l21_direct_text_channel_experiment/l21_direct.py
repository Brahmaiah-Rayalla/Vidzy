"""L.21: direct text-channel injection existence proof (2 arms x 750 steps)."""
import json, sys, time, torch, numpy as np, hashlib
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l21_direct_text_channel_experiment')
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
SEQ=64; FEAT=8; NCH=SEQ*FEAT

def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1

def build_te():
    te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
    for p in te.parameters(): p.requires_grad_(False)
    return te

def build_model():
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    # expand conv_in: 3 -> 3+NCH
    old=su.conv_in
    new=torch.nn.Conv2d(3+NCH,old.out_channels,old.kernel_size,old.stride,old.padding)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias)
        new.weight[:,:3].copy_(old.weight)
        new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new
    tokproj=torch.nn.Linear(256,FEAT)
    torch.nn.init.normal_(tokproj.weight,std=0.02); torch.nn.init.zeros_(tokproj.bias)
    return su,tokproj

def text_channels(te,tokproj,cap):
    ids,m=tok.encode(cap,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
    tp=tokproj(e)                       # [B,S,F]
    tp=tp*mk.unsqueeze(-1).float()
    return tp.reshape(1,-1,1,1).expand(1,-1,64,64), e, mk

def forward_eps(su,tokproj,te,x,t,cap,zero_text=False):
    tc,e,mk=text_channels(te,tokproj,cap)
    if zero_text: tc=torch.zeros_like(tc)
    xc=torch.cat([x,tc],dim=1)
    ze=torch.zeros_like(e) if zero_text else e   # legacy path gets zeros -> constant
    zm=torch.ones_like(mk)
    return su(xc,t,ze,zm)

def enc_ids(cap):
    ids,m=tok.encode(cap,max_length=64); return ids,m

# ---------- Phase 3: initial forward sensitivity ----------
print('Phase3 init sensitivity',flush=True)
su0,tp0=build_model(); su0.eval(); tp0.eval(); te0=build_te()
diag_caps=[train_meta[i]['caption'] for i in [0,1,2,5,17,33,60,88]]
g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); t=torch.tensor([int(ts[12])])
sens={}
for cap in diag_caps[:4]:
    with torch.no_grad():
        p0=forward_eps(su0,tp0,te0,xn,t,cap)
        pz=forward_eps(su0,tp0,te0,xn,t,cap,zero_text=True)
        pw=forward_eps(su0,tp0,te0,xn,t,diag_caps[1] if cap==diag_caps[0] else diag_caps[0])
    sens[cap[:30]]={'text_vs_zero':float((p0-pz).abs().mean()),'text_vs_swap':float((p0-pw).abs().mean()),'eps_std':float(p0.std())}
print(' sens sample:',list(sens.items())[0],flush=True)
with open(OUT/'L21_INITIAL_FORWARD_SENSITIVITY.json','w') as f: json.dump(sens,f,indent=2)
# binding unit test
def binding_test():
    lc=torch.tensor(0.02); lw=torch.tensor(0.05); m=0.02
    lb=torch.clamp(m+lc-lw,min=0)
    cases={'correct_better':float(lb)==0.0,'equal_positive':float(torch.clamp(m+torch.tensor(0.03)-torch.tensor(0.03),min=0))>0,
           'wrong_better_larger':float(torch.clamp(m+torch.tensor(0.05)-torch.tensor(0.01),min=0))>float(lb)}
    return cases
bt=binding_test()
with open(OUT/'L21_BINDING_OBJECTIVE_AUDIT.json','w') as f: json.dump({'hinge_tests':bt,'lambda':0.1,'margin':0.02,'note':'L=max(0,margin+Lc-Lw); verified zero when Lc<Lw-margin'},f,indent=2)
print(' binding tests:',bt,flush=True)

# ---------- Phase 5-6: two arms ----------
man=json.load(open(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics\L17_OVERFIT_DATASET_MANIFEST.json'))
subset=[train_meta[i] for i in man['indices']]
mismatch=[subset[(j+3)%10]['caption'] for j in range(10)]

def run_arm(arm):
    su,tp=build_model(); te=build_te()
    su.train(); tp.train()
    params=list(su.parameters())+list(tp.parameters())
    opt=optim.AdamW(params,lr=0.0001,weight_decay=0.01)
    g=torch.Generator().manual_seed(42)
    mets={'steps':[],'binding':[] if arm=='B' else None,'evals':{}}
    EVAL_AT={0,100,250,500,750}
    def evalall(tag):
        su.eval(); tp.eval(); res={}
        for j,it in enumerate(subset):
            gg=torch.Generator().manual_seed(6000+j); x=torch.randn(1,3,64,64,generator=gg)
            sub=ts[::2]
            for i in range(len(sub)-1):
                tt,tp_=sub[i],sub[i+1]
                with torch.no_grad(): ep=forward_eps(su,tp,te,x,torch.tensor([int(tt)]),it['caption'])
                a_t,a_p=ac[tt],ac[tp_]
                x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            img=(x.clamp(-1,1)+1)/2
            d=IMG/f'{arm}_{tag}_p{j}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
            gt=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
            res[f'p{j}']={'mse_to_gt':float(((img[0].permute(1,2,0).numpy()-gt)**2).mean())}
        su.train(); tp.train(); return res
    mets['evals']['0']=evalall('s0')
    t0=time.time(); N=750
    for step in range(N):
        j=step%10; it=subset[j]; img=load_img(it)
        tt=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
        a=abars[tt].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
        pred=forward_eps(su,tp,te,xt,tt,it['caption'])
        lc=torch.nn.functional.mse_loss(pred,nz)
        if arm=='B':
            predw=forward_eps(su,tp,te,xt,tt,mismatch[j])
            lw=torch.nn.functional.mse_loss(predw,nz)
            lb=torch.clamp(0.02+lc-lw,min=0)
            loss=lc+0.1*lb
            if step%50==0: mets['binding'].append({'step':step+1,'lc':float(lc),'lw':float(lw),'lb':float(lb)})
        else: loss=lc
        opt.zero_grad(); loss.backward(); opt.step()
        s=step+1
        if s%20==0 or s==1:
            gn=sum(p.grad.norm().item() for p in params if p.grad is not None)
            tpgn=sum(p.grad.norm().item() for p in tp.parameters() if p.grad is not None)
            mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'tp_gn':tpgn,'sec':round(time.time()-t0,1)})
            if s%100==0: print(f'  {arm} s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
        if s in EVAL_AT and s!=0:
            mets['evals'][str(s)]=evalall(f's{s}')
            torch.save({'su':su.state_dict(),'tokproj':tp.state_dict(),'step':s},OUT/f'{arm}_step{s}.pt')
    mets['total_sec']=round(time.time()-t0,1)
    return mets

mA=run_arm('A')
with open(OUT/'L21_ARM_A_TRAINING_METRICS.json','w') as f: json.dump(mA,f,indent=2)
print('ARM A done',mA['total_sec'],flush=True)
mB=run_arm('B')
with open(OUT/'L21_ARM_B_TRAINING_METRICS.json','w') as f: json.dump(mB,f,indent=2)
print('ARM B done',mB['total_sec'],flush=True)

# caption sensitivity post-training (arm B final)
sens_post={}
for tag,m in [('A',mA),('B',mB)]:
    for ckpt in ['250','750']:
        sd=torch.load(OUT/f'{tag}_step{ckpt}.pt',map_location='cpu')
        su,tp=build_model(); su.load_state_dict(sd['su']); tp.load_state_dict(sd['tokproj']); su.eval(); tp.eval(); te=build_te()
        g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); tt=torch.tensor([int(ts[12])])
        with torch.no_grad():
            p0=forward_eps(su,tp,te,xn,tt,subset[0]['caption'])
            p1=forward_eps(su,tp,te,xn,tt,subset[1]['caption'])
            pz=forward_eps(su,tp,te,xn,tt,subset[0]['caption'],zero_text=True)
        sens_post[f'{tag}_s{ckpt}']={'swap_mean_abs':float((p0-p1).abs().mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(p0.flatten(),p1.flatten(),dim=0)),
                                     'zero_ablate':float((p0-pz).abs().mean())}
comp={}
for arm,m in [('A',mA),('B',mB)]:
    for tg,ev in m['evals'].items():
        ms=[v['mse_to_gt'] for v in ev.values()]
        comp.setdefault(arm,{})[tg]={'mse_mean':float(np.mean(ms)),'mse_min':float(np.min(ms))}
with open(OUT/'L21_CAPTION_BINDING_EVALUATION.json','w') as f: json.dump({'evals':comp,'post_sensitivity':sens_post},f,indent=2)
with open(OUT/'L21_ARM_COMPARISON.json','w') as f: json.dump({'evals':comp,'sensitivity':sens_post,'binding_B':mB.get('binding',[])[-3:] if mB.get('binding') else None},f,indent=2)
with open(OUT/'L21_TEXT_CHANNEL_ABLATION.json','w') as f: json.dump({'note':'zero_text ablation deltas in post_sensitivity.zero_ablate'},f,indent=2)
print('DONE ALL')
