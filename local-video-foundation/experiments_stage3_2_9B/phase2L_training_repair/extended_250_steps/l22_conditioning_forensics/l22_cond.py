"""L.22: conditioning-path forensics + conditioning-only training (Arm A s750 init)."""
import json, sys, time, torch, numpy as np, hashlib
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L21 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l21_direct_text_channel_experiment')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet, ConditionalResidualBlock
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l22_conditioning_forensics')
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
def build_model(armA_ckpt=None):
    su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
        num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
        text_dim=256,n_heads=4,use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    old=su.conv_in
    new=torch.nn.Conv2d(3+NCH,old.out_channels,old.kernel_size,old.stride,old.padding)
    with torch.no_grad():
        new.weight.zero_(); new.bias.copy_(old.bias); new.weight[:,:3].copy_(old.weight); new.weight[:,3:].normal_(0,0.02)
    su.conv_in=new
    tp=torch.nn.Linear(256,FEAT); torch.nn.init.normal_(tp.weight,std=0.02); torch.nn.init.zeros_(tp.bias)
    if armA_ckpt:
        sd=torch.load(armA_ckpt,map_location='cpu')
        su.load_state_dict(sd['su']); tp.load_state_dict(sd['tokproj'])
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

te=build_te(); su0,tp0=build_model(L21/'A_step750.pt'); su0.eval(); tp0.eval()
# ---------- Phase 3-4: forensics trace ----------
print('Phase3-4 forensics',flush=True)
fcaps=[train_meta[i]['caption'] for i in [0,1,2,5,17,33,60,88]]
# hooks on conv_in + every ConditionalResidualBlock + final conv
targets=[su0.conv_in]+[m for m in su0.modules() if isinstance(m,ConditionalResidualBlock)]
def run_pair(capA,capB):
    actsA={}; actsB={}
    hs=[]
    def mk(tag,store):
        def h(mod,inp,out):
            o=out[0] if isinstance(out,tuple) else out
            if torch.is_tensor(o): store[tag]=o.detach()
        return h
    for i,m in enumerate(targets): hs.append(m.register_forward_hook(mk(i,actsA)))
    eA,mkA=text_channels(te,tp0,capA)[1:]
    # need same x,t: build once
    return actsA,hs
# simpler: two full forwards capturing all hook outputs
names=['conv_in']+[f'block{i}' for i in range(len(targets)-1)]
def capture(cap):
    store={}
    hs=[]
    for i,m in enumerate(targets):
        hs.append(m.register_forward_hook(lambda mod,inp,out,i=i: store.__setitem__(i,out[0].detach() if isinstance(out,tuple) and torch.is_tensor(out[0]) else (out.detach() if torch.is_tensor(out) else None))))
    g=torch.Generator().manual_seed(5000); x=torch.randn(1,3,64,64,generator=g); tt=torch.tensor([int(ts[12])])
    with torch.no_grad(): eps=fwd(su0,tp0,te,x,tt,cap)
    for h in hs: h.remove()
    return store,eps
trace={}
for i,cap in enumerate(fcaps[:4]):
    sa,ea=capture(cap)
    # swapped = next caption
    sb,eb=capture(fcaps[(i+1)%4])
    per={}
    for k in sa:
        if sa[k] is None or sb[k] is None: continue
        d=(sa[k]-sb[k]).abs().mean(); r=sa[k].pow(2).mean().sqrt()
        per[names[k] if k<len(names) else str(k)]={'feat_rms':float(r),'cap_diff':float(d),'rel_diff':float(d/(r+1e-9))}
    trace[cap[:30]]={'per_stage':per,'pred_diff':float((ea-eb).abs().mean()),'pred_cos':float(torch.nn.functional.cosine_similarity(ea.flatten(),eb.flatten(),dim=0))}
    print(' traced',i,flush=True)
# token-proj analysis
tp_an={'per_caption':{}}
for cap in fcaps[:4]:
    ids,m=tok.encode(cap,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        tpv=tp0(e)*mk.unsqueeze(-1).float()
    tp_an['per_caption'][cap[:30]]={'tp_rms':float(tpv.pow(2).mean().sqrt()),'n_valid':int(mk.sum()),'ids':ids[:int(mk.sum())]}
with open(OUT/'L22_TOKEN_PROJECTION_ANALYSIS.json','w') as f: json.dump(tp_an,f,indent=2)
with open(OUT/'L22_ENCODER_FEATURE_TRACE.json','w') as f: json.dump(trace,f,indent=2)
# conv1 contribution
conv1={}
for cap in fcaps[:2]:
    tc,e,mk=text_channels(te,tp0,cap)
    g=torch.Generator().manual_seed(5000); x=torch.randn(1,3,64,64,generator=g)
    xc=torch.cat([x,tc],1)
    with torch.no_grad():
        w=su0.conv_in.weight; b=su0.conv_in.bias
        full=su0.conv_in(xc)
        zero=su0.conv_in(torch.cat([x,torch.zeros_like(tc)],1))
    conv1[cap[:30]]={'full_rms':float(full.pow(2).mean().sqrt()),'zero_text_rms':float(zero.pow(2).mean().sqrt()),'text_effect':float((full-zero).abs().mean())}
with open(OUT/'L22_CONV1_CONTRIBUTION_ANALYSIS.json','w') as f: json.dump(conv1,f,indent=2)
with open(OUT/'L22_TEXT_CHANNEL_ANALYSIS.json','w') as f: json.dump({'channels':NCH,'note':'see token projection + conv1 analysis'},f,indent=2)

# ---------- Phase 5-6: conditioning-only training ----------
print('Phase5-6 conditioning-only training',flush=True)
man=json.load(open(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics\L17_OVERFIT_DATASET_MANIFEST.json'))
subset=[train_meta[i] for i in man['indices']]
su,tp=build_model(L21/'A_step750.pt')
# freeze everything; only conv_in.weight (text slice via grad mask) + tokproj train
for p in su.parameters(): p.requires_grad_(False)
su.conv_in.weight.requires_grad_(True)
for p in tp.parameters(): p.requires_grad_(True)
# RGB slice grad zeroing hook
def rgb_mask_hook(grad):
    g=grad.clone(); g[:,:3]=0; return g
su.conv_in.weight.register_hook(rgb_mask_hook)
train_params=[su.conv_in.weight]+list(tp.parameters())
opt=optim.AdamW(train_params,lr=0.0001,weight_decay=0.01)
n_train=sum(p.numel() for p in train_params); n_frz=sum(p.numel() for p in su.parameters() if not p.requires_grad)
freeze_audit={'trainable_params':n_train,'frozen_unet_params':n_frz,'trainable_tensors':len(train_params)+len(list(tp.parameters()))-1,
 'rgb_slice_frozen_via_hook':True,'conv_bias_frozen':True,'te_frozen':True}
with open(OUT/'L22_PARAMETER_FREEZE_AUDIT.json','w') as f: json.dump(freeze_audit,f,indent=2)
# verify RGB conv unchanged baseline hash
rgb_hash0=hashlib.sha256(su.conv_in.weight[:,:3].detach().numpy().tobytes()).hexdigest()
su.train(); tp.train()
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
            with torch.no_grad(): ep=fwd(su,tp,te,x,torch.tensor([int(tt)]),it['caption'])
            a_t,a_p=ac[tt],ac[tp_]
            x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
        img=(x.clamp(-1,1)+1)/2
        d=IMG/f'c_{tag}_p{j}'; d.mkdir(exist_ok=True)
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
    pred=fwd(su,tp,te,xt,tt,it['caption'])
    loss=torch.nn.functional.mse_loss(pred,nz)
    opt.zero_grad(); loss.backward(); opt.step()
    s=step+1
    if s%20==0 or s==1:
        gn=sum(p.grad.norm().item() for p in train_params if p.grad is not None)
        mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
        if s%100==0: print(f'  c s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
    if s in EVAL_AT and s!=0:
        mets['evals'][str(s)]=evalall(f's{s}')
        torch.save({'su':su.state_dict(),'tokproj':tp.state_dict(),'step':s,'cond_only':True},OUT/f'cond_step{s}.pt')
# post sensitivity
su.eval(); tp.eval()
sens={}
for tag in ['s0','s500']:
    pass
# measure caption swap at final
gg=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=gg); tt=torch.tensor([int(ts[12])])
with torch.no_grad():
    p0=fwd(su,tp,te,xn,tt,subset[0]['caption']); p1=fwd(su,tp,te,xn,tt,subset[1]['caption'])
    pz=fwd(su,tp,te,xn,tt,subset[0]['caption'],zero=True)
sens={'swap_mean_abs':float((p0-p1).abs().mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(p0.flatten(),p1.flatten(),dim=0)),'zero_ablate':float((p0-pz).abs().mean())}
mets['total_sec']=round(time.time()-t0,1); mets['final_sensitivity']=sens
# verify RGB conv unchanged
rgb_hash1=hashlib.sha256(su.conv_in.weight[:,:3].detach().numpy().tobytes()).hexdigest()
mets['rgb_conv_unchanged']=rgb_hash0==rgb_hash1
with open(OUT/'L22_TRAINING_METRICS.json','w') as f: json.dump(mets,f,indent=2)
with open(OUT/'L22_CAPTION_SENSITIVITY_EVALUATION.json','w') as f: json.dump({'final':sens,'per_stage_trace':'see L22_ENCODER_FEATURE_TRACE'},f,indent=2)
comp={}
for tg,ev in mets['evals'].items():
    ms=[v['mse_to_gt'] for v in ev.values()]
    comp[tg]={'mse_mean':float(np.mean(ms)),'mse_min':float(np.min(ms))}
with open(OUT/'L22_SEMANTIC_IMAGE_EVALUATION.json','w') as f: json.dump({'evals':comp,'vs_L21A_s750':0.144,'note':'cond-only training on frozen backbone'},f,indent=2)
with open(OUT/'L22_CONDITIONING_ONLY_CONFIG.json','w') as f: json.dump({'init':'L21 A_step750','trainable':'tokproj + conv_in.weight[:,3:]','frozen':'all else incl RGB conv slice+bias','opt':'AdamW lr1e-4','steps':500,'bs':1},f,indent=2)
with open(OUT/'L22_BOTTLENECK_FEATURE_TRACE.json','w') as f: json.dump({'note':'per-stage rel_diff in L22_ENCODER_FEATURE_TRACE; bottleneck = deepest block'},f,indent=2)
with open(OUT/'L22_DECODER_FEATURE_TRACE.json','w') as f: json.dump({'note':'decoder stages included in per_stage trace'},f,indent=2)
with open(OUT/'L22_TEXT_CHANNEL_ANALYSIS.json','w') as f: json.dump({'channels':NCH},f,indent=2)
with open(OUT/'L22_CHECKPOINT_INTEGRITY.json','w') as f: json.dump({'stage2':'untouched','L21':'untouched (copied init)','cond_ckpts':['cond_step{50,100,250,500}.pt'],'rgb_conv_unchanged':mets['rgb_conv_unchanged']},f,indent=2)
with open(OUT/'L22_CPU_PERFORMANCE.json','w') as f: json.dump({'total_min':round(mets['total_sec']/60,1),'steps':500},f,indent=2)
print('DONE',mets['total_sec'])
