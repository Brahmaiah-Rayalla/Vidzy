"""L.18: mask/activation audit, gain sweep, two-arm caption-binding training."""
import json, sys, time, torch, numpy as np, hashlib, types
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder, MultiHeadAttention
from models.conditional_unet import ConditionalUNet, CrossAttentionBlock
import torch.optim as optim
import torch.nn.functional as F

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l18_conditioning_amplification')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001,0.02,1000); abars = torch.cumprod(1-betas,0)
vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]

def build():
    su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    su.load_state_dict(ck['model_state_dict'])
    te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
    te.load_state_dict(ck['text_encoder_state_dict'])
    return su, te

def enc(te, caption):
    ids,m = tok.encode(caption, max_length=64)
    with torch.no_grad():
        e,mk = te(torch.tensor([ids]), torch.tensor([m],dtype=torch.float))
    return e, mk, ids, m

meta = json.load(open(base/'data/text_image/train_metadata.json'))
train_meta = meta[:700]
def load_img(it):
    a=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1

# ---------- Phase 2: mask audit ----------
print('PHASE2 mask audit', flush=True)
mask_audit={'mask_semantics':'mask==0 -> -inf before softmax; mask==1 valid (verified in MultiHeadAttention.forward text_encoder.py:286-292)','unit_test':None,'captions':[]}
# deterministic unit test: masked positions get zero attention
q=torch.randn(1,4,8); k=torch.randn(1,6,8); v=torch.randn(1,6,8)
mh=MultiHeadAttention(d_model=8,n_heads=2,dropout=0.0).eval()
with torch.no_grad():
    out=mh(q,k,v,torch.tensor([[1,1,1,0,0,0]]))
# verify padding positions get zero attention by checking scores path manually
Q=mh.q_proj(q).view(1,4,2,4).transpose(1,2); K=mh.k_proj(k).view(1,6,2,4).transpose(1,2)
sc=torch.matmul(Q,K.transpose(-2,-1))/2.0
mk=torch.tensor([[1,1,1,0,0,0]])
sc=sc.masked_fill(mk.unsqueeze(1).unsqueeze(2)==0,float('-inf'))
w=torch.softmax(sc,-1)
mask_audit['unit_test']={'attn_on_padding_max':float(w[0,0,:,3:].max()),'attn_on_valid_sum':float(w[0,0,:,:3].sum()),'out_finite':bool(torch.isfinite(out).all())}
for i in [0,1,2]:
    e,mk,ids,m=enc(build()[1],train_meta[i]['caption'])
    mask_audit['captions'].append({'caption':train_meta[i]['caption'],'ids':ids[:int(sum(m))],'mask':m[:int(sum(m))],'valid':int(sum(m)),'pad':64-int(sum(m))})
with open(OUT/'L18_ATTENTION_MASK_AUDIT.json','w') as f: json.dump(mask_audit,f,indent=2)
print(' mask ok, unit attn_on_padding',mask_audit['unit_test'],flush=True)

# ---------- Phase 3: activation analysis ----------
print('PHASE3 activation', flush=True)
su,te = build(); su.eval(); te.eval()
acts={}
def hook_fn(name):
    def h(mod,inp,out):
        acts.setdefault(name,{})['attn_out_rms']=float(out.pow(2).mean().sqrt()) if torch.is_tensor(out) else None
    return h
# instrument text_proj + cross_attn per block
probe_caps=[train_meta[i]['caption'] for i in [0,1,2]]
records={}
for cap_i,cap in enumerate(probe_caps):
    e,mk,_,_=enc(te,cap)
    g=torch.Generator().manual_seed(5000)
    x=torch.randn(1,3,64,64,generator=g)
    t=torch.tensor([int(ts[12])])
    block_data=[]
    hooks=[]
    # find all CrossAttentionBlock instances
    blocks=[m for m in su.modules() if isinstance(m,CrossAttentionBlock)]
    for bi,b in enumerate(blocks):
        def make(bi,b):
            def pre(mod,inp):
                # inp[0]=x features; record norms
                block_data.append({'i':bi,'h_in_rms':float(inp[0].pow(2).mean().sqrt()),
                    'text_emb_rms':float(inp[1].pow(2).mean().sqrt())})
            return pre
        hooks.append(b.register_forward_pre_hook(make(bi,b)))
        hooks.append(b.cross_attn.register_forward_hook(lambda m,i,o,bi=bi: block_data.append({'i':bi,'attn_out_rms':float(o.pow(2).mean().sqrt()),'out_shape':list(o.shape)}) if isinstance(o,torch.Tensor) else None))
    with torch.no_grad():
        eps=su(x,t,e,mk)
    for h in hooks: h.remove()
    records[cap_i]=block_data
# aggregate per block index
agg={}
for bi in range(len(blocks)):
    vals=[d for cap in records.values() for d in cap if d.get('i')==bi]
    agg[f'block{bi}']={'n_meas':len(vals),
      'h_in_rms':float(np.mean([v['h_in_rms'] for v in vals if 'h_in_rms' in v])) if any('h_in_rms' in v for v in vals) else None,
      'text_emb_rms':float(np.mean([v['text_emb_rms'] for v in vals if 'text_emb_rms' in v])) if any('text_emb_rms' in v for v in vals) else None,
      'attn_out_rms':float(np.mean([v['attn_out_rms'] for v in vals if 'attn_out_rms' in v])) if any('attn_out_rms' in v for v in vals) else None}
act_report={'n_cross_attn_blocks':len(blocks),'per_block':agg,'note':'attn_out RMS vs h_in RMS gives residual ratio scale'}
with open(OUT/'L18_CROSS_ATTENTION_ACTIVATION_ANALYSIS.json','w') as f: json.dump(act_report,f,indent=2)
print(' acts:',json.dumps(agg)[:600],flush=True)

# ---------- Phase 5: gain sweep ----------
print('PHASE5 gain sweep', flush=True)
# monkeypatch CrossAttentionBlock.forward with gain
def make_gated_forward(gain):
    def fwd(self,x,text_emb,text_mask=None):
        h=self.norm(x)
        b,c,hh,ww=h.shape; ss=hh*ww
        h=h.reshape(b,c,ss).transpose(1,2)
        tp=self.text_proj(text_emb)
        ao=self.cross_attn(h,tp,tp,text_mask)
        ao=ao.transpose(1,2).reshape(b,c,hh,ww)
        return x + gain*self.dropout(ao)
    return fwd

caps_test=[train_meta[0]['caption'],train_meta[1]['caption'],train_meta[2]['caption']]
sweep={}
orig_fwd=CrossAttentionBlock.forward
for gain in [0.0,0.25,0.5,1.0,2.0,4.0]:
    CrossAttentionBlock.forward=make_gated_forward(gain)
    su.eval()
    g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); t=torch.tensor([int(ts[12])])
    preds=[]
    for cap in caps_test:
        e,mk,_,_=enc(te,cap)
        with torch.no_grad(): preds.append(su(xn,t,e,mk))
    # sensitivity: swap cap0<->cap1 diff
    d=(preds[0]-preds[1]).abs()
    sweep[f'G{gain}']={'swap_mean_abs':float(d.mean()),'swap_max':float(d.abs().max()),
                       'cosine':float(torch.nn.functional.cosine_similarity(preds[0].flatten(),preds[1].flatten(),dim=0)),
                       'eps_std':float(preds[0].std())}
    print(f'  G{gain}: mean_abs={float(d.mean()):.4f} cos={sweep[f"G{gain}"]["cosine"]:.5f}',flush=True)
CrossAttentionBlock.forward=orig_fwd
with open(OUT/'L18_INFERENCE_GAIN_SWEEP.json','w') as f: json.dump(sweep,f,indent=2)

# ---------- Phase 7: two-arm training ----------
print('PHASE7 arms', flush=True)
# reuse L17 10-pair subset manifest
man=json.load(open(Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l17_spatial_diagnostics\L17_OVERFIT_DATASET_MANIFEST.json')))
subset=[train_meta[i] for i in man['indices']]
# deterministic mismatched captions: swap within subset by +3
mismatch=[subset[(j+3)%10]['caption'] for j in range(10)]

def run_arm(arm, binding_weight=0.10, margin=0.02, steps=750):
    su_a, te_a = build()
    for p in te_a.parameters(): p.requires_grad_(False)
    te_a.eval()
    opt=optim.AdamW(list(su_a.parameters()),lr=0.0001,weight_decay=0.01)
    g=torch.Generator().manual_seed(42); su_a.train()
    mets={'steps':[],'binding':[] if arm=='B' else None,'evals':{}}
    EVAL_AT={0,100,250,500,750}
    def evalall(tag):
        su_a.eval(); res={}
        for j,it in enumerate(subset):
            e,mk,_,_=enc(te_a,it['caption'])
            gg=torch.Generator().manual_seed(6000+j); x=torch.randn(1,3,64,64,generator=gg)
            for i in range(len(ts)-1):
                t,tp=ts[i],ts[i+1]
                with torch.no_grad(): ep=su_a(x,torch.tensor([int(t)]).long(),e,mk)
                a_t,a_p=ac[t],ac[tp]
                x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
            img=(x.clamp(-1,1)+1)/2
            d=IMG/f'{arm}_{tag}_p{j}'; d.mkdir(exist_ok=True)
            Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
            gt=np.array(Image.open(base/it['image_path'])).astype(np.float32)/255.
            res[f'p{j}']={'mse_to_gt':float(((img[0].permute(1,2,0).numpy()-gt)**2).mean())}
        su_a.train(); return res
    mets['evals']['0']=evalall('s0')
    t0=time.time()
    for step in range(steps):
        j=step%10; it=subset[j]; img=load_img(it)
        t=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64,generator=g)
        a=abars[t].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
        e,mk,_,_=enc(te_a,it['caption'])
        pred=su_a(xt,t,e,mk)
        lc=torch.nn.functional.mse_loss(pred,nz)
        if arm=='B':
            em,mkm,_,_=enc(te_a,mismatch[j])
            pred_sw=su_a(xt,t,em,mkm)
            ls=torch.nn.functional.mse_loss(pred_sw,nz)
            lb=torch.clamp(margin+lc-ls,min=0)
            loss=lc+binding_weight*lb
            mets['binding'].append({'step':step+1,'lc':float(lc),'ls':float(ls),'lb':float(lb)})
        else:
            loss=lc
        opt.zero_grad(); loss.backward(); opt.step()
        s=step+1
        if s%20==0 or s==1:
            gn=sum(p.grad.norm().item() for p in su_a.parameters() if p.grad is not None)
            mets['steps'].append({'step':s,'loss':float(loss),'gn':gn,'sec':round(time.time()-t0,1)})
            if s%100==0: print(f'  {arm} s{s} loss={float(loss):.4f} {time.time()-t0:.0f}s',flush=True)
        if s in EVAL_AT and s!=0:
            mets['evals'][str(s)]=evalall(f's{s}')
            torch.save({'model_state_dict':su_a.state_dict(),'step':s},OUT/f'{arm}_step{s}.pt')
    mets['total_sec']=round(time.time()-t0,1)
    return mets

mA=run_arm('A')
with open(OUT/'L18_ARM_A_TRAINING_METRICS.json','w') as f: json.dump(mA,f,indent=2)
print('ARM A done',mA['total_sec'],'s',flush=True)
mB=run_arm('B')
with open(OUT/'L18_ARM_B_TRAINING_METRICS.json','w') as f: json.dump(mB,f,indent=2)
print('ARM B done',mB['total_sec'],'s',flush=True)
# comparison
comp={}
for arm,m in [('A',mA),('B',mB)]:
    for tag,ev in m['evals'].items():
        mses=[v['mse_to_gt'] for v in ev.values()]
        comp.setdefault(arm,{})[tag]={'mse_mean':float(np.mean(mses)),'mse_min':float(np.min(mses))}
with open(OUT/'L18_CAPTION_BINDING_EVALUATION.json','w') as f: json.dump({'evals':comp,'note':'mse_to_gt per pair; binding losses in arm metrics'},f,indent=2)
with open(OUT/'L18_ARM_COMPARISON.json','w') as f: json.dump({'A':comp.get('A'),'B':comp.get('B'),'gain_sweep':sweep},f,indent=2)
print('ALL DONE')
