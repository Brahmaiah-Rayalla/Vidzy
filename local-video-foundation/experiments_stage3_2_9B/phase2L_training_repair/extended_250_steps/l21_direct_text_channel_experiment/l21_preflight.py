"""L.21 preflight corrections: baselines, bit-identity, gradient audit, ablation."""
import json, sys, torch, numpy as np, hashlib
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

def build_model():
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

report={}
# 1) RGB weight bit-identity
su,tp=build_model()
orig=ck['model_state_dict']['conv_in.weight']
report['rgb_weight_bit_identical']=bool(torch.equal(su.conv_in.weight[:,:3].detach(),orig))
report['rgb_bias_bit_identical']=bool(torch.equal(su.conv_in.bias.detach(),ck['model_state_dict']['conv_in.bias']))
# 2) four baselines on one fixed input
te=build_te(); it=train_meta[0]
g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); tt=torch.tensor([int(ts[12])])
# B0: original stage2 native cross-attn
suN=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,text_dim=256,n_heads=4,use_cross_attn=True)
suN.load_state_dict(ck['model_state_dict']); suN.eval()
ids,m=tok.encode(it['caption'],max_length=64)
with torch.no_grad():
    e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
    p_native=suN(xn,tt,e,mk)
# B1-B3: L21 variants
su.eval(); tp.eval()
with torch.no_grad():
    p_zero=fwd(su,tp,te,xn,tt,it['caption'],zero=True)
    p_corr=fwd(su,tp,te,xn,tt,it['caption'])
    p_wrong=fwd(su,tp,te,xn,tt,train_meta[1]['caption'])
report['baselines']={'native_crossattn_vs_L21zero':float((p_native-p_zero).abs().mean()),
 'L21_zero_vs_correct':float((p_zero-p_corr).abs().mean()),
 'correct_vs_wrong':float((p_corr-p_wrong).abs().mean()),
 'native_vs_correct':float((p_native-p_corr).abs().mean())}
# 3) gradient audit: one train step, per-group norms
su.train(); tp.train()
img=load_img(it); t=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64)
a=abars[t].view(1,1,1,1); xt=torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz
pred=fwd(su,tp,te,xt,t,it['caption'])
loss=torch.nn.functional.mse_loss(pred,nz); loss.backward()
groups={'tokproj':list(tp.parameters()),'conv_in_weight':[su.conv_in.weight],'conv_in_bias':[su.conv_in.bias],'other_unet':[p for n,p in su.named_parameters() if 'conv_in' not in n]}
gn={}
for k,v in groups.items():
    ps=v if isinstance(v,list) else [v]
    gn[k]=float(sum(p.grad.norm().item() for p in ps if p.grad is not None))
tp_grad_nonzero=gn['tokproj']>0; conv_grad_nonzero=gn['conv_in_weight']>0
gn['conv_in_text_slice_norm']=float(su.conv_in.weight.grad[:,3:].norm()) if su.conv_in.weight.grad is not None else 0.0
gn['conv_in_rgb_slice_norm']=float(su.conv_in.weight.grad[:,:3].norm()) if su.conv_in.weight.grad is not None else 0.0
report['gradient_audit']={'groups':gn,'tokproj_nonzero':tp_grad_nonzero,'conv_in_nonzero':conv_grad_nonzero,
 'text_slice_grad_norm':gn['conv_in_text_slice_norm'],'rgb_slice_grad_norm':gn['conv_in_rgb_slice_norm'],
 'loss':float(loss)}
# 4) text-channel gradient ablation: zero channels -> does loss change vs correct?
su.eval(); tp.eval()
with torch.no_grad():
    p_c=fwd(su,tp,te,xt,t,it['caption']); p_z=fwd(su,tp,te,xt,t,it['caption'],zero=True)
report['ablation']={'correct_vs_zeroed_pred_diff':float((p_c-p_z).abs().mean())}
# 5) TE bit-identity
te2=build_te()
same=all(torch.equal(p,q) for (n,p),(n2,q) in zip(te.state_dict().items(),te2.state_dict().items()))
report['te_bit_identical']=same
with open(OUT/'L21_PREFLIGHT_AUDIT.json','w') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=1))
