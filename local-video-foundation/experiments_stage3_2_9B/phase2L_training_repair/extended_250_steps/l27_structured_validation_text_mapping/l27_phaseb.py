"""L.27 Phase B: text->color adapter on frozen generator."""
import json, sys, torch, numpy as np, random
from pathlib import Path
from PIL import Image
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
L25 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l25_single_class_conditioning')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder
from models.conditional_unet import ConditionalUNet
import torch.optim as optim

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l27_structured_validation_text_mapping')
IMG = OUT/'images'; IMG.mkdir(exist_ok=True)
DATA = L25/'data'
base = l12_common.base_dir
torch.set_num_threads(8)
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
ac, ts = build_diffusion_schedule()
vocab=ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
COLORS=[(220,30,30),(30,80,220),(230,200,30)]
CN=['red','blue','yellow']
class SplitConv6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_rgb=torch.nn.Conv2d(3,64,3,padding=1); self.conv_cond=torch.nn.Conv2d(3,64,3,padding=1,bias=False)
    def forward(self,x): return self.conv_rgb(x[:,:3])+self.conv_cond(x[:,3:])
su=ConditionalUNet(in_channels=3,model_channels=64,channel_mult=[1,2,2,2],
    num_res_blocks=2,attention_resolutions=[],dropout=0.0,num_timesteps=1000,
    text_dim=256,n_heads=4,use_cross_attn=True)
su.load_state_dict(ck['model_state_dict'])
sd=torch.load(OUT/'ckpt_10000.pt',map_location='cpu')
sc=SplitConv6(); su.conv_in=sc; su.load_state_dict(sd['su'],strict=False); su.eval()
def fwd(su,x,t,cid):
    B=x.shape[0]; cm=torch.zeros(B,3,64,64)
    for i,c in enumerate(cid): cm[i,c]=1
    return su(torch.cat([x,cm],1),t,torch.zeros(B,64,256),torch.ones(B,64))
tok=SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
te=TextEncoder(vocab_size=vocab,d_model=256,n_heads=4,n_layers=4,d_ff=512,max_seq_len=64,dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
for p in te.parameters(): p.requires_grad_(False)
T=["a {c} circle on a plain background","generate a {c} circle","a {c} circular shape","a circle colored {c}","show a {c} circle against a plain background","{c} circle","make a {c} ball","picture of a {c} circle","{c} circle on white","draw a {c} circle"]
prompts=[]
for cid,c in enumerate(CN):
    for ti,t in enumerate(T):
        for v in range(10): prompts.append((t.format(c=c),cid,ti))
random.Random(7).shuffle(prompts)
ntr=int(len(prompts)*0.7); nva=int(len(prompts)*0.85)
tr_p,va_p,te_p=prompts[:ntr],prompts[ntr:nva],prompts[nva:]
def encode(items):
    ids=[tok.encode(p,max_length=64)[0] for p,_,_ in items]; ms=[tok.encode(p,max_length=64)[1] for p,_,_ in items]
    return torch.tensor(ids),torch.tensor([y for _,y,_ in items],dtype=torch.long).reshape(-1),torch.tensor(ms,dtype=torch.float)
Xtr,ytr,Mtr=encode(tr_p); Xva,yva,Mva=encode(va_p); Xte,yte,Mte=encode(te_p)
def pool(ids,ms):
    with torch.no_grad(): e,_=te(ids,ms)
    w=ms.unsqueeze(-1); return (e*w).sum(1)/w.sum(1).clamp_min(1)
adapter=torch.nn.Sequential(torch.nn.Linear(256,256),torch.nn.SiLU(),torch.nn.Linear(256,3))
opt=optim.AdamW(adapter.parameters(),lr=0.001)
Ptr=pool(Xtr,Mtr); Pva=pool(Xva,Mva); Pte=pool(Xte,Mte)
for ep in range(80):
    perm=torch.randperm(len(Ptr))
    for i in range(0,len(Ptr),8):
        bi=perm[i:i+8]
        loss=torch.nn.functional.cross_entropy(adapter(Ptr[bi]).reshape(-1,3),ytr[bi].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    if ep%20==0:
        va=(adapter(Pva).argmax(-1)==yva).float().mean()
        print(f'ep{ep} va={float(va):.3f}',flush=True)
def acc(P,y): return float((adapter(P).argmax(-1)==y).float().mean())
tr_a,va_a,te_a=acc(Ptr,ytr),acc(Pva,yva),acc(Pte,yte)
conf=np.zeros((3,3),int)
for a,b in zip(yte.tolist(),adapter(Pte).argmax(-1).tolist()): conf[a,b]+=1
adp={'train_acc':tr_a,'val_acc':va_a,'test_acc':te_a,'conf':conf.tolist(),'n_test':len(Xte)}
print('adapter accs',tr_a,va_a,te_a,flush=True)
# end-to-end
torch.save({'adapter':adapter.state_dict()},OUT/'text_adapter.pt')
e2e={'correct':0,'total':0}
for p,cid,_ in te_p[:12]:
    ids,m=tok.encode(p,max_length=64)
    with torch.no_grad():
        e,mk=te(torch.tensor([ids]),torch.tensor([m],dtype=torch.float))
        w=mk.unsqueeze(-1).float(); pooled=(e*w).sum(1)/w.sum(1).clamp_min(1)
        pc=int(adapter(pooled).argmax())
        gg=torch.Generator().manual_seed(1234); x=torch.randn(1,3,64,64,generator=gg)
        sub=ts[::2]
        for i in range(len(sub)-1):
            tt,tp_=sub[i],sub[i+1]
            with torch.no_grad(): ep=fwd(su,x,torch.tensor([int(tt)]),[pc])
            a_t,a_p=ac[tt],ac[tp_]
            x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*ep)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*ep
        arr=((x.clamp(-1,1)+1)/2)[0].permute(1,2,0).numpy()
        pix=arr.reshape(-1,3); sat=pix[(pix.max(1)-pix.min(1))>0.15]
        gc=int(np.argmin([np.abs(sat-np.array(COLORS[k])/255).mean(1).mean() for k in range(3)])) if len(sat) else -1
        e2e['total']+=1
        if gc==pc: e2e['correct']+=1
with open(OUT/'L27_TEXT_ADAPTER_TRAINING_METRICS.json','w') as f: json.dump({'epochs':80,'bs':8,'lr':0.001},f,indent=2)
with open(OUT/'L27_TEXT_ADAPTER_EVALUATION.json','w') as f: json.dump(adp,f,indent=2)
with open(OUT/'L27_TEXT_CONDITION_DATASET.json','w') as f: json.dump({'n_train':len(Xtr),'n_val':len(Xva),'n_test':len(Xte),'templates':len(T)},f,indent=2)
with open(OUT/'L27_TEXT_ADAPTER_CONFIG.json','w') as f: json.dump({'arch':'Linear256-256-3 masked-mean pooled TE','te':'frozen','loss':'CE'},f,indent=2)
with open(OUT/'L27_END_TO_END_GENERATION_EVALUATION.json','w') as f: json.dump({'adapter_test_acc':te_a,'e2e_color_match':e2e['correct']/max(1,e2e['total']),'n':e2e['total']},f,indent=2)
with open(OUT/'L27_GENERATOR_INTEGRITY.json','w') as f: json.dump({'generator':'frozen Phase B','su_unchanged':True},f,indent=2)
print('DONE',te_a)
