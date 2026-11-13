"""L.16: extended spatial training - UNet-only, frozen TE, bs4, lr2e-4, 2000 steps."""
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

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l16_extended_spatial_training')
IMG = OUT/'images'; IMG.mkdir(parents=True, exist_ok=True)
CK = OUT/'checkpoints'; CK.mkdir(exist_ok=True)
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')

vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
    num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
    text_dim=256, n_heads=4, use_cross_attn=True)
su.load_state_dict(ck['model_state_dict'])
te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict'])
for p in te.parameters(): p.requires_grad_(False)
te.eval()
# verify freeze
assert all(not p.requires_grad for p in te.parameters())
assert all(p.requires_grad for p in su.parameters())
te_ref = {n: p.detach().clone() for n,p in te.state_dict().items()}

ac, ts = build_diffusion_schedule()
betas = torch.linspace(0.0001,0.02,1000); abars = torch.cumprod(1-betas,0)

meta = json.load(open(base/'data/text_image/train_metadata.json'))
train_meta, val_meta = meta[:700], meta[700:800]
def load_img(item):
    a = np.array(Image.open(base/item['image_path'])).astype(np.float32)/255.
    return torch.from_numpy(a).permute(2,0,1)*2-1

# fixed val set: 16 imgs, fixed noise+timesteps
torch.manual_seed(123)
val_data=[]
for it in val_meta[:16]:
    img=load_img(it); tt=torch.randint(0,1000,(1,)); nz=torch.randn(1,3,64,64)
    a=abars[tt]; val_data.append((it,img,tt,nz,torch.sqrt(a)*img.unsqueeze(0)+torch.sqrt(1-a)*nz))

def encode_batch(prompts):
    ids_list=[tok.encode(p,max_length=64) for p in prompts]
    L=max(len(i[0]) for i in ids_list)
    ids=torch.zeros(len(prompts),L,dtype=torch.long); mk=torch.zeros(len(prompts),L)
    for i,(ii,mm) in enumerate(ids_list):
        ids[i,:len(ii)]=torch.tensor(ii); mk[i,:len(mm)]=torch.tensor(mm,dtype=torch.float)
    with torch.no_grad(): return te(ids,mk)

EVAL_PROMPTS=[('red_circle','A red circle on a white background'),('blue_square','A blue square on a white background'),
('yellow_triangle','A yellow triangle on a white background'),('green_circle','A green circle at the center'),
('red_square','A red square on a white background'),('train_ltr','A medium blue circle slowly moving from left to right on a white background.'),
('train_rtl','A large pink triangle quickly moving from right to left on a white background.'),
('train_stat','A medium cyan square on a gray background, remaining stationary.'),
('val1',val_meta[0]['caption']),('val2',val_meta[1]['caption']),('val3',val_meta[2]['caption']),('val4',val_meta[3]['caption'])]

def ddim(su,te,prompt,seed):
    e,mk=encode_batch([prompt])
    g=torch.Generator().manual_seed(seed); x=torch.randn(1,3,64,64,generator=g)
    for i in range(len(ts)-1):
        t,tp=ts[i],ts[i+1]
        with torch.no_grad(): eps=su(x,torch.tensor([int(t)]).long(),e,mk)
        a_t,a_p=ac[t],ac[tp]
        x=torch.sqrt(a_p)*(x-torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t)+torch.sqrt(1-a_p)*eps
    return (x.clamp(-1,1)+1)/2

def sample_eval(tag):
    su.eval()
    res={}
    for name,prompt in EVAL_PROMPTS:
        img=ddim(su,te,prompt,5000)
        d=IMG/f'{tag}_{name}'; d.mkdir(exist_ok=True)
        Image.fromarray((img[0].permute(1,2,0).numpy()*255).astype(np.uint8)).save(d/'image.png')
        rgb=img[0].permute(1,2,0).numpy()
        res[name]={'corner_std':float(np.concatenate([rgb[:8,:8].reshape(-1,3),rgb[-8:,-8:].reshape(-1,3)]).std()),
                   'mean_rgb':rgb.reshape(-1,3).mean(0).round(3).tolist(),'std':float(rgb.std())}
    su.train()
    return res

def val_loss():
    su.eval(); vl=[]
    with torch.no_grad():
        for it,img,tt,nz,xt in val_data:
            e,mk=encode_batch([it['caption']])
            vl.append(float(torch.nn.functional.mse_loss(su(xt,tt,e,mk),nz)))
    su.train(); return vl

metrics={'config':{'init':'stage2 checkpoint','trainable':'UNet only','frozen':'text encoder','opt':'AdamW lr2e-4 wd0.01 new optimizer','bs':4,'steps':2000},
 'steps':[],'val':[],'evals':{}}
opt=optim.AdamW(list(su.parameters()),lr=0.0002,weight_decay=0.01)
g=torch.Generator().manual_seed(42)
su.train(); t0=time.time(); N=2000
EVAL_AT={0,250,500,1000,1500,2000}
CK_AT={0,250,500,750,1000,1250,1500,1750,2000}
# step0 eval
metrics['evals']['0']=sample_eval('step0')
for step in range(N):
    items=[train_meta[(step*4+j*97+step//7)%len(train_meta)] for j in range(4)]
    imgs=torch.stack([load_img(it) for it in items])
    t=torch.randint(0,1000,(4,))
    noise=torch.randn(4,3,64,64,generator=g)
    a=abars[t].view(4,1,1,1); xt=torch.sqrt(a)*imgs+torch.sqrt(1-a)*noise
    caps=[it['caption'] for it in items]
    e,mk=encode_batch(caps)
    pred=su(xt,t,e,mk)
    loss=torch.nn.functional.mse_loss(pred,noise)
    opt.zero_grad(); loss.backward()
    gn=sum(p.grad.norm().item() for p in su.parameters() if p.grad is not None)
    opt.step()
    s=step+1
    if s%10==0 or s==1:
        metrics['steps'].append({'step':s,'loss':float(loss),'grad_norm':gn,'sec':round(time.time()-t0,1)})
        if s%50==0: print(f's{s} loss={float(loss):.4f} gn={gn:.2f} {time.time()-t0:.0f}s',flush=True)
    if s in CK_AT:
        torch.save({'model_state_dict':su.state_dict(),'optimizer_state_dict':opt.state_dict(),'step':s,'config':'L16 bs4 lr2e-4'},CK/f'step{s}.pt')
        # TE integrity check
        te_ok=all(torch.equal(p,te_ref[n]) for n,p in te.state_dict().items())
        metrics.setdefault('te_integrity',{})[str(s)]=te_ok
    if s in EVAL_AT:
        vl=val_loss(); metrics['val'].append({'step':s,'val_mean':float(np.mean(vl)),'val_std':float(np.std(vl)),'val_min':float(np.min(vl)),'val_max':float(np.max(vl))})
        metrics['evals'][str(s)]=sample_eval(f'step{s}')
        print(f'  EVAL s{s} val={np.mean(vl):.4f}',flush=True)
# final integrity
te_ok=all(torch.equal(p,te_ref[n]) for n,p in te.state_dict().items())
metrics['final_te_bit_identical']=te_ok
metrics['total_sec']=round(time.time()-t0,1)
with open(OUT/'L16_TRAINING_METRICS.json','w') as f: json.dump(metrics,f,indent=2)
print('DONE',(time.time()-t0)/60,'min te_ok',te_ok)
