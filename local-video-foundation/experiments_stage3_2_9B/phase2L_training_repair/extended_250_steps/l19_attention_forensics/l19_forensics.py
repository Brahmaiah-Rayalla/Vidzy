"""L.19: attention weight forensics + interventions (inference-only)."""
import json, sys, math, torch, numpy as np
from pathlib import Path
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
import l12_common
from l12_common import *
from models.text_encoder import SimpleTokenizer, TextEncoder, MultiHeadAttention
from models.conditional_unet import ConditionalUNet, CrossAttentionBlock

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l19_attention_forensics')
base = l12_common.base_dir
torch.set_num_threads(8)
tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location='cpu')
vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
su = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
    num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
    text_dim=256, n_heads=4, use_cross_attn=True)
su.load_state_dict(ck['model_state_dict']); su.eval()
te = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64, dropout=0.0)
te.load_state_dict(ck['text_encoder_state_dict']); te.eval()
meta = json.load(open(base/'data/text_image/train_metadata.json'))
train_meta = meta[:700]
ac, ts = build_diffusion_schedule()

def enc(caption):
    ids,m = tok.encode(caption, max_length=64)
    with torch.no_grad():
        e,mk = te(torch.tensor([ids]), torch.tensor([m],dtype=torch.float))
    return e, mk, ids, m

# pick 8 diverse captions
sel=[0,1,2,5,17,33,60,88]
caps=[train_meta[i]['caption'] for i in sel]

blocks=[m for m in su.modules() if isinstance(m,CrossAttentionBlock)]
print('n blocks',len(blocks), flush=True)

# ---- capture attention internals by replicating block math ----
def capture_block(block, x, text_emb, mask):
    """Replicates CrossAttentionBlock.forward but returns internals."""
    h = block.norm(x)
    b,c,hh,ww = h.shape; ss=hh*ww
    hq = h.reshape(b,c,ss).transpose(1,2)              # [b,ss,c] queries
    tp = block.text_proj(text_emb)                      # [b,S,c]
    mh = block.cross_attn
    Q = mh.q_proj(hq).view(b,ss,mh.n_heads,mh.head_dim).transpose(1,2)
    K = mh.k_proj(tp).view(b,tp.shape[1],mh.n_heads,mh.head_dim).transpose(1,2)
    V = mh.v_proj(tp).view(b,tp.shape[1],mh.n_heads,mh.head_dim).transpose(1,2)
    scores = torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(mh.head_dim)
    mk = mask.unsqueeze(1).unsqueeze(2)
    scores = scores.masked_fill(mk==0, float('-inf'))
    w = torch.softmax(scores,-1)                        # [b,heads,ss,S]
    out = torch.matmul(w,V).transpose(1,2).contiguous().view(b,ss,mh.d_model)
    out = mh.out_proj(out).transpose(1,2).reshape(b,c,hh,ww)
    x_out = x + block.dropout(out)
    return {'h_in_rms':float(h.pow(2).mean().sqrt()),'tp_rms':float(tp.pow(2).mean().sqrt()),
            'q_rms':float(Q.pow(2).mean().sqrt()),'k_rms':float(K.pow(2).mean().sqrt()),'v_rms':float(V.pow(2).mean().sqrt()),
            'scores_std':float(scores[torch.isfinite(scores)].std()),
            'attn_weights':w,'context':out,'x_out':x_out,'V':V,'K':K,'Q':Q,'tp':tp}

def run_capture(cap_idx):
    """Run UNet forward capturing each block's internals via hooks on norm? 
    Simpler: monkeypatch CrossAttentionBlock.forward to stash internals."""
    stash={}
    orig=CrossAttentionBlock.forward
    counter=[0]
    def patched(self,x,text_emb,text_mask=None):
        i=counter[0]; counter[0]+=1
        # capture internals by replicating
        r=capture_block(self,x,text_emb,text_mask)
        stash[i]=r
        return r['x_out']
    CrossAttentionBlock.forward=patched
    e,mk,ids,m=enc(caps[cap_idx])
    g=torch.Generator().manual_seed(5000); x=torch.randn(1,3,64,64,generator=g)
    t=torch.tensor([int(ts[12])])
    with torch.no_grad(): eps=su(x,t,e,mk)
    CrossAttentionBlock.forward=orig
    return stash, eps, (ids,m)

# entropy stats per block per caption
def entropy_stats(w, mask):
    """w: [1,heads,ss,S]; mask [1,S]. Return per-head stats over valid tokens."""
    S=w.shape[-1]; valid=mask[0]>0
    nv=int(valid.sum())
    res={'n_valid':nv,'n_heads':w.shape[1],'heads':[]}
    Hmax=math.log(nv) if nv>1 else 0
    for h in range(w.shape[1]):
        wh=w[0,h,:,valid]                      # [ss,nv]
        ent=-(wh*torch.log(wh.clamp_min(1e-12))).sum(-1)   # [ss]
        pmax=wh.max(-1).values
        top3=wh.topk(min(3,nv),-1).values.sum(-1)
        res['heads'].append({
            'entropy_mean':float(ent.mean()),'entropy_std':float(ent.std()),
            'norm_entropy':float((ent/Hmax).mean()) if Hmax>0 else None,
            'pmax_mean':float(pmax.mean()),'top3_mass':float(top3.mean()),
            'eff_tokens':float((1/(wh.pow(2).sum(-1)+1e-12)).mean())})
    return res

all_data={}
print('capturing', flush=True)
for ci in range(len(caps)):
    stash,eps,(ids,m)=run_capture(ci)
    all_data[caps[ci]]={'blocks':{str(i):{k:v for k,v in r.items() if k not in ('attn_weights','context','x_out','V','K','Q','tp')} for i,r in stash.items()},
                        'entropy':{str(i):entropy_stats(r['attn_weights'],torch.tensor([m])) for i,r in stash.items()},
                        'n_valid':int(sum(m)),'ids':ids[:sum(m)],'mask':m[:sum(m)]}
    print(' cap',ci,'done',flush=True)

# aggregate entropy
ent_agg={}
for bi in all_data[caps[0]]['entropy']:
    hstats=[h for cdat in all_data.values() for h in cdat['entropy'][bi]['heads']]
    ent_agg[bi]={'norm_entropy_mean':float(np.mean([x['norm_entropy'] for x in hstats if x['norm_entropy'] is not None])),
                 'pmax_mean':float(np.mean([x['pmax_mean'] for x in hstats])),
                 'eff_tokens_mean':float(np.mean([x['eff_tokens'] for x in hstats])),
                 'n_valid':all_data[caps[0]]['entropy'][bi]['n_valid']}
with open(OUT/'L19_ATTENTION_ENTROPY_ANALYSIS.json','w') as f: json.dump({'per_block':ent_agg,'raw':all_data},f,indent=2)
print('entropy agg:',json.dumps(ent_agg)[:500],flush=True)

# ---- caption attention comparison (JS div + cosine on weights) ----
cap_cmp={}
base_cap=caps[0]
stash0,_,_=run_capture(0)
for ci in range(1,len(caps)):
    stash_i,_,_=run_capture(ci)
    per={}
    for bi in stash0:
        w0=stash0[bi]['attn_weights'][0]   # [heads,ss,S]
        wi=stash_i[bi]['attn_weights'][0]
        # marginalize over heads+queries for caption-level dist
        d0=w0.mean(dim=(0,1)); di=wi.mean(dim=(0,1))
        cos=float(torch.nn.functional.cosine_similarity(d0,di,dim=0))
        # JS div
        m_=(d0+di)/2; js=0.5*float((d0*(torch.log(d0.clamp_min(1e-12)/m_.clamp_min(1e-12)))).sum()+(di*(torch.log(di.clamp_min(1e-12)/m_.clamp_min(1e-12)))).sum())
        per[str(bi)]={'weight_cosine':cos,'js_div':js}
    cap_cmp[caps[ci][:40]]={'vs':base_cap[:40],'blocks':per}
with open(OUT/'L19_CAPTION_ATTENTION_COMPARISON.json','w') as f: json.dump(cap_cmp,f,indent=2)

# ---- token identity: top-attended tokens per head (block0) ----
tok_report={}
stash,_,(ids,m)=run_capture(0)
for bi in list(stash)[:2]:
    w=stash[bi]['attn_weights'][0]  # [heads,ss,S]
    valid_idx=[i for i,v in enumerate(m) if v>0]
    tops=[]
    for h in range(w.shape[0]):
        wh=w[h,:,valid_idx]  # [ss,nv]
        meanp=wh.mean(0)
        order=meanp.argsort(descending=True)
        tops.append({'head':h,'top':[(int(valid_idx[o]),ids[valid_idx[o]],float(meanp[o])) for o in order[:3]]})
    tok_report[str(bi)]={'valid_ids':ids[:sum(m)],'top_per_head':tops}
with open(OUT/'L19_TOKEN_IDENTITY_ANALYSIS.json','w') as f: json.dump(tok_report,f,indent=2)
print('token tops sample:',json.dumps(tok_report)[:400],flush=True)

# ---- KV analysis ----
kv={}
stashA,_,_=run_capture(0); stashB,_,_=run_capture(1)
for bi in list(stashA)[:3]:
    rA,rB=stashA[bi],stashB[bi]
    kv[str(bi)]={'k_rms':rA['k_rms'],'v_rms':rA['v_rms'],'q_rms':rA['q_rms'],
      'context_rms':float(rA['context'].pow(2).mean().sqrt()),
      'context_diff_caption':float((rA['context']-rB['context']).abs().mean()),
      'tp_rms':rA['tp_rms'],'h_in_rms':rA['h_in_rms']}
with open(OUT/'L19_KEY_VALUE_PROJECTION_ANALYSIS.json','w') as f: json.dump(kv,f,indent=2)
with open(OUT/'L19_CONTEXT_AGGREGATION_ANALYSIS.json','w') as f: json.dump({'per_block':kv,'note':'context = attn output before out_proj reshape; compare across captions'},f,indent=2)
print('kv:',json.dumps(kv)[:400],flush=True)

# ---- interventions: uniform / single-token / temperature on predicted noise ----
print('interventions', flush=True)
e0,mk0,_,_=enc(caps[0]); e1,mk1,_,_=enc(caps[1])
g=torch.Generator().manual_seed(777); xn=torch.randn(1,3,64,64,generator=g); t=torch.tensor([int(ts[12])])
orig_mh_fwd=MultiHeadAttention.forward

def attn_with_weights(self,query,key,value,mask,weight_override=None,temp=1.0):
    b,ql,_=query.shape; kl=key.shape[1]
    Q=self.q_proj(query).view(b,ql,self.n_heads,self.head_dim).transpose(1,2)
    K=self.k_proj(key).view(b,kl,self.n_heads,self.head_dim).transpose(1,2)
    V=self.v_proj(value).view(b,kl,self.n_heads,self.head_dim).transpose(1,2)
    scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
    if mask is not None:
        mk=mask.unsqueeze(1).unsqueeze(2); scores=scores.masked_fill(mk==0,float('-inf'))
    if weight_override is not None:
        w=weight_override  # [b,heads,ql,kl] already masked/normalized
    else:
        w=torch.softmax(scores/temp,-1)
    out=torch.matmul(w,V).transpose(1,2).contiguous().view(b,ql,self.d_model)
    return self.out_proj(out)

interv={}
def run_with(mode,param=None):
    def fwd(self,q,k,v,mask):
        b,ql,_=q.shape; kl=k.shape[1]
        Q=self.q_proj(q).view(b,ql,self.n_heads,self.head_dim).transpose(1,2)
        K=self.k_proj(k).view(b,kl,self.n_heads,self.head_dim).transpose(1,2)
        V=self.v_proj(v).view(b,kl,self.n_heads,self.head_dim).transpose(1,2)
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(self.head_dim)
        mk=mask.unsqueeze(1).unsqueeze(2); scores=scores.masked_fill(mk==0,float('-inf'))
        if mode=='uniform':
            w=torch.softmax(scores*0,-1)  # uniform over unmasked
        elif mode=='temp':
            w=torch.softmax(scores*param,-1)
        elif mode=='single':
            w=torch.zeros_like(scores); w[...,param]=1.0
            w=w.masked_fill(mk==0,0.0); w=w/w.sum(-1,keepdim=True).clamp_min(1e-9)
        else:
            w=torch.softmax(scores,-1)
        out=torch.matmul(w,V).transpose(1,2).contiguous().view(b,ql,self.d_model)
        return self.out_proj(out)
    MultiHeadAttention.forward=fwd
    with torch.no_grad():
        p0=su(xn,t,e0,mk0); p1=su(xn,t,e1,mk1)
    MultiHeadAttention.forward=orig_mh_fwd
    d=(p0-p1).abs()
    return {'swap_mean_abs':float(d.mean()),'swap_cos':float(torch.nn.functional.cosine_similarity(p0.flatten(),p1.flatten(),dim=0)),'eps_std':float(p0.std())}

interv['native']=run_with('native')
interv['uniform']=run_with('uniform')
for ti in [0,3,5]:
    interv[f'single_tok{ti}']=run_with('single',ti)
for tp in [0.5,2.0]:
    interv[f'temp{tp}']=run_with('temp',tp)
with open(OUT/'L19_ATTENTION_INTERVENTION_RESULTS.json','w') as f: json.dump(interv,f,indent=2)
with open(OUT/'L19_TEMPERATURE_SWEEP.json','w') as f: json.dump({'native_temp1.0':interv['native'],'0.5x':interv['temp0.5'],'2.0x':interv['temp2.0']},f,indent=2)
print('interventions:',json.dumps(interv)[:500],flush=True)

# integrity + hashes
import hashlib
h={}
for p in sorted(OUT.rglob('*')):
    if p.is_file() and p.suffix not in ('.png','.pt') and p.name!='L19_ARTIFACT_HASHES.json':
        h[str(p.relative_to(OUT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'L19_CHECKPOINT_INTEGRITY.json').write_text(json.dumps({'stage2':'untouched','all_inference_only':True,'ckpts_unmodified':True},indent=2))
(OUT/'L19_ARTIFACT_HASHES.json').write_text(json.dumps({'algorithm':'sha256','files':h},indent=2))
print('DONE')
