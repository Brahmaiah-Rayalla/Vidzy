import numpy as np, json
from pathlib import Path
from PIL import Image
DATA=Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l31_high_salience_shape_conditioning\data')
manifest=[json.loads(l) for l in open(DATA/'manifest.jsonl')]
val=[m for m in manifest if m['file'].startswith('val')]
def radial_shape(arr,ret_score=False):
    bg=np.array([245,245,245])/255.; mask=np.abs(arr-bg).max(2)>0.15
    ys,xs=np.where(mask)
    if len(xs)<20: return (-1,0) if ret_score else -1
    cy,cx=ys.mean(),xs.mean()
    angs=np.linspace(0,2*np.pi,72,endpoint=False); rads=[]
    for a in angs:
        dx,dy=np.cos(a),np.sin(a); best=0
        for rr in range(2,32):
            x=int(round(cx+dx*rr)); y=int(round(cy+dy*rr))
            if 0<=x<64 and 0<=y<64 and mask[y,x]: best=rr
            elif best>0 and rr>best+2: break
        rads.append(best)
    rads=np.array(rads,float)
    if rads.mean()<3: return (-1,0) if ret_score else -1
    k3=np.abs(np.sum(rads*np.exp(-3j*angs)))/np.sum(rads)
    k0var=rads.std()/rads.mean()
    tri_score=k3*0.6+min(1.0,k0var/0.4)*0.4
    pred=1 if tri_score>0.42 else 0
    return (pred,tri_score,k3,k0var) if ret_score else pred
for cid in range(6):
    scores=[]
    for m in [x for x in val if x['cid']==cid][:10]:
        arr=np.array(Image.open(DATA/(m['file']+'.png'))).astype(np.float32)/255.
        scores.append(radial_shape(arr,True))
    print('cid',cid,'shape',cid%2,'scores',[round(s[1],3) for s in scores],'k3',[round(s[2],3) for s in scores],'k0var',[round(s[3],3) for s in scores])
