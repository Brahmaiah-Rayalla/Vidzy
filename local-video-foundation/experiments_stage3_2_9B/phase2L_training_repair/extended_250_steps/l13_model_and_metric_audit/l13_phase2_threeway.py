"""
L.13 Phase 2: three-way spatial quality comparison.
A = Stage2.1 conditional image model (checkpoint_stage2_motion_expanded.pt)
B = C25 ckpt, temporal disabled (frame-independent path), noise repeated across frames
C = C25 ckpt, temporal enabled (native)
Same prompts/tokenizer/seed schedule. B uses A's image noise tiled across 16 frames
so frame-independent outputs should match A exactly if integration is correct.
"""
import json, sys, time, torch, numpy as np
from pathlib import Path
from PIL import Image

L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
from l12_common import *
import l12_common
from models.text_encoder import SimpleTokenizer, TextEncoder

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l13_model_and_metric_audit')
VD = OUT/'three_way'; VD.mkdir(exist_ok=True)
base = l12_common.base_dir

PROMPTS = [
 'A red circle on a white background',
 'A blue square on a white background',
 'A yellow triangle on a white background',
 'A green circle at the center',
 'A red square on a white background',
]

def ddim_image(model, te, tm, noise_img, ac, ts):
    """DDIM for single image [1,3,64,64]; model(x[B,C,H,W]...) -> ConditionalUNet
    For video_unet frame-independent we handle separately."""
    x = noise_img.clone()
    for i in range(len(ts)-1):
        t, tp = ts[i], ts[i+1]
        with torch.no_grad():
            eps = model(x, torch.tensor([int(t)]).long(), te, tm)
        a_t, a_p = ac[t], ac[tp]
        x = torch.sqrt(a_p)*(x - torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t) + torch.sqrt(1-a_p)*eps
    return (x.clamp(-1,1)+1)/2

def ddim_video(model, tok, caption, noise5d, ac, ts):
    te, tm = encode_prompt(model, tok, caption)
    x = noise5d.clone()
    for i in range(len(ts)-1):
        t, tp = ts[i], ts[i+1]
        with torch.no_grad():
            eps = model(x, torch.tensor([int(t)]).long(), te, tm)
        a_t, a_p = ac[t], ac[tp]
        x = torch.sqrt(a_p)*(x - torch.sqrt(1-a_t)*eps)/torch.sqrt(a_t) + torch.sqrt(1-a_p)*eps
    return (x.clamp(-1,1)+1)/2

def save_png(t_img, path):
    a = (t_img[0].permute(1,2,0).numpy()*255).astype(np.uint8)
    Image.fromarray(a).save(path)

def main():
    t0=time.time()
    tok = SimpleTokenizer.load(base/'experiments_stage3_2_7/checkpoints/tokenizer_motion_expanded.json')
    ac, ts = build_diffusion_schedule()
    results = {'prompts': PROMPTS, 'configs': {}}

    # ---------- Config A: Stage2.1 conditional image model ----------
    print('Loading Stage2.1 conditional model...')
    device = torch.device('cpu')
    ck = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_motion_expanded.pt', map_location=device)
    vocab = ck['text_encoder_state_dict']['token_emb.weight'].shape[0]
    suA = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    suA.load_state_dict(ck['model_state_dict']); suA.eval()
    teA = TextEncoder(vocab_size=vocab, d_model=256, n_heads=4, n_layers=4, d_ff=512,
                      max_seq_len=64, dropout=0.0)
    teA.load_state_dict(ck['text_encoder_state_dict']); teA.eval()

    results['configs']['A_stage2_image'] = {}
    for pi, prompt in enumerate(PROMPTS):
        ids, m = tok.encode(prompt, max_length=64)
        ids_t = torch.tensor([ids]); m_t = torch.tensor([m], dtype=torch.float)
        with torch.no_grad():
            emb, mask = teA(ids_t, m_t)
        g = torch.Generator().manual_seed(5000+pi)
        noise = torch.randn(1,3,64,64,generator=g)
        img = ddim_image(suA, emb, mask, noise, ac, ts)
        d = VD/f'A_{pi}'; d.mkdir(exist_ok=True)
        save_png(img, d/'image.png')
        np.save(d/'image.npy', img[0].permute(1,2,0).numpy())
        # simple color metrics
        rgb = img[0].permute(1,2,0).numpy()
        results['configs']['A_stage2_image'][prompt] = {
            'mean_rgb': rgb.reshape(-1,3).mean(0).round(3).tolist(),
            'std': float(rgb.std()), 'min': float(rgb.min()), 'max': float(rgb.max())}
        print(f'  A prompt {pi+1}/5 done', flush=True)

    results['configs']['A2_stage2_spatial_ref'] = {}
    ck2 = torch.load(base/'experiments_stage3_2_7/checkpoints/checkpoint_stage2_spatial.pt', map_location=device)
    vocab2 = ck2['text_encoder_state_dict']['token_emb.weight'].shape[0]
    suA2 = ConditionalUNet(in_channels=3, model_channels=64, channel_mult=[1,2,2,2],
        num_res_blocks=2, attention_resolutions=[], dropout=0.0, num_timesteps=1000,
        text_dim=256, n_heads=4, use_cross_attn=True)
    suA2.load_state_dict(ck2['model_state_dict']); suA2.eval()
    teA2 = TextEncoder(vocab_size=vocab2, d_model=256, n_heads=4, n_layers=4, d_ff=512,
                       max_seq_len=64, dropout=0.0)
    teA2.load_state_dict(ck2['text_encoder_state_dict']); teA2.eval()
    for pi, prompt in enumerate(PROMPTS):
        ids, m = tok.encode(prompt, max_length=64)
        ids_t = torch.tensor([ids]); m_t = torch.tensor([m], dtype=torch.float)
        with torch.no_grad():
            emb, mask = teA2(ids_t, m_t)
        g = torch.Generator().manual_seed(5000+pi)
        noise = torch.randn(1,3,64,64,generator=g)
        img = ddim_image(suA2, emb, mask, noise, ac, ts)
        d = VD/f'A2_{pi}'; d.mkdir(exist_ok=True)
        save_png(img, d/'image.png')
        rgb = img[0].permute(1,2,0).numpy()
        results['configs']['A2_stage2_spatial_ref'][prompt] = {
            'mean_rgb': rgb.reshape(-1,3).mean(0).round(3).tolist(), 'std': float(rgb.std())}
        print(f'  A2 prompt {pi+1}/5 done', flush=True)
    del suA2, teA2, ck2

    # ---------- Config B & C: C25 video model ----------
    print('Loading C25 video model...')
    modelC = load_model(CKPT_C25)
    results['configs']['B_temporal_disabled'] = {}
    results['configs']['C_temporal_enabled'] = {}

    for pi, prompt in enumerate(PROMPTS):
        # A's noise tiled across 16 frames for B; same 5D noise seed for C
        gA = torch.Generator().manual_seed(5000+pi)
        noise_img = torch.randn(1,3,64,64,generator=gA)
        noise5d_B = noise_img.unsqueeze(2).expand(1,3,16,64,64).contiguous()
        gC = torch.Generator().manual_seed(6000+pi)
        noise5d_C = torch.randn(1,3,16,64,64,generator=gC)

        # B: temporal disabled -> frame-independent
        modelB = load_model(CKPT_C25)  # fresh to avoid state issues
        modelB.enable_temporal_processing = False
        vb = ddim_video(modelB, tok, prompt, noise5d_B, ac, ts)
        del modelB
        # C: native temporal
        vc = ddim_video(modelC, tok, prompt, noise5d_C, ac, ts)

        for tag, vid in [('B',vb),('C',vc)]:
            d = VD/f'{tag}_{pi}'; d.mkdir(exist_ok=True)
            v = vid[0].permute(1,2,3,0).numpy()  # F,H,W,3
            for fi in [0,4,8,12,15]:
                Image.fromarray((np.clip(v[fi],0,1)*255).astype(np.uint8)).save(d/f'frame_{fi:02d}.png')
            np.savez_compressed(d/'video.npz', video=v.astype(np.float32))
            rgb = v.reshape(-1,3)
            key = 'B_temporal_disabled' if tag=='B' else 'C_temporal_enabled'
            results['configs'][key][prompt] = {
                'mean_rgb': rgb.mean(0).round(3).tolist(), 'std': float(rgb.std()),
                'frame0_mean_rgb': v[0].reshape(-1,3).mean(0).round(3).tolist(),
                'temporal_activity': float(np.abs(np.diff(v.mean(axis=3),axis=0)).mean())}
        print(f'  B/C prompt {pi+1}/5 done ({time.time()-t0:.0f}s)', flush=True)

    # A vs B image equality check: B frame0 vs A image (same noise, same text path expected)
    eq = {}
    for pi, prompt in enumerate(PROMPTS):
        a = np.load(VD/f'A_{pi}'/'image.npy')
        bvid = np.load(VD/f'B_{pi}'/'video.npz')['video']
        b0 = bvid[0]
        eq[prompt] = {'max_abs_diff': float(np.abs(a-b0).max()), 'mean_abs_diff': float(np.abs(a-b0).mean())}
    results['A_vs_B_frame0_diff'] = eq

    with open(OUT/'L13_THREE_WAY_RAW.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'Done in {(time.time()-t0)/60:.1f} min')

if __name__=='__main__':
    main()
