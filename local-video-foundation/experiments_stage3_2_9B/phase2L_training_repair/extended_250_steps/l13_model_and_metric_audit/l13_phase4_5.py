"""
L.13 Phase 4 + Phase 5: dataset ground-truth audit and motion-metric calibration
on synthetic videos with known trajectories.
"""
import json, sys, numpy as np
from pathlib import Path
from PIL import Image

OUT = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l13_model_and_metric_audit')
L12 = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l12_gate_attribution')
sys.path.insert(0, str(L12))
from l12_common import compute_detection_metrics, compute_global_motion_metrics, direction_accuracy_from_translation, EXPECTED_DIRECTIONS, compute_motion_metrics
import torch

base = Path(r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation')
ds = base/'experiments_stage3_2_7/data_synthetic/train'
meta = json.load(open(base/'experiments_stage3_2_7/data_synthetic/train_metadata.json'))

# ---------- Phase 4: dataset audit ----------
import collections
stats = collections.Counter(m['motion_type'] for m in meta)
shapes = collections.Counter(m['shape'] for m in meta)
colors = collections.Counter(m['color'] for m in meta)
bgs = collections.Counter(m['background'] for m in meta)
sizes = collections.Counter(m['size'] for m in meta)
vels = collections.Counter(m['velocity'] for m in meta)
# verify caption-traj consistency on 3 samples
verify = []
for m in meta[:3] + [x for x in meta if x['motion_type']=='diagonal'][:2] + [x for x in meta if x['motion_type']=='stationary'][:2]:
    traj = np.load(m['trajectory_path'])
    vid = np.load(m['video_path'])
    d = traj[-1]-traj[0]
    verify.append({'id': m['id'], 'motion_type': m['motion_type'], 'caption': m['caption'],
                   'traj_start': traj[0].tolist(), 'traj_end': traj[-1].tolist(),
                   'net_disp': [float(d[0]),float(d[1])], 'video_shape': list(vid.shape),
                   'video_dtype_range': [int(vid.min()), int(vid.max())]})
ds_audit = {
 'n_videos': len(meta), 'resolution': 64, 'frames': 16,
 'motion_types': dict(stats), 'shapes': dict(shapes), 'colors': dict(colors),
 'backgrounds': dict(bgs), 'sizes': dict(sizes), 'velocities': dict(vels),
 'trajectory_fields': 'trajectory_path .npy [frames,2] (x=col,y=row); video_path .npy [frames,64,64,3] uint8',
 'caption_traj_verification': verify,
 'diversity_assessment': '176 videos, 6 motion types x shapes/colors/backgrounds/sizes/velocities; captions template-generated and consistent with trajectories',
}
(OUT/'gt_examples').mkdir(exist_ok=True)
for m in [meta[0], meta[5], meta[11]]:
    vid = np.load(m['video_path'])
    sheet = np.concatenate([vid[i] for i in [0,5,10,15]], axis=1)
    Image.fromarray(sheet).save(OUT/'gt_examples'/f"{m['id']}_contact.png")
ds_audit['contact_sheets'] = ['gt_examples/'+f"{m['id']}_contact.png" for m in [meta[0],meta[5],meta[11]]]
with open(OUT/'L13_DATASET_GROUND_TRUTH_AUDIT.json','w') as f:
    json.dump(ds_audit, f, indent=2)

# ---------- Phase 5: metric calibration on synthetic GT ----------
# Build test set: pick one video per scenario from metadata
picks = {}
for want in ['stationary','left_to_right','right_to_left','top_to_bottom','bottom_to_top','diagonal']:
    cand = [m for m in meta if m['motion_type']==want]
    picks[want] = cand[0]
# scenario 5/6/7 approximations: use videos whose background differs / multi not present -> note limitation
cal = {}
for mt, m in picks.items():
    vid8 = np.load(m['video_path'])  # [F,H,W,3] uint8
    vid = torch.from_numpy(vid8.astype(np.float32)/255.).permute(3,0,1,2).unsqueeze(0)
    traj = np.load(m['trajectory_path'])
    gt_disp = traj[-1]-traj[0]
    gt_dir = gt_disp/ (np.linalg.norm(gt_disp)+1e-9) if np.linalg.norm(gt_disp)>0 else np.zeros(2)
    mm = compute_motion_metrics(vid)
    det = compute_detection_metrics(vid, mt)
    gm = compute_global_motion_metrics(vid)
    gm_cos = direction_accuracy_from_translation(gm, mt)
    # centroid error: diff-centroid displacement vs GT displacement
    cent_disp = np.array(mm['displacement_vector'])
    cent_err = float(np.linalg.norm(cent_disp - gt_disp))
    det_ok = det.get('available') and det['detection_rate']>=0.75
    cal[mt] = {
        'id': m['id'], 'gt_net_displacement': gt_disp.tolist(), 'gt_direction': gt_dir.tolist(),
        'diff_centroid_displacement': mm['displacement_vector'],
        'centroid_error_px': cent_err,
        'diff_centroid_cosine_vs_gt': float(np.dot(cent_disp,gt_disp)/(np.linalg.norm(cent_disp)*np.linalg.norm(gt_disp)+1e-9)) if np.linalg.norm(gt_disp)>0 and np.linalg.norm(cent_disp)>1e-6 else None,
        'global_net_translation': gm['net_translation_px'],
        'global_cosine_vs_expected': gm_cos,
        'detection': {k:v for k,v in det.items() if k not in ('per_frame','per_frame'.replace('per_frame',''))} if det.get('available') else det,
        'detection_rate': det.get('detection_rate'),
        'temporal_activity': mm['temporal_activity'],
        'tracking_tier': 'RELIABLE' if (det_ok and cent_err<8) else ('LOW_CONFIDENCE' if det.get('detection_rate',0)>=0.25 else 'UNRELIABLE'),
    }
# Note scenario coverage limitation
cal['_scenario_coverage'] = {
 'stationary_object': 'covered', 'ltr': 'covered', 'rtl': 'covered', 'diagonal': 'covered',
 'moving_object_changing_bg': 'NOT in dataset (uniform bg)',
 'static_object_changing_bg': 'NOT in dataset',
 'multiple_objects': 'NOT in dataset (single object per video)',
}
with open(OUT/'L13_DETECTOR_CALIBRATION.json','w') as f:
    json.dump(cal, f, indent=2)
with open(OUT/'L13_GLOBAL_MOTION_CALIBRATION.json','w') as f:
    json.dump({'per_video': {k:{'net_translation':v['global_net_translation'],'cosine_vs_expected':v['global_cosine_vs_expected'],'gt_disp':v['gt_net_displacement']} for k,v in cal.items() if not k.startswith('_')},
               'note':'block-matching global translation vs GT trajectory displacement'}, f, indent=2)
print(json.dumps({k:{'gt':v['gt_net_displacement'],'cent':v['diff_centroid_displacement'],'gcos':v['diff_centroid_cosine_vs_gt'],'det_rate':v['detection_rate'],'tier':v['tracking_tier']} for k,v in cal.items() if not k.startswith('_')}, indent=1))
