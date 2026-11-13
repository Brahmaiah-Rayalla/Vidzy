p = r'C:\Brahma\AI\AIDirector\Vidzy\local-video-foundation\experiments_stage3_2_9B\phase2L_training_repair\extended_250_steps\l28_shape_color_conditioning\l28_full.py'
s = open(p, encoding='utf-8').read()
s = s.replace("su,sd27=build()", "su,sd27=build()\n_rck=torch.load(OUT/'ckpt_1000.pt',map_location='cpu')\nsu.load_state_dict(_rck['su'],strict=False)")
s = s.replace("opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)  # warm-start new opt (shape changed)",
              "opt=optim.AdamW(params,lr=1e-4,weight_decay=0.01)\nopt.load_state_dict(_rck['opt'])")
s = s.replace("for step in range(N):", "START=1000\nfor step in range(START,N):")
open(p.replace('l28_full.py','l28_resume.py'),'w',encoding='utf-8').write(s)
print('written')
