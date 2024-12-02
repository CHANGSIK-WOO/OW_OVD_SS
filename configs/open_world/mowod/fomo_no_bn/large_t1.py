_base_ = ('./large_base.py')

prev_intro_cls = 0
cur_intro_cls = 20
train_json = 'MOWOD/t1_train.json'
setting_path = 'data/VOC2007/MOWOD/t1_setting.pth'

# fomo pipline
pipline = [dict(type='att_select', epoch=5, train_par=['att_linear'], mean_embedding=True),
           dict(type='att_adapt', epoch=3, train_par=['att_embedding']),
           dict(type='att_refinement', epoch=5, train_par=['att_embedding'])]





