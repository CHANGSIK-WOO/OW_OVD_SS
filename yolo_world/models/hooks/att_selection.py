import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import MMDistributedDataParallel


@HOOKS.register_module()
class WorkPiplineHook(Hook):
    def before_train_epoch(self, runner):
        if isinstance(runner.model, MMDistributedDataParallel):
            pipline = runner.model.module.pipline
        else:
            pipline = runner.model.pipline
            
        if 'train_par' in pipline[0]:
            for name in pipline[0]['train_par']:
                for name, param in (runner.model.module.named_parameters() 
                                    if isinstance(runner.model, MMDistributedDataParallel) else runner.model.named_parameters()):
                    if name not in name:
                        param.requires_grad = False
                    if name in name:
                        param.requires_grad = True
            # reset optimizer
            runner.optim_wrapper = runner.build_optim_wrapper(runner.cfg.get('optim_wrapper'))
            
            if isinstance(runner.model, MMDistributedDataParallel):
                runner.model.module.pipline[0].pop('train_par')
            else:
                runner.model.pipline[0].pop('train_par')
    
    def after_train_epoch(self, runner):
        if isinstance(runner.model, MMDistributedDataParallel):
            runner.model.module.pipline[0]['epoch'] -= 1
            if 'mean_embedding' in runner.model.module.pipline[0]:
                runner.model.module.pipline[0]['mean_embedding'] = False
                # just collect one epoch
                runner.model.module.bbox_head.set_collect(False)
                runner.model.module.prepare_adapt()
                
            if runner.model.module.pipline[0]['epoch'] > 0:
                return
            if 'att_select' == runner.model.module.pipline[0]['type']:
                runner.model.module.select_att()
            if 'att_refinement' == runner.model.module.pipline[0]['type']:
                runner.model.module.get_obj()
            
            runner.model.module.pipline.pop(0)    
        else:
        # runner.model
            runner.model.pipline[0]['epoch'] -= 1
            
            if 'mean_embedding' in runner.model.pipline[0]:
                runner.model.pipline[0]['mean_embedding'] = False
                # just collect one epoch
                runner.model.bbox_head.set_collect(False)
                runner.model.prepare_adapt()
            
            if runner.model.pipline[0]['epoch'] > 0:
                return 
                
            if 'att_select' == runner.model.pipline[0]['type']:
                runner.model.select_att()
            if 'att_refinement' == runner.model.pipline[0]['type']:
                runner.model.get_obj()
            runner.model.pipline.pop(0)
            
    def before_test_epoch(self, runner):   
        # check shape
        check_point = runner._load_from
        model_pars = torch.load(check_point)['state_dict']
        check_point_att = model_pars['att_embedding']
        check_point_linear = model_pars['bbox_head.att_linear.weight']
        valid = model_pars['bbox_head.valid']
        # check shape
        if isinstance(runner.model, MMDistributedDataParallel):
            att_num = runner.model.module.att_embedding.shape[0]
            
            if att_num != check_point_att.shape[0]:
                runner.model.module.att_embedding = torch.nn.Parameter(check_point_att.to(runner.model.module.att_embedding))
            if att_num != check_point_linear.shape[1]:
                runner.model.module.bbox_head.set_linear({'weight': check_point_linear.to(runner.model.module.bbox_head.att_linear.weight)})
                runner.model.module.bbox_head.valid = torch.nn.Parameter(valid.to(runner.model.module.bbox_head.valid))
                print('update att_embedding and att_linear')  
        else:
            att_num = runner.model.att_embedding.shape[0]
            if att_num != check_point_att.shape[0]:
                runner.model.att_embedding = torch.nn.Parameter(check_point_att.to(runner.model.att_embedding))
            if att_num != check_point_linear.shape[1]:
                runner.model.bbox_head.set_linear({'weight': check_point_linear.to(runner.model.bbox_head.att_linear.weight)})
                runner.model.bbox_head.set_vaild(torch.nn.Parameter(valid.to(runner.model.bbox_head.valid)))
                print('update att_embedding and att_linear')    
        
    
            
