import torch
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.model import MMDistributedDataParallel


@HOOKS.register_module()
class OurWorkPiplineHook(Hook):
    def before_train_epoch(self, runner):
        if isinstance(runner.model, MMDistributedDataParallel):
            if len(runner.model.module.pipline) == 0:
                return
            
            if 'att_select' == runner.model.module.pipline[0]['type']:
                runner.model.module.pipline[0]['log_start_epoch'] -= 1
                if runner.model.module.pipline[0]['log_start_epoch'] > 0:
                    runner.model.module.bbox_head.disable_log()
                elif runner.model.module.pipline[0]['log_start_epoch'] == 0:
                    runner.model.module.bbox_head.enable_log()
                else:
                    runner.model.module.bbox_head.disable_log()
        else:
            if len(runner.model.pipline) == 0:
                return
            
            if 'att_select' == runner.model.pipline[0]['type']:
                runner.model.pipline[0]['log_start_epoch'] -= 1
                if runner.model.pipline[0]['log_start_epoch'] > 0:
                    runner.model.bbox_head.disable_log()
                elif runner.model.pipline[0]['log_start_epoch'] == 0:
                    runner.model.bbox_head.enable_log()
                else:
                    runner.model.bbox_head.disable_log()
                    
    def after_train_epoch(self, runner):
        if isinstance(runner.model, MMDistributedDataParallel):
            if len(runner.model.module.pipline) == 0:
                return
            if  runner.model.module.pipline[0]['type'] == 'att_select': 
                if runner.model.module.pipline[0]['log_start_epoch'] == 0:
                    runner.model.module.bbox_head.select_att()
                    runner.model.module.bbox_head.disable_log()
                    runner.model.module.pipline.pop(0)
        else:
            if len(runner.model.pipline) == 0:
                return
            if runner.model.pipline[0]['type'] == 'att_select':
                if runner.model.pipline[0]['log_start_epoch'] == 0:
                    runner.model.bbox_head.select_att()
                    runner.model.bbox_head.disable_log()
                    runner.model.pipline.pop(0)
                    
            
        
    
            
