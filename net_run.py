# -*- coding: utf-8 -*-
# from __future__ import print_function, division
import sys
from util.parse_config import parse_config
from net_run.agent_cls import ClassificationAgent
from agent_seg import SegmentationAgent

def main():
    # if(len(sys.argv) < 3):
    #     print('Number of arguments should be 3. e.g.')
    #     print('   pymic_net_run train config.cfg')
    #     exit()
    # stage    = str(sys.argv[1])
   
        stage    = 'test'
        #one/much
        
        # cfg_file = str(sys.argv[2])
        cfg_file='./dataset/config/train_test.cfg'
        config   = parse_config(cfg_file)
      #   config['training']['learning_rate']=config['training']['learning_rate']
      #   config['training']['weight_decay']=config['training']['weight_decay']
        task     = config['dataset']['task_type']
        assert task in ['cls', 'cls_nexcl', 'seg']
      #   for i in range(45):
        if True:
               # for j in range(2):
               if True:    
                  if(task == 'cls' or task == 'cls_nexcl'):  
                     agent = ClassificationAgent(config, stage)
                  else:
                     agent = SegmentationAgent(config, stage) 
                  agent.run()
                  agent=0
               config['training']['learning_rate']=config['training']['learning_rate']-0.0001
                  #  config['training']['iter_save']=config['training']['iter_save']+50
               print(str(config['training']['learning_rate']))
             
        

if __name__ == "__main__":
    main()
    