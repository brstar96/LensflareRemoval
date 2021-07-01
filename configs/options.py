import yaml, os, sys, shutil, torch
import os.path as osp
sys.path.append('../')
from utils import logger

def parse(opt):
    yaml_path = opt.yaml_path

    with open(yaml_path, 'r') as fp:
        args = yaml.full_load(fp.read())
    
    name = args['modelname']
    resume = args['trainer']['resume']

    lg = logger(name, 'log/{}.log'.format(name), resume)

    # general settings
    args['name'] = name

    # create experiment root
    root = osp.join(args['paths']['experiment_root'], name)
    args['paths']['root'] = root
    args['paths']['trained_models'] = osp.join(root, 'best_models')
    args['paths']['visualizations'] = osp.join(root, 'visualization')

    if osp.exists(root) and resume==False:
        lg.info('Remove existing dir: [{}]'.format(root))
        shutil.rmtree(root, True)
    for name, path in args['paths'].items(): 
        if name == 'state':
            continue
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path))

    lg.info('Batch size: {}, Patch size:{}, flip/rot: {}/{}, lr: {}'.format(
                                                    args['datasets']['train']['batch_size'], 
                                                    args['datasets']['train']['patch_size'], 
                                                    args['datasets']['train']['flip'], 
                                                    args['datasets']['train']['rot'], 
                                                    args['trainer']['lr']))
    
    return dict_to_nonedict(args), lg

class NoneDict(dict):
    def __missing__(self, key):
        return None

def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        for k,v in opt.items():
            opt[k] = dict_to_nonedict(v)
        return NoneDict(**opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(x) for x in opt]
    else:
        return opt
