import yaml, os, logging, sys, shutil
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
        lg.info('Remove dir: [{}]'.format(root))
        shutil.rmtree(root, True)
    for name, path in args['paths'].items(): 
        if name == 'state':
            continue
        if not osp.exists(path):
            os.mkdir(path)
            lg.info('Create directory: {}'.format(path)) 

    # GPU environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['networks']['gpu_ids'])
    lg.info('Available gpu: {}'.format(args['networks']['gpu_ids']))
    
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
