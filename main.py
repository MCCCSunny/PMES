import os,sys
from collections import OrderedDict
import argparse
import json
import random
from absl import flags
import numpy as np
import tensorflow as tf

from common import Config
from common.env import make_envs
from common import killers_map

from rl.agent import A2CAgent, A2CAgent_expert, A2CAgent_distill
from rl.model import fully_conv
from rl import Runner, Runner_distill, EnvWrapper

import time

flags.FLAGS(['main.py'])

if __name__ == '__main__':
    # yapf: disable
    flags.FLAGS(['main.py'])
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--sz', type=int, default=16, help='screen size and minimap size')
    parser.add_argument('--envs', type=int, default=16, help='the number of parallel environments')
    parser.add_argument('--map', type=str, default='BuildMarines')
    parser.add_argument('--maps', type=str, default='', help='a json that specifies maps and their count. will ignore `map` and `envs` once set')
    parser.add_argument('--render', type=int, default=0)
    parser.add_argument('--steps', type=int, default=16, help='n_step, how many steps for one update')
    parser.add_argument('--updates', type=int, default=1000000, help='how many new updates to run')
    parser.add_argument('--max_update', type=int, default=1000000, help='max number of global updates')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--lr_decay', type=bool, default=False, help='whether or not the lr regularize')
    parser.add_argument('--lr_decay_step', type=float, default=1000000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.96, help='0.12 every 50 decay')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='coef for value loss')
    parser.add_argument('--ent_coef', type=float, default=1e-3, help='entropy coef')
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--clip_grads', type=float, default=1.)
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--run_id', type=str, default='-1')
    parser.add_argument('--save_interval', type=int, default=500)
    parser.add_argument('--save_best_start', type=int, default=10)
    parser.add_argument('--save_best_inc', type=int, default=1)
    parser.add_argument('--cfg_path', type=str, default='config.json.dist')
    parser.add_argument('--test', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--restore', type=bool, nargs='?', const=True, default=False, help='restore from default path')
    parser.add_argument('--remote_restore', type=str, default='', help='restore from another path. note set `restore` will invalidade this')
    parser.add_argument('--warmup', type=bool, nargs='?', const=True, default=False, help='warm up a little bit')
    parser.add_argument('--save_replay', type=bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_snapshot', type=int, default=5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--decay', type=float, default=0.9)
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--step_mul', type=int, default=8, help='show many game step between two actions')
    parser.add_argument('--seed', default=None)
    parser.add_argument('--StatePath', type=str, default=None)
    parser.add_argument('--ckptPath', type=str, default=None, help="the path of the trained model")
    parser.add_argument('--nextStatePath', type=str, help='the path to save the next state')
    parser.add_argument('--scale', type=int, default=0, help='use to scale the reward in the shaping algorithm, 111ARPshaping=1, 124shaping=4, 139shaping=9')
    parser.add_argument('--distill', type=bool, default=False, help='whether or not run the distillation')
    parser.add_argument('--expertPath', type=str, default=None, help='the paths where the experts model are saved')
    parser.add_argument('--anneal_settings', type=dict, default={'annealing_schedule':'log', 'initial_temperature': 2.0}, help='the anneal setting of the prob')
    parser.add_argument('--Student_restore', type=bool, default=False, help='if the distill process is stopped, restore from the checkpoint')
    parser.add_argument('--Student_number', type=int, default=None, help='restart from the number **')
    parser.add_argument('--policy_coef', type=float, default=0.0005, help ='when the distill model is adopted.')
    
    args = parser.parse_args()

    # seeds
    if args.seed is not None:
        seed = int(args.seed)
        random.seed(seed)
        np_seed = random.randint(0, 1e5)
        np.random.seed(np_seed)
        tf_seed = random.randint(0, 1e5)
        tf.set_random_seed(tf_seed)
        print('random seed ', seed, 'np seed ', np_seed, 'tf seed ', tf_seed)

    # preprocess args
    if args.maps == '':
        args.maps = '{"' + args.map + '": ' + str(args.envs) + '}'
        
    args.maps = json.loads(args.maps, object_pairs_hook=OrderedDict)
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Allow gpu growth
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    config = Config(args.sz, '_'.join(args.maps), args.run_id)
    weight_dir = os.path.join(args.work_dir, 'weights', config.full_id())
    log_dir = os.path.join(args.work_dir, 'logs', config.full_id())
    print('weights are saved at ', weight_dir)
    os.makedirs(weight_dir, exist_ok=True)
    cfg_path = os.path.join(weight_dir, 'config.json')

    if args.restore and (not os.path.isfile(os.path.join(weight_dir, 'checkpoint'))) and args.remote_restore != '':
        args.restore_path = args.remote_restore
        config.build(os.path.join(args.restore_path, 'config.json'))
        config.save(cfg_path)
        assert os.path.isfile(os.path.join(args.restore_path, 'checkpoint'))
    elif args.restore:
        if os.path.isfile(cfg_path):
            config.build(cfg_path)
        else:
            config.build(args.cfg_path)
        args.restore_path = weight_dir
        assert os.path.isfile(os.path.join(args.restore_path, 'checkpoint'))
    elif args.remote_restore != '':
        args.restore_path = args.remote_restore
        config.build(os.path.join(args.restore_path, 'config.json'))
        assert os.path.isfile(os.path.join(args.restore_path, 'checkpoint'))
    else:
        args.restore_path = None
        config.build(args.cfg_path)

    if not args.restore and not args.test:
        config.save(cfg_path)

    # The following part is used to calculate the change of the state. 
    # When you train the agent, you can set StatePath is None, this part will be ingnored
    if args.StatePath:
        stateList = os.listdir(args.StatePath)
        state0List = []
        for one in stateList:
            if one.split('_')[0]=='stepvalueArray0':
                state0List.append(one)
        allState = []
        for one0 in state0List:
            value0 = np.load(args.StatePath+one0).reshape(-1, 16, 16, 16, 17)
            num_ = one0.split('_')[1]
            value1 = np.load(args.StatePath+'stepvalueArray1_'+str(num_)).reshape(-1, 16, 16, 16, 7)
            value2 = np.load(args.StatePath+'stepvalueArray2_'+str(num_)).reshape(-1, 16, 11)
            value3 = np.load(args.StatePath+'stepvalueArray3_'+str(num_)).reshape(-1, 16, 549)
            for i in range(value0.shape[0]):
                oneStateValue = [value0[i], value1[i], value2[i], value3[i]]
                allState.append(oneStateValue)

        args.allState = allState
        #envs = EnvWrapper(make_envs(args), config)

        weightsList = os.listdir(args.ckptPath)
        weightsList.remove('config.json')
        weightsList.remove('checkpoint')
        weightsList = [oneWeight.split('.')[0] for oneWeight in weightsList]
        weightsList = list(set(weightsList))
        npyList = os.listdir(args.nextStatePath)
        npyList1 = [oneNpy.split('_')[1][:-4] for oneNpy in npyList]
        npyList2 = list(set(npyList1))
        for oneCkpt in weightsList:
            tf.reset_default_graph()
            if oneCkpt.split('.')[0][4:] in npyList2:
                continue
            try:
                sess = tf.Session(config=tf_config)
                args.ckptfile = oneCkpt
                agent = A2CAgent(sess, fully_conv, config, args.discount, args.lr, args.vf_coef, args.ent_coef, args.clip_grads,
                         weight_dir, log_dir, args)
                runner = Runner(None, agent, args, args.steps)
                NextStateValue = runner.forwardState()
                np.save(args.nextStatePath+'nextState_'+str(oneCkpt.split('-')[1]) + '.npy', NextStateValue)
            except:
            	pass

        time.sleep(10)
        os._exit(0)  
    else:
        args.ckptfile = None

    # begin to train the agent 
    if not args.distill:
        try:
            sess = tf.Session(config=tf_config)
            envs = EnvWrapper(make_envs(args), config)
            agent = A2CAgent(sess, fully_conv, config, args.discount, args.lr, args.vf_coef, args.ent_coef, args.clip_grads,
                             weight_dir, log_dir, args)
            runner = Runner(envs, agent, args, args.steps)

        except Exception as e:  
            print('--exit--')
            print(e)
            if envs is not None:
                envs.close()
            import sys
            sys.exit(-1)

        runner.run(args.updates, not args.test)

        if args.save_replay:
            envs.save_replay(replay_dir=('PySC2Replays' if args.run_id == -1 else args.run_id))

        envs.close()
        
    elif args.distill:
        if args.expertPath:
            paths = args.expertPath.split('?')
            args.paths = paths
        else:
            print ('Please input the path of expert models')
            import sys
            sys.exit(-1)

        agent_eA = A2CAgent_expert(None, fully_conv, config, args, paths[0])
        agent_eB = A2CAgent_expert(None, fully_conv, config, args, paths[1])
        agent = A2CAgent_distill(None, agent_eA, agent_eB, fully_conv, config, weight_dir, log_dir, args)#, 'distill')

        envs = EnvWrapper(make_envs(args), config)
        runner_distill = Runner_distill(envs, agent_eA, agent_eB, agent, args, args.steps)
        runner_distill.run(args.updates, args.anneal_settings, not args.test)

        if args.save_replay:
            envs.save_replay(replay_dir=('PySC2Replays' if args.run_id == -1 else args.run_id))

        envs.close()
       
