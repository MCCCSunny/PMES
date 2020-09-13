import time
import numpy as np
from common import logger
from common import flatten_lists
import tensorflow as tf
import pdb

class Runner_distill:
    def __init__(self, envs, agent_eA, agent_eB, agent, args, n_steps=8):
        self.state = self.logs = None
        self.agent_eA, self.agent_eB, self.agent, self.envs, self.n_steps = agent_eA, agent_eB, agent, envs, n_steps
        self.args = args
        self.max_update = args.max_update
        self.maps = args.maps
        self.scale = args.scale

    def run(self, num_updates=1, anneal_settings, train=True):
        # based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
        self.reset()
        if self.args.Student_restore:
            k = int(tf.train.get_checkpoint_state(self.args.paths[-1]).model_checkpoint_path.split('/')[-1].split('-')[-1])
            k = k - self.args.Student_number
        else:
            k = 0
        print ('the step of anneal value start from %d'%k)
        for i in range(k, num_updates):
            # set the anneal value
            p = anneal_value(float(i/num_updates), anneal_settings) # settings is the anneal function
            r = np.random.rand(1)[0]
            if r>p:
                expert_ = False # use the distilation
            else:
                expert_ = True  # use the expert models
            if self.agent.get_global_step() >= self.max_update or i == num_updates - 1:
                print('reached the max updates, global limit ', self.max_update, ', local limit', num_updates)
                if i > 0:
                    self.agent.save()
                break
            if train:
                rollout = self.collect_rollout(expert_)  #values  collect the actions/ values/
                self.agent.train_distill(i, *rollout)


    def forwardState(self):    
        allnextState = []
        for oneState in self.args.allState:
            nextStateValue = self.agent.get_value(oneState)
            allnextState.append(nextStateValue)
        return np.array(allnextState)

    def updateState(self):
        for step in range(self.n_steps):
            action,_ = self.agent.act(self.state)
            self.state, *_ = self.envs.step(action)

    def collect_rollout(self, expert_):
        '''
        collect the data which created by experts
        expert_: use to decide to adopt the ddistill model or expert model,boolean, True or False
        return: flatten_lists(States): list;values len=4,(256,16,16,17)+(256,16,16,7)+(256,16,11)+(256,549)
                flatten_lists(Policys_experts): list;values len=14,(256,549)+(256,256)+(256,256)+...
                rewards: array; values
                dones: array; values
                done_values: array, (16,16)
                last_value: array, (16,)
                flatten_lists(Actions_experts): list, len=14
                Values_experts: array, (16,16)
                Values_distill: array, (16,16)
        '''
        States, Policys_experts = [None] * self.n_steps, [None] * self.n_steps
        rewards, dones, Values_experts, done_values = np.zeros((4, self.n_steps, self.envs.num_envs))
        for step in range(self.n_steps):
            States[step] = self.state
            # expert values
            policy_eA, action_eA, value_eA = self.agent_eA.act_expert(self.state)           # values
            policy_eB, action_eB, value_eB = self.agent_eB.act_expert(self.state)           # values
            # expert actions
            actionAB = list(np.concatenate([np.array(action_eA)[:,:8], np.array(action_eB)[:,-8:]], axis=1)) # value
            # expert policy
            policyAB = []
            for policy_i in range(len(policy_eA)):
                policyAB.append(np.concatenate([policy_eA[policy_i][:8,:], policy_eB[policy_i][-8:,:]], axis=0))
            Policys_experts[step] = policyAB   # value
            # expert value
            valueAB = np.concatenate([value_eA[:8], value_eB[-8:]],axis=0)  # value
            Values_experts[step] = valueAB # value
            # state + reward
            if expert_:
                self.state, rewards[step], dones[step] = self.envs.step(actionAB)
            else:
                 action_distill, _ = self.agent.act(self.state)
                 self.state, rewards[step], dones[step] = self.envs.step(action_distill)
            if sum(dones[step]) > 0:
                done_valuesA = self.agent_eA.get_value(States[step])
                done_valuesB = self.agent_eB.get_value(States[step])
                done_values[step] = np.concatenate([done_valuesA[:8], done_valuesB[-8:]], axis=0)
            last_valueA = self.agent_eA.get_value(self.state)
            last_valueB = self.agent_eB.get_value(self.state)
            last_value = np.concatenate([last_valueA[:8], last_valueB[-8:]], axis=0)
        return flatten_lists(States), flatten_lists(Policys_experts), rewards, dones, done_values, last_value, Values_experts

    def warm_up(self):
        total_steps = self.n_steps * 100
        reset_pts = np.random.choice(total_steps - 1, self.envs.num_envs - 1, replace=False)
        reset_dict = {}
        for i, n in enumerate(reset_pts):
            reset_dict[n] = i

        for step in range(total_steps):
            action, _ = self.agent.act(self.state)
            if step in reset_dict:
                i = reset_dict[step]
                print('reset env ', i, ' at step ', step, ' for warming up, global step ', self.agent.get_global_step())
                self.envs.reset(i)
                for a in action:
                    a[i] = 0

            self.state, *_ = self.envs.step(action)

    @staticmethod
    def parse_rewards(raw_rewards, scale):
        '''
        reward: be used to train the model
        aux_reward: be used to monitor the real score in the shaping experiment
        '''
        rewards = (np.array(raw_rewards) - 500) % 1000 - 500
        aux_rewards = (raw_rewards - rewards) // 1000 #num of marines
        if scale > 0:
            rewards = rewards/float(scale)
        return rewards, aux_rewards

    def reset(self):
        self.state, *_ = self.envs.reset()

    def log(self, rewards, aux_rewards, dones):
        dones = np.array(dones, dtype=int)
        for i, d in enumerate(self.logs['dones']):
            self.logs['ep_rew'][d, i] += rewards[i] #the record is the reward rather than the aux_reward
            self.logs['aux_ep_rew'][d, i] += aux_rewards[i]
            if self.logs['dones'][i] + dones[i] == 2:
                self.logs['ep_rew'][0, i] = self.logs['ep_rew'][1, i]
                self.logs['aux_ep_rew'][0, i] = self.logs['aux_ep_rew'][1, i]
                self.logs['ep_rew'][1, i] = 0
                self.logs['aux_ep_rew'][1, i] = 0

        self.logs['eps'] += sum(dones)
        self.logs['dones'] = np.maximum(self.logs['dones'], dones)
        if sum(self.logs['dones']) < self.envs.num_envs:
            return

        left = right = 0
        for key, value in self.maps.items():
            right += value
            self.logs[key + '_ep_rew'] = self.logs['ep_rew'][0][left:right]
            left = right

        self.logs['ep_rews'] = np.mean(self.logs['ep_rew'][0])
        self.logs['aux_ep_rews'] = np.mean(self.logs['aux_ep_rew'][0])
        self.logs['rew_best'] = max(self.logs['rew_best'], self.logs['ep_rews'])
        self.logs['aux_rew_best'] = max(self.logs['aux_rew_best'], self.logs['aux_ep_rews'])

        for key in self.maps:
            hasdata = len(self.logs[key + '_ep_rew']) > 0
            self.logs[key + '_ep_rews'] = np.mean(self.logs[key + '_ep_rew']) if hasdata else 0
            self.logs[key + '_rew_best'] = max(self.logs[key + '_rew_best'],
                                               self.logs[key + '_ep_rews']) if hasdata else 0

        if self.logs[list(self.maps)[0] + '_ep_rews'] > self.agent.next_best:
            self.agent.save_best()
            self.agent.next_best = self.logs[list(self.maps)[0] + '_ep_rews'] + self.agent.best_interval
            print('best snapshot saved')

        elapsed_time = time.time() - self.logs['start_time']
        frames = self.envs.num_envs * self.n_steps * self.logs['updates']

        logger.logkv('fps', int(frames / elapsed_time))
        logger.logkv('elapsed_time', int(elapsed_time))
        logger.logkv('n_eps', self.logs['eps'])
        logger.logkv('n_samples', frames)
        logger.logkv('n_updates', self.logs['updates'])
        logger.logkv('global_step', self.agent.get_global_step())
        logger.logkv('lr', self.agent.get_lr())
        logger.logkv('aux_ep_rew_best', self.logs['aux_rew_best'])
        logger.logkv('aux_ep_rew_max', np.max(self.logs['aux_ep_rew'][0]))
        logger.logkv('aux_ep_rew_mean', self.logs['aux_ep_rews'])
        i = 0
        for key in self.maps:
            pre = str(i) + '_' + key
            best_key = key + '_rew_best'
            ep_key = key + '_ep_rew'
            hasdata = len(self.logs[ep_key]) > 0
            logger.logkv(pre + '_rew_best', self.logs[best_key] if hasdata else '-')
            logger.logkv(pre + '_rew_max', np.max(self.logs[ep_key]) if hasdata else '-')
            logger.logkv(pre + '_rew_mean', np.mean(self.logs[ep_key]) if hasdata else '-')
            logger.logkv(pre + '_rew_std', np.std(self.logs[ep_key]) if hasdata else '-')
            logger.logkv(pre + '_rew_min', np.min(self.logs[ep_key]) if hasdata else '-')
            i += 1
        logger.dumpkvs()

        self.logs['dones'] = np.zeros(self.envs.num_envs, dtype=int)
        self.logs['ep_rew'][0] = self.logs['ep_rew'][1]
        self.logs['ep_rew'][1] = np.zeros(self.envs.num_envs)
        self.logs['aux_ep_rew'][0] = self.logs['aux_ep_rew'][1]
        self.logs['aux_ep_rew'][1] = np.zeros(self.envs.num_envs)

def anneal_value(d, settings_):
    """
        d is a value between 0 and 1.0 that is an indicator to
        how close something is to the end or from the begining
        0 being the start and 1.0 being the end.
        settings_: dict
        refer to the PLAID code： https://github.com/FracturedPlane/RL-Framework
    """
    d = float(d)
    anneal_type = settings_['annealing_schedule']
    if (anneal_type == 'linear'):
        p = 1.0 - (d)
    elif (anneal_type == "log"):
        p = (1.0 - (math.log((d)+1.0)))**settings_['initial_temperature']
    elif (anneal_type == "square"):
        d = 1.0 - (d)
        p = (d**2)
    elif (anneal_type == "exp"):
        d = 1.0 - (d)
        p = (d**round_)

    return p

