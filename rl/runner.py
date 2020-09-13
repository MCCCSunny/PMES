import time
import numpy as np
from common import logger
from common import flatten_lists
import tensorflow as tf
import pdb

class Runner:
    def __init__(self, envs, agent, args, n_steps=8):
        self.state = self.logs = None
        self.agent, self.envs, self.n_steps = agent, envs, n_steps
        self.args = args
        self.max_update = args.max_update
        self.maps = args.maps
        self.scale = args.scale

    def run(self, num_updates=1, train=True):
        # based on https://github.com/deepmind/pysc2/blob/master/pysc2/env/run_loop.py
        self.reset()
        try:
            if self.args.warmup:
                print('warming up, pls wait...')
                self.warm_up()
                print('warm up done.')
            
            for i in range(num_updates):
                if self.agent.get_global_step() >= self.max_update or i == num_updates - 1:
                    print('reached the max updates, global limit ', self.max_update, ', local limit', num_updates)
                    if i > 0:
                        self.agent.save()
                    break

                self.logs['updates'] += 1
                rollout = self.collect_rollout()  #values
                if train:
                    log_dict = {k: v for k, v in self.logs.items() if k.endswith('_ep_rews')}
                    log_dict['rewards'] = self.logs['ep_rews']
                    self.agent.train(i, *rollout, log_dict)
                
        except KeyboardInterrupt:
            pass
        finally:
            elapsed_time = time.time() - self.logs['start_time']
            frames = self.envs.num_envs * self.n_steps * self.logs['updates']
            print("Took %.3f seconds for %s steps: %.3f fps" % (elapsed_time, frames, frames / elapsed_time))

    def forwardState(self):
        allnextState = []
        for oneState in self.args.allState:
            nextStateValue = self.agent.get_value(oneState)
            allnextState.append(nextStateValue)
        return np.array(allnextState)

    def collect_rollout(self):
        states, actions = [None] * self.n_steps, [None] * self.n_steps
        rewards, dones, values, done_values = np.zeros((4, self.n_steps, self.envs.num_envs))

        for step in range(self.n_steps):
            action, values[step] = self.agent.act(self.state)
            states[step], actions[step] = self.state, action
            self.state, raw_rewards, dones[step] = self.envs.step(action) # self.state:list, len=4,
            rewards[step], aux_rewards = self.parse_rewards(raw_rewards, self.scale)
            if sum(dones[step]) > 0:
                done_values[step] = self.agent.get_value(states[step])            
            self.log(rewards[step], aux_rewards, dones[step])

        last_value = self.agent.get_value(self.state)
        return flatten_lists(states), flatten_lists(actions), rewards, dones, done_values, last_value

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
        self.logs = {
            'updates': 0,
            'eps': 0,
            'start_time': time.time(),
            'rew_best': 0,
            'ep_rews': 0,
            'ep_rew': np.zeros((2, self.envs.num_envs)),
            'aux_rew_best': 0,
            'aux_ep_rew': np.zeros((2, self.envs.num_envs)),
            'dones': np.zeros(self.envs.num_envs, dtype=int),
        }
        for key, value in self.maps.items():
            self.logs[key + '_rew_best'] = 0
            self.logs[key + '_ep_rews'] = 0
            self.logs[key + '_ep_rew'] = np.zeros(value)

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
            if not self.args.distill_restore:
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
