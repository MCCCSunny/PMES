import random
from multiprocessing import Process, Pipe, Event
from pysc2.env import sc2_env


def make_envs(args):
    envs = []
    i = 0
    for m in args.maps:
        for _ in range(args.maps[m]):
            map_seed = None if args.seed is None else random.randint(0, 1e5)
            print('map ', i, ' ', m, ' seed ', map_seed)
            envargs = dict(map_name=m, step_mul=args.step_mul, game_steps_per_episode=0, random_seed=map_seed)
            envs.append(make_env(args.sz, **dict(envargs, visualize=False)))
            i += 1

    return EnvPool(envs)


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
def make_env(sz=32, **params):
    def _thunk():
        params['agent_interface_format'] = [
            sc2_env.AgentInterfaceFormat(feature_dimensions=sc2_env.Dimensions(screen=(sz, sz), minimap=(sz, sz)))
        ]
        env = sc2_env.SC2Env(**params)
        return env

    return _thunk


# based on https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# SC2Env::step expects actions list and returns obs list so we send [data] and process obs[0]
def worker(remote, env_fn_wrapper, quit):
    try:
        env = env_fn_wrapper.x()

        while not quit.is_set():
            cmd, data = remote.recv()
            if cmd == 'spec':
                remote.send((env.observation_spec(), env.action_spec()))
            elif cmd == 'step':
                obs = env.step([data])
                remote.send(obs[0])
            elif cmd == 'reset':
                obs = env.reset()
                remote.send(obs[0])
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'save_replay':
                env.save_replay(data)
            else:
                raise NotImplementedError

    except Exception as e:  # pylint: disable=W0703
        quit.set()
        print('--exit--')
        print(e)
        remote.send('exit!')
        remote.send('exit!!')
        remote.send('exit!!!')


class CloudpickleWrapper(object):
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class EnvPool(object):
    def __init__(self, env_fns):
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.quit = Event()
        self.ps = [
            Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn), self.quit))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)
        ]
        for p in self.ps:
            p.start()

    def spec(self):
        for remote in self.remotes:
            remote.send(('spec', None))
        self.check_quit()
        results = [remote.recv() for remote in self.remotes]
        self.check_quit()
        return results[0]

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.check_quit()
        results = [remote.recv() for remote in self.remotes]
        self.check_quit()
        return results

    def reset(self, i=None):
        if i is not None:
            self.remotes[i].send(('reset', None))
            self.check_quit()
            result = self.remotes[i].recv()
            self.check_quit()
            return result

        for remote in self.remotes:
            remote.send(('reset', None))
        self.check_quit()
        results = [remote.recv() for remote in self.remotes]
        self.check_quit()
        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for i, p in enumerate(self.ps):
            p.join()
            print('enviroment ', i, ' shutdown')

    def save_replay(self, replay_dir='PySC2Replays'):
        for remote in self.remotes:
            remote.send(('save_replay', replay_dir))
            import time
            time.sleep(1.5)

    def check_quit(self):
        if self.quit.is_set():
            print('some error happened, wait for quiting...')
            self.close()
            import sys
            sys.exit(-1)

    @property
    def num_envs(self):
        return len(self.remotes)
