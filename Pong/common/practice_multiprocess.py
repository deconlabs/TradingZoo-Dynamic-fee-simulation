import numpy as numpy
from multiprocessing import Process, Pipe

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd =="step":
            ob, reward, done, info = env.step(data)

            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info)) 
        elif cmd =='reset':
            ob = env.reset()
            remote.send(ob)

class VecEnv(object):

    def __ini

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids =np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids , : ] , actions[rand_ids, :], log_probs[rand_ids, : ] ,\
            returns, advantage[]
            
        