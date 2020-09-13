import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import pdb
import tensorflow.contrib.slim as slim

class A2CAgent_expert:
    def __init__(self, sess_expert, model_fn, config, args, path):
        gg = tf.Graph()
        with gg.as_default():
            self.config = config
            (self.policy_e, self.value_e), self.input_e = model_fn(config)
            self.action_e = [sample(p) for p in self.policy_e]
            self.step = tf.Variable(0, trainable=False)
        
        sess_expert = self.sess_expert = tf.Session(graph=gg)
        with gg.as_default():
            self.saver = tf.train.Saver(max_to_keep=args.num_snapshot) # the main difference
            self.saver.restore(self.sess_expert, tf.train.get_checkpoint_state(path).model_checkpoint_path)

    def act_expert(self, state):
        '''
        the input value from the student model
        '''
        policy_e, action_e, value_e = self.sess_expert.run([self.policy_e, self.action_e, self.value_e],
                                    feed_dict=dict(zip(self.input_e, state)))
        return policy_e, action_e, value_e

    def get_value(self, state):
        return self.sess_expert.run(self.value_e, feed_dict=dict(zip(self.input_e, state)))

    def get_global_step(self):
        return self.sess_expert.run(self.step)

class A2CAgent_distill:
    def __init__(self, sess, agent_eA, agent_eB, model_fn, config, weight_dir, log_dir, args):
        gg = tf.Graph()
        with gg.as_default():
            self.config, self.discount = config, args.discount
            self.vf_coef, self.ent_coef = args.vf_coef, args.ent_coef
            self.weight_dir = weight_dir
            self.log_dir = log_dir
            self.save_interval = args.save_interval if args is not None else 500
            self.args = args
            (self.policy, self.value), self.inputs = model_fn(config)
            self.action = [sample(p) for p in self.policy]
            loss_distill, self.loss_distill_inputs, self.distill_policyLoss_scalar, self.distill_value_loss_scalar = self._distill_loss_func()

            self.step = tf.Variable(0, trainable=False)
            if self.args.lr_decay:
                self.lr = tf.train.exponential_decay(args.lr, self.step, args.lr_decay_step, args.lr_decay_rate, staircase=True)
            else:
                self.lr = args.lr
            self.next_best = args.save_best_start
            self.best_interval = args.save_best_inc
            if args.optimizer == 'adam':
                opt = tf.train.AdamOptimizer(
                    learning_rate=self.lr, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
            elif args.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(
                    learning_rate=self.lr, decay=args.decay, momentum=args.momentum, epsilon=args.epsilon)

            self.train_op_distill = layers.optimize_loss(
                loss=loss_distill, optimizer=opt, learning_rate=None, global_step=self.step)

        sess = self.sess = tf.Session(graph=gg)
        print('weights restored from', args.paths[-1])

        # restore the weights by restore
        with gg.as_default():              
            self.saver = tf.train.Saver(max_to_keep=args.num_snapshot) # the main difference
            print ('restrore the model from', tf.train.get_checkpoint_state(args.paths[-1]).model_checkpoint_path)
            self.saver.restore(self.sess, tf.train.get_checkpoint_state(args.paths[-1]).model_checkpoint_path)
            if not self.args.Student_restore:
                self.sess.run(tf.variables_initializer(opt.variables()))
                self.sess.run(tf.assign(self.step, 0))
                print ('the global step start from', sess.run(self.step))
                print ('the optimizer is initialized')          
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=None)
            self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))

        self.eA_input = agent_eA.input_e
        self.eB_input = agent_eB.input_e


    def train_distill(self, step, states, policys_expert, rewards, dones, done_values, last_value, values_expert):
        '''
        inputs: (states, Actions_experts, rewards, dones, done_values, last_value, values_expert, values_distill) values not tensor
        values_expert: (16,16)
        '''
        print(f'starting training {step} step model')
        if step % self.save_interval == 0:
            self.save()
        returns = self._compute_returns(rewards, dones, done_values, last_value) #values # (256,)
        values_expert = values_expert.reshape(-1,)
        feed_dict = dict(zip(self.inputs + self.loss_distill_inputs, states + [returns] + policys_expert + [values_expert]))
        result, result_summary, step = self.sess.run([self.train_op_distill, self.summary_op, self.step], feed_dict)

        self.summary_writer.add_summary(result_summary, step)

        return result

    def save(self):
        self.saver.save(self.sess, os.path.join(self.weight_dir, 'a2c'), global_step=self.step)

    def save_best(self):
        self.best_saver.save(self.sess, os.path.join(self.weight_dir, 'a2c'), global_step=self.step)

    def act(self, state):
        return self.sess.run([self.action, self.value], feed_dict=dict(zip(self.inputs, state)))

    def get_value(self, state):
        return self.sess.run(self.value, feed_dict=dict(zip(self.inputs, state)))

    def get_inputs(self, state):
        return self.sess.run(self.inputs, feed_dict=dict(zip(self.inputs, state)))

    def get_global_step(self):
        return self.sess.run(self.step)

    def _distill_loss_func(self):
        """
        the loss of distill model
        """
        returns = tf.placeholder(tf.float32, [None]) # (256,)
        policys_expert = [tf.placeholder(tf.float32, one.shape) for one in self.policy] # list, len=14, each is (256,*)
        values_expert = tf.placeholder(tf.float32, [None])

        policys_expert = [clip_log(one) for one in policys_expert]
        policys_distill = [clip_log(one) for one in self.policy]

        policyLossList = [] # list len=14, each is (256,)
        for policy_i in range(len(policys_distill)):
            distill_action_loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policys_expert[policy_i],logits=policys_distill[policy_i])
            policyLossList.append(distill_action_loss_i)
            
        distill_policyLoss = tf.reduce_mean(policyLossList) * self.args.policy_coef
        distill_value_loss = tf.reduce_mean(tf.square(self.value-values_expert))     # the MSE value of values

        distill_policyLoss_scalar = tf.summary.scalar('loss/distill_policy', distill_policyLoss)
        distill_value_loss_scalar = tf.summary.scalar('loss/distill_value', distill_value_loss)

        return distill_value_loss + distill_policyLoss, [returns] + policys_expert + [values_expert], distill_policyLoss_scalar, distill_value_loss_scalar


    def _compute_returns(self, rewards, dones, done_values, last_value):
        returns = np.zeros((dones.shape[0] + 1, dones.shape[1]))
        returns[-1] = last_value
        for t in reversed(range(dones.shape[0])):
            returns[t] = rewards[t] + self.discount * returns[t + 1] * (1 - dones[t]) + dones[t] * done_values[t]
        returns = returns[:-1]
        return returns.flatten()




class A2CAgent:
    def __init__(self,
                 sess,
                 model_fn,
                 config,
                 discount=0.99,
                 lr=1e-4,
                 vf_coef=0.25,
                 ent_coef=1e-3,
                 clip_grads=1.,
                 weight_dir=None,
                 log_dir=None,
                 args=None):
        self.sess, self.config, self.discount = sess, config, discount
        self.vf_coef, self.ent_coef = vf_coef, ent_coef
        self.weight_dir = weight_dir
        self.log_dir = log_dir
        self.save_interval = args.save_interval if args is not None else 500
        self.args=args

        (self.policy, self.value), self.inputs = model_fn(config)
        self.action = [sample(p) for p in self.policy]
        loss_fn, self.loss_inputs, self.policy_scalar, self.entropy_scalar, self.value_scalar = self._loss_func()

        self.step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(lr, self.step, args.lr_decay_step, args.lr_decay_rate, staircase=True)
        self.next_best = args.save_best_start
        self.best_interval = args.save_best_inc
        if args.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(
                learning_rate=self.lr, beta1=args.beta1, beta2=args.beta2, epsilon=args.epsilon)
        elif args.optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(
                learning_rate=self.lr, decay=args.decay, momentum=args.momentum, epsilon=args.epsilon)

        self.train_op = layers.optimize_loss(
            loss=loss_fn, optimizer=opt, learning_rate=None, global_step=self.step, clip_gradients=clip_grads)

        if not args.distill_restore:    #if restore from the model which restored from the distill model or not adopt the distill model
            print ('restore the model by the original way: saver and merge all')
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=args.num_snapshot)
            self.best_saver = tf.train.Saver(max_to_keep=1)

            if args.ckptfile:
                print('weights restored from ', args.ckptfile)
                tf.reset_default_graph()
                self.saver.restore(self.sess, args.ckptPath+args.ckptfile)
            elif args.restore_each:
                print('weights restored from ', args.restore_each)
                tf.reset_default_graph()
                self.saver.restore(self.sess, args.restore_path+args.restore_each)
            else:
                if args.restore_path is not None:
                    print('weights restored from ', args.restore_path)
                    print('weights from', tf.train.latest_checkpoint(args.restore_path))
                    self.saver.restore(self.sess, tf.train.latest_checkpoint(args.restore_path))
            if self.args.init_op:
                print ('init the optimizer and the step')
                print ('the original step is', self.sess.run(self.step))
                self.sess.run(tf.variables_initializer(opt.variables()))
                self.sess.run(tf.assign(self.step, 0))
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=None)
            self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(self.step))
            
        elif args.distill_restore:  # restore the model from the distill model
            print ('weights restored from distillation', args.restore_path)
            metafile = tf.train.get_checkpoint_state(args.restore_path).model_checkpoint_path+'.meta'
            print('restore weights is', metafile)
            self.saver = tf.train.import_meta_graph(metafile)
            self.saver.restore(self.sess, tf.train.get_checkpoint_state(args.restore_path).model_checkpoint_path)
            self.sess.run(tf.global_variables_initializer())

            self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=self.sess.graph)
            self.summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), self.sess.run(self.step))
        else:
            print ('the parameter is wrong!')
            sys.exit(1)

    def train(self, step, states, actions, rewards, dones, done_values, last_value, log_dict):
        if step % self.save_interval == 0:
            self.save()

        returns = self._compute_returns(rewards, dones, done_values, last_value)

        feed_dict = dict(zip(self.inputs + self.loss_inputs, states + actions + [returns]))
        if not self.args.distill_restore:
            result, result_summary, step = self.sess.run([self.train_op, self.summary_op, self.step], feed_dict)
            self.summary_writer.add_summary(result_summary, step)
            self.summary_writer.add_summary(summarize(log_dict), step)
        else:
            # record the rewards
            summary = tf.Summary(value=[tf.Summary.Value(tag="rew_mean", simple_value=np.mean(rewards)), tf.Summary.Value(tag="rew_max", simple_value=np.max(rewards))])
            self.summary_writer.add_summary(summary, step)
            result, policy_summary, entropy_summary, value_summary, step = self.sess.run([self.train_op, self.policy_scalar, self.entropy_scalar, self.value_scalar, self.step], feed_dict)
            self.summary_writer.add_summary(policy_summary, step)
            self.summary_writer.add_summary(entropy_summary, step)
            self.summary_writer.add_summary(value_summary, step)
            self.summary_writer.add_summary(summarize(log_dict), step)

        return result


    def save(self):
        self.saver.save(self.sess, os.path.join(self.weight_dir, 'a2c'), global_step=self.step)

    def save_best(self):
        self.best_saver.save(self.sess, os.path.join(self.weight_dir, 'a2c'), global_step=self.step)

    def act(self, state):
        return self.sess.run([self.action, self.value], feed_dict=dict(zip(self.inputs, state)))

    def get_value(self, state):
        return self.sess.run(self.value, feed_dict=dict(zip(self.inputs, state)))

    def get_global_step(self):
        return self.sess.run(self.step)

    def get_lr(self):
        return self.sess.run(self.lr)

    def _loss_func(self):
        """return policy_loss + entropy_loss + value_loss, actions + [returns]"""
        returns = tf.placeholder(tf.float32, [None])
        actions = [tf.placeholder(tf.int32, [None]) for _ in range(len(self.policy))] # len=14

        adv = tf.stop_gradient(returns - self.value)
        logli = sum([clip_log(select(a, p)) for a, p in zip(actions, self.policy)])
        entropy = sum([-tf.reduce_sum(p * clip_log(p), axis=-1) for p in self.policy])

        policy_loss = -tf.reduce_mean(logli * adv)
        entropy_loss = -self.ent_coef * tf.reduce_mean(entropy)
        value_loss = self.vf_coef * tf.reduce_mean(tf.square(returns - self.value))

        policy_scalar = tf.summary.scalar('loss/policy', policy_loss)
        entropy_scalar = tf.summary.scalar('loss/entropy', entropy_loss)
        value_scalar = tf.summary.scalar('loss/value', value_loss)

        return policy_loss + entropy_loss + value_loss, actions + [returns], policy_scalar, entropy_scalar, value_scalar

    def _compute_returns(self, rewards, dones, done_values, last_value):
        returns = np.zeros((dones.shape[0] + 1, dones.shape[1]))
        returns[-1] = last_value
        for t in reversed(range(dones.shape[0])):
            returns[t] = rewards[t] + self.discount * returns[t + 1] * (1 - dones[t]) + dones[t] * done_values[t]
        returns = returns[:-1] # (16,16)
        return returns.flatten() #(256,)


def select(acts, policy):
    return tf.gather_nd(policy, tf.stack([tf.range(tf.shape(policy)[0]), acts], axis=1))


# based on https://github.com/pekaalto/sc2aibot/blob/master/common/util.py#L5-L11
def sample(probs):
    u = tf.random_uniform(tf.shape(probs))
    return tf.argmax(tf.log(u) / probs, axis=1)


def clip_log(probs):
    return tf.log(tf.clip_by_value(probs, 1e-12, 1.0))


def summarize(info):
    summary = tf.Summary()
    for k, v in info.items():
        summary.value.add(tag=k, simple_value=v)
    return summary
