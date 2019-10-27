"""
Training helpers for supervised meta-learning.
"""

import os
import time

import numpy as np
import tensorflow as tf
import tqdm

from .reptile import Reptile
from .variables import weight_decay

# pylint: disable=R0913,R0914
def train(sess,
          model,
          train_set,
          test_set,
          save_dir,
          num_classes=5,
          num_shots=5,
          inner_batch_size=5,
          inner_iters=20,
          replacement=False,
          meta_step_size=0.1,
          meta_step_size_final=0.1,
          meta_batch_size=1,
          meta_iters=400000,
          eval_inner_batch_size=5,
          eval_inner_iters=50,
          eval_interval=100,
          weight_decay_rate=1,
          time_deadline=None,
          train_shots=None,
          transductive=False,
          adaptive=0.0,
          reptile_fn=Reptile,
          log_fn=print):
    """
    Train a model on a dataset.
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    saver = tf.train.Saver()
    reptile = reptile_fn(sess,
                         #transductive=transductive,
                         pre_step_op=weight_decay(weight_decay_rate))
    accuracy_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('accuracy', accuracy_ph)
    loss_ph = tf.placeholder(tf.float32, shape=())
    tf.summary.scalar('loss', loss_ph)
    merged = tf.summary.merge_all()
    writers = {('regular',): {'train': tf.summary.FileWriter(os.path.join(save_dir, 'train'), sess.graph),
                              'test': tf.summary.FileWriter(os.path.join(save_dir, 'test'), sess.graph)}}
    if transductive:
        writers[('transductive',)] = {'train': tf.summary.FileWriter(os.path.join(save_dir, 'transductive_train'), sess.graph),
                                      'test': tf.summary.FileWriter(os.path.join(save_dir, 'transductive_test'), sess.graph)}
    if adaptive:
        writers[('adaptive',)] = {'train': tf.summary.FileWriter(os.path.join(save_dir, 'adaptive_train'), sess.graph),
                                  'test': tf.summary.FileWriter(os.path.join(save_dir, 'adaptive_test'), sess.graph)}
        if transductive:
            writers[('adaptive', 'transductive')] = {'train': tf.summary.FileWriter(os.path.join(save_dir, 'adaptive_transductive_train'), sess.graph),
                                                     'test': tf.summary.FileWriter(os.path.join(save_dir, 'adaptive_transductive_test'), sess.graph)}
    tf.global_variables_initializer().run()
    sess.run(tf.global_variables_initializer())

    accs = {'train': [], 'test': []}
    with tqdm.trange(meta_iters) as pbar:
        for i in pbar:
            frac_done = i / meta_iters
            cur_meta_step_size = frac_done * meta_step_size_final + (1 - frac_done) * meta_step_size
            reptile.train_step(train_set, model.input_ph, model.label_ph, model.minimize_op,
                               num_classes=num_classes, num_shots=(train_shots or num_shots),
                               inner_batch_size=inner_batch_size, inner_iters=inner_iters,
                               replacement=replacement,
                               meta_step_size=cur_meta_step_size, meta_batch_size=meta_batch_size)
            if i % eval_interval == 0:
                desc = 'meta-acc: '
                for ev in writers.keys():
                    for mode, dataset in [('train', train_set), ('test', test_set)]:
                        writer = writers[ev][mode]
                        minimize_op = model.evaluate_ops[adaptive] if 'adaptive' in ev else model.evaluate_ops[0.0]
                        correct, total_loss = reptile.evaluate(dataset, model.input_ph, model.label_ph,
                                                   minimize_op, [model.predictions, model.loss],
                                                   num_classes=num_classes, num_shots=num_shots,
                                                   inner_batch_size=eval_inner_batch_size,
                                                   inner_iters=eval_inner_iters, 
                                                   transductive='transductive' in ev,
                                                   replacement=replacement)
                        summary = sess.run(merged, feed_dict={accuracy_ph: correct/num_classes, loss_ph: total_loss/num_classes})
                        writer.add_summary(summary, i)
                        writer.flush()
                        if ev[0] == 'regular':
                            accs[mode].append(correct / num_classes)
                            desc += mode + '=' + str(round(np.mean(accs[mode]), 4)) + ', '

                pbar.set_description(desc[:-2])
            if i % 5000 == 0 or i == meta_iters-1:
                saver.save(sess, os.path.join(save_dir, 'model.ckpt'), global_step=i)
            if time_deadline is not None and time.time() > time_deadline:
                break
