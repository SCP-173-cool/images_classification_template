"""
Created on Fri May 18 23:52:46 2018

@author: loktar
"""

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def restore_variable_into_metaGraph(path, model_name, sess):
    model_path = os.path.join(path, model_name)
    meta_file = model_path+'.meta'

    restore_saver = tf.train.import_meta_graph(meta_file)
    restore_saver.restore(sess, model_path)
    return sess

if __name__ == '__main__':
    from glob import glob
    import numpy as np
    path = '../output/train_results/checkpoints/'
    model_name = glob(os.path.join(path, '*.meta'))[0].split('/')[-1].split('.meta')[0]
    print("Restore from {}, the model name is {} ...".format(path, model_name))

    sess = tf.Session()
    sess = restore_variable_into_metaGraph(path, model_name, sess)

    print("Trainable Parameters:")
    all_total = 0
    for i in tf.trainable_variables():
        params_num = np.prod(i.get_shape().as_list())
        print(i.name, '--->', params_num)
        all_total += params_num
    print("=================================================================")
    print("All params number is {}".format(all_total))
    sess.close()
