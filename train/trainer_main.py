from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from google.protobuf import text_format

import sys
import socket
import multiprocessing

sys.path.insert(0, '/specific/netapp5_2/gamir/achiya/vqa/Cap2Det_1st_attempt/')
sys.path.insert(1, '/specific/netapp5_2/gamir/achiya/vqa/Cap2Det_1st_attempt/tensorflow_models/research/slim')

from protos import pipeline_pb2
from train import trainer
from train.ckpts_saver import ckpts_saver

flags = tf.app.flags

tf.logging.set_verbosity(tf.logging.DEBUG)

flags.DEFINE_string('type', '', 'A message string passed from command-line.')

flags.DEFINE_string('pipeline_proto', '', 'Path to the pipeline proto file.')

flags.DEFINE_string('model_dir', '',
                    'Path to the directory which holds model checkpoints.')

FLAGS = flags.FLAGS


def _load_pipeline_proto(filename):
    """Loads pipeline proto from file.

    Args:
      filename: path to the pipeline config file.

    Returns:
      an instance of pipeline_pb2.Pipeline.
    """
    pipeline_proto = pipeline_pb2.Pipeline()
    with tf.gfile.GFile(filename, 'r') as fp:
        text_format.Merge(fp.read(), pipeline_proto)
    return pipeline_proto


def main(_):
    pipeline_proto = _load_pipeline_proto(FLAGS.pipeline_proto)

    if FLAGS.model_dir:
        pipeline_proto.model_dir = FLAGS.model_dir
        tf.logging.info("Override model checkpoint dir: %s", FLAGS.model_dir)

    tf.logging.info("Pipeline configure: %s", '=' * 128)
    tf.logging.info(pipeline_proto)

    trainer.create_train_and_evaluate(pipeline_proto)

    tf.logging.info('Done')


if __name__ == '__main__':
    print(f'Running on PC: {socket.gethostname()}')
    saver_process = multiprocessing.Process(target=ckpts_saver, args=(FLAGS.model_dir, ), daemon=True)
    saver_process.start()
    tf.app.run()
