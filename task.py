import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import json
import os

from trainer.model import create_model

PER_REPLICA_BATCH_SIZE = 64

def get_args():
  '''Parses args.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--epochs',
      required=True,
      type=int,
      help='number training epochs')
  parser.add_argument(
      '--job-dir',
      required=True,
      type=str,
      help='bucket to save model')
  args = parser.parse_args()
  return args


def preprocess_data(image, label):
  '''Resizes and scales images.'''

  image = tf.image.resize(image, (300,300))
  return tf.cast(image, tf.float32) / 255., label


def create_dataset(batch_size):
  '''Loads Cassava dataset and preprocesses data.'''
  
  data, info = tfds.load(name='cassava', as_supervised=True, with_info=True)
  number_of_classes = info.features['label'].num_classes
  train_data = data['train'].map(preprocess_data, 
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_data  = train_data.shuffle(1000)
  train_data  = train_data.batch(batch_size)
  train_data  = train_data.prefetch(tf.data.experimental.AUTOTUNE)
  options = tf.data.Options()
  options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
  train_data = train_data.with_options(options) 
  return train_data, number_of_classes


def _is_chief(task_type, task_id):
  '''Determines of machine is chief.'''

  return task_type == 'chief'


def _get_temp_dir(dirpath, task_id):
  '''Gets temporary directory for saving model.'''

  base_dirpath = 'workertemp_' + str(task_id)
  temp_dir = os.path.join(dirpath, base_dirpath)
  tf.io.gfile.makedirs(temp_dir)
  return temp_dir


def write_filepath(filepath, task_type, task_id):
  '''Gets filepath to save model.'''

  dirpath = os.path.dirname(filepath)
  base = os.path.basename(filepath)
  if not _is_chief(task_type, task_id):
    dirpath = _get_temp_dir(dirpath, task_id)
  return os.path.join(dirpath, base)


def main():
  args = get_args()
  strategy = tf.distribute.MultiWorkerMirroredStrategy()
  global_batch_size = PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync
  train_data, number_of_classes = create_dataset(global_batch_size)

  with strategy.scope():
    model = create_model(number_of_classes)

  model.fit(train_data, epochs=args.epochs)

  # Determine type and task of the machine from 
  # the strategy cluster resolver
  task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)

  # Based on the type and task, write to the desired model path 
  write_model_path = write_filepath(args.job_dir, task_type, task_id)
  model.save(write_model_path)

if __name__ == "__main__":
    main()
