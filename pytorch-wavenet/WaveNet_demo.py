import torch
from wavenet_model import *
from audio_data import WavenetDataset
from wavenet_training import *
from model_logging import *
from local_condition_dataset import LocalConditionedDataset

# initialize cuda option
dtype = torch.FloatTensor # data type
ltype = torch.LongTensor # label type

use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor
else:
    print('no gpu')

local_conditioning = True

model = WaveNetModel(layers=10,
                     blocks=3,
                     dilation_channels=32,
                     residual_channels=32,
                     skip_channels=1024,
                     end_channels=512,
                     output_length=16,
                     dtype=dtype,
                     bias=True,
                     local_conditioning = local_conditioning)
# model = load_latest_model_from('snapshots', use_cuda=use_cuda)

if use_cuda:
  model.cuda()

print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

waveform_data = WavenetDataset(dataset_file='../CS230PianoUbyte.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
                      test_stride=500)

local_condition = None

if local_conditioning:
    local_condition = WavenetDataset(dataset_file='../CS230PianoUbyteFreqs.npz',
                        item_length=model.receptive_field + model.output_length - 1,
                        target_length=model.output_length,
                        test_stride=500)

data = LocalConditionedDataset(waveform_data, local_condition, train=True, target_length=model.output_length)
print('the dataset has ' + str(len(data)) + ' items')

def generate_and_log_samples(step):
    sample_length=32000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=16000)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=16000)
    print("audio clips generated")


# logger = TensorboardLogger(log_interval=200,
#                            validation_interval=400,
#                            generate_interval=1000,
#                            generate_function=generate_and_log_samples,
#                            log_dir="logs/chaconne_model")

logger = Logger(log_interval=200,
                validation_interval=400,
                generate_interval=1000)

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.001,
                         snapshot_path='snapshots',
                         snapshot_name='chaconne_model',
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=16,
              epochs=10)

start_inputs = data[len(data) // 2] # use start data from the data set. Halfway through the file.
condition_data = None
if local_conditioning:
    start_data, condition_data = start_inputs
else:
    start_data = start_inputs
start_data = start_data[0]
start_data = torch.max(start_data, 0)[1] # convert one hot vectors to integers

if condition_data is not None:
    condition_data = torch.max(condition_data, 0)[1] # convert one hot vectors to integers

def prog_callback(step, total_steps):
    print(str(100 * step // total_steps) + "% generated")

generated = model.generate_fast(num_samples=160000,
                                 first_samples=start_data,
                                 progress_callback=prog_callback,
                                 progress_interval=1000,
                                 temperature=1.0,
                                 regularize=0.,
                                 local_condition=condition_data)

import IPython.display as ipd

ipd.Audio(generated, rate=16000)

