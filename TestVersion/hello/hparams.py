import tensorflow as tf

basic_params={
    'base_dir':'input/',
    'data_dir':'data/',
    'target_dir':'target/',
    'rate_hz':22050,
    'fft_size':1024,
    'hopsamp':1024//8,
    'iterations':100,
    'split_range':5,
}

hparams = tf.contrib.training.HParams(**basic_params)
