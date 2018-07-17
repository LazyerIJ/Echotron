import tensorflow as tf

basic_params={
    'data_dir':'./data/',
    'fname_x':'ij_hello_1.wav',
    'fname_y':'yr_hello_1.wav',
    'rate_hz':22050,
    'fft_size':2048,
    'hopsamp':2048//8,
    'iterations':50,
}

hparams = tf.contrib.training.HParams(**basic_params)
