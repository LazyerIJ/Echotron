import argparse
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense
from keras.models import Model,Sequential
import itertools
import argparse
import librosa
import scipy.io.wavfile
import audio_utilities
import numpy as np
from convModel import *
from pylab import *
from hparams import basic_params as b_params
import os

tf.logging.set_verbosity(tf.logging.INFO)

data_dir=b_params['data_dir']
fname_x = os.path.join(data_dir,'ij_hello_1.wav')
fname_x_2 = os.path.join(data_dir,'ij_hello_2.wav')
fname_y = os.path.join(data_dir,'yr_hello_1.wav')
rate_hz = b_params['rate_hz']
fft_size = b_params['fft_size']
hopsamp = b_params['hopsamp']
iterations=b_params['iterations']

def get_stft_modified(fname,hopsamp,rate_hz=44100,
                      fft_size=2048,enableMel=False,enableFilter=False):

    input_signal = audio_utilities.get_signal(fname,expected_fs=rate_hz)
    stft_full = audio_utilities.stft_for_reconstruction(input_signal, 
                                                        fft_size,hopsamp)

    stft_mag = abs(stft_full)**2.0
    scale = 1.0 / np.amax(stft_mag)

    stft_mag *= scale
    stft_modified = stft_mag

    print('[*]stft_modified : ' , stft_modified.shape)
    imshow(stft_mag.T**0.125, origin='lower', cmap=cm.hot, aspect='auto',
           interpolation='nearest')
    colorbar()
    title('Unmodified spectrogram')
    xlabel('time index')
    ylabel('frequency bin index')
    savefig(fname+'unmodified_spectrogram.png', dpi=150)

    if enableMel:
        min_freq_hz = 70
        max_freq_hz = 8000
        mel_bin_count = 200
        linear_bin_count = 1 + fft_size//2
        filterbank = audio_utilities.make_mel_filterbank(min_freq_hz, 
                                                         max_freq_hz, 
                                                         mel_bin_count,
                                                         linear_bin_count , 
                                                         rate_hz)

        mel_spectrogram = np.dot(filterbank, stft_mag.T)
        inverted_mel_to_linear_freq_spectrogram = np.dot(filterbank.T, 
                                                         mel_spectrogram)

        stft_modified = inverted_mel_to_linear_freq_spectrogram.T

    if enableFilter:

        cutoff_freq = 1000
        cutoff_bin = round(cutoff_freq*fft_size/rate_hz)
        stft_modified[:, cutoff_bin:] = 0
    
    return stft_modified,scale
   
def get_wav_from_stft(stft_modified,rate_hz,modified_scale,fft_size,
                      hopsamp,iterations=2000,outfile=None):

    stft_modified_scaled = stft_modified / modified_scale
    stft_modified_scaled = stft_modified_scaled**0.5

    x_reconstruct = audio_utilities.reconstruct_signal_griffin_lim\
            (stft_modified_scaled,
             fft_size=fft_size,
             hopsamp=hopsamp,
             iterations=iterations)

    max_sample = np.max(abs(x_reconstruct))
    if max_sample>1.0:
        x_reconstruct = x_reconstruct / max_sample

    audio_utilities.save_audio_to_file(x_reconstruct,
                                       rate_hz,
                                       outfile = outfile)

    return stft_modified_scaled

def autoencoder(x_train, x_test, data_dims=None, encoding_dim=512,iterations=128):

    if data_dims==None:
        data_dims = x_train.shape[1]

    input_img = Input(shape=(data_dims,))
    model = Sequential()
    model.add(Dense(256, input_dim=data_dims))
    model.add(Dense(256,activation='tanh'))
    model.add(Dense(data_dims, activation='relu'))
    model.compile(optimizer='adam',loss='binary_crossentropy')
    model.fit(x_train,x_test,nb_epoch=iterations,shuffle=False,validation_data=(x_train,x_test))

    return model

def split_data(x,split_range):

    split_num = x.shape[0]//split_range+1

    for step in range(split_num):
        idx_start = step*split_range
        idx_end = (step+1)*split_range
        print('{} ~ {}'.format(idx_start,idx_end))
        data = x[idx_start:idx_end,:]
        if len(data)>0:
            yield x[idx_start:idx_end,:]


def train(split_range,x,y):

    result = np.zeros((x.shape)) 
    model_list = []

    for x,y in zip(split_data(x=x,split_range=split_range),
                   split_data(x=y,split_range=split_range)):

        model = autoencoder(x,y,iterations=iterations)
        model_list.append(model)

    return model_list

def test(split_range,x,model_list):
    result = []
    for x,model in zip(split_data(x=x,split_range=split_range), model_list):
        result.append(model.predict(x))

    return np.array(np.concatenate([t for t in result]))


if __name__=='__main__':

    x_stft,x_scale = get_stft_modified(fname=fname_x,
                                       rate_hz=rate_hz,
                                       fft_size=fft_size,
                                       hopsamp=hopsamp)

    y_stft,y_scale = get_stft_modified(fname=fname_y,
                                       rate_hz=rate_hz,
                                       fft_size=fft_size,
                                       hopsamp=hopsamp)

    x_stft2,x_scale2 = get_stft_modified(fname=fname_x_2,
                                       rate_hz=rate_hz,
                                       fft_size=fft_size,
                                       hopsamp=hopsamp)

    print('[*]x_stft_modified : ' , x_stft.shape)#(199,1025)
    print('[*]x_scaled : ' , x_scale)
    print('[*]y_stft_modified : ' , y_stft.shape)#(199,1025)
    print('[*]y_scaled : ' , y_scale)

    split_range=5

    model_list = train(x=x_stft,y=y_stft,split_range=split_range)

    ##test_1
    result1 = test(x=x_stft,split_range=split_range, model_list=model_list)
    get_wav_from_stft(stft_modified=result1,
                      modified_scale =x_scale, 
                      rate_hz=rate_hz,
                      fft_size=fft_size,
                      hopsamp=hopsamp,
                      iterations=iterations,
                      outfile="output_test1.wav")
    ##test_2

    result2 = test(x=x_stft2,split_range=split_range)
    get_wav_from_stft(stft_modified=result2,
                      modified_scale =x_scale2, 
                      rate_hz=rate_hz,
                      fft_size=fft_size,
                      hopsamp=hopsamp,
                      iterations=iterations,
                      outfile="output_test2.wav")
