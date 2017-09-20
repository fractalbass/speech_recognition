from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt


#Read in the audio file
(rate,sig) = wav.read("./data/miles/one.wav")

# Calculate the mfcc features based on the file data
mfcc_feat = mfcc(sig, rate, nfft=1200)

# Calculate the filterbank from the audio file
fbank_feat = logfbank(sig, rate, nfft=1200)

#Print the result
print(fbank_feat[1:3, :])

filterbank_features = fbank_feat.T
plt.matshow(filterbank_features)
plt.title('Filter bank')

plt.show()
