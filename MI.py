from depen import *

def clean_data(data_epochs, new_rate = 100, channels = "MI"):
  if channels == "MI":
    channel_choosen = create_channel_list("FC", "5/3/1/2/4/6") + create_channel_list("C", "5/3/1/z/2/4/6") + create_channel_list("CP", "5/3/1/z/2/4/6")
  else:
    channel_choosen = channels
  data_epochs.pick(channel_choosen)
  if new_rate is not None:
    data_epochs.resample(new_rate)
  print("new data shape: ", data_epochs.get_data().shape)
  return data_epochs

def plot_spectral(data_epochs, fmin = 5, fmax = 50, channels = ['C4'], left = 'b', right = 'r'):
  psds, freqs = mne.time_frequency.psd_multitaper(data_epochs["labels_name == 'left'"], fmin = fmin, fmax = fmax, picks = channels)
  psds1, freqs1 = mne.time_frequency.psd_multitaper(data_epochs["labels_name == 'right'"], fmin = fmin, fmax = fmax, picks = channels)

  plt.plot(freqs, np.mean(psds, axis=0)[0], left, freqs1, np.mean(psds1, axis=0)[0], right) # #left - blue, right - red
  plt.title('PSD: power spectral density')
  plt.xlabel('Frequency')
  plt.ylabel('Power')
  plt.tight_layout()

def preproc_data(x, l_freq = 8, h_freq = 30, picks = None, order = 5):
  iir_params = dict(order = order, ftype='butter')
  x.filter(l_freq = l_freq, h_freq = h_freq, picks=picks, method='iir', iir_params=iir_params,)
  return x

def csp_time_window(x, low = 100, high = 350): #0 - right, 1 - left
  left_data = x["ids == 1"].copy().get_data()[:,:, low:high]
  right_data = x["ids == 0"].copy().get_data()[:, :, low:high]
  left_data = np.transpose(left_data, (1, 0, 2))
  right_data = np.transpose(right_data, (1, 0, 2))
  #
  left_flatten = left_data.reshape((20, -1))
  right_flatten = right_data.reshape((20, -1))
  #
  cov_l = np.cov(left_flatten)#np.dot(left_data, left_data.T)#/left_data.shape[1]
  cov_r = np.cov(right_flatten)#np.dot(right_data, right_data.T)#/right_data.shape[1]
  #
  d, W = scipy.linalg.eigh(a=cov_l, b=cov_l+cov_r)
  
  iii = 0
  xxx = W[:, iii]
  print("checking eigenvector ", iii)
  print(np.var(np.dot(xxx.T, left_flatten)))
  print(np.var(np.dot(xxx.T, right_flatten)))  
  
  return d, W

def choose_csp_filters(W, positions = 2):
  if type(positions) == int:
    csp = np.concatenate((W[:, 0:positions], W[:, (len(W[0])-positions):]), axis = 1)
  else:
    csp = W[:, positions]
  print("csp shape: ", csp.shape)
  return csp


def forward_csp(data, csp, low = 100, high = 350):
  data = data.copy().get_data()[:,:, low:high]
  data = np.transpose(data, (1, 0, 2))
  result = np.einsum('lk,knb->nlb', csp.T, data)
  var = np.var(result, axis=2)
  var = np.log(var)
  return var


class MI_csp_lda():
  def __init__(self):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    self.lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.25)
  
  def train_on_data(self, data):
    data = get_data(data)
    data = clean_data(data)
    plot_spectral(data)
    data = preproc_data(data)
    plot_spectral(data, left = 'cyan', right= 'pink')    

    d, W = csp_time_window(data)
    self.csp = choose_csp_filters(W)

    var_left = forward_csp(data["labels_name == 'left'"], self.csp)
    var_right = forward_csp(data["labels_name == 'right'"], self.csp)

    data_points = np.concatenate((var_left, var_right), axis = 0)
    labels = [1] * 50 + [0] * 50
    print(labels)
    self.lda.fit(data_points, labels)

    print(self.lda.predict([var_right[0]]))
  
  def check_data(self, x):
    
    x = get_data(x)
    x = clean_data(x)

    plot_spectral(x) #left - blue, right - red

    x = preproc_data(x)
    plot_spectral(x, left = 'cyan', right= 'pink')

    labels = x.metadata['ids'].tolist()

    var = forward_csp(x, self.csp)

    label_list = self.lda.predict(var)
    acc = self.lda.score(var, labels)

    print(
      "label_list: ",
      label_list, "\n accuracy: ", 
      acc,
    )

    return acc, label_list

"""

d, W = csp_time_window(vau)

csp = choose_csp_filters(W)

print(vau["labels_name == 'left'"].get_data().shape)
var_left = forward_csp(vau["labels_name == 'left'"], csp)
var_right = forward_csp(vau["labels_name == 'right'"], csp)

for_gr11 = var_left[:, 0]
for_gr12 = var_left[:, 3]
for_gr21 = var_right[:, 0]
for_gr22 = var_right[:, 3]

fig, ax = plt.subplots()

ax.scatter(for_gr11, for_gr12, c = 'b') 
ax.scatter(for_gr21, for_gr22, c = 'r') 

plt.show()

"""


"""

raw_hilb = epochs_data.copy()
hilb_picks = mne.pick_types(epochs_data.info, meg=False, eeg=True)
raw_hilb.apply_hilbert(hilb_picks, envelope=True, )
evokeds = dict()
query = "labels_name == '{}'"
for label in raw_hilb.metadata['labels_name'].unique():
    aaa = raw_hilb[query.format(label)].average().copy()
    aaa.data = (aaa.data - np.expand_dims(np.mean(aaa.data, axis=1), axis = 1))
    evokeds[label] = aaa

mne.viz.plot_compare_evokeds(evokeds, cmap=('label', 'viridis'),
                             picks='C4')
                             
                             
"""