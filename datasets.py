from depen import *

class BCI_Dataset(Dataset) :
    def __init__(self, path_MI, path_ERP, path_SSVEP, resample = 100, concatenate_train_test = True, decrease_erp = True, typo = "train"): 
      data_mi_train, data_mi_test = get_mat_file(path_MI)
      data_mi_train, data_mi_test = get_data(data_mi_train), get_data(data_mi_test)
      
      data_erp_train, data_erp_test = get_mat_file(path_ERP, types='EEG_ERP')
      data_erp_train, data_erp_test = get_data_erp(data_erp_train), get_data_erp(data_erp_test)

      data_ssvep_train, data_ssvep_test = get_mat_file(path_SSVEP, types='EEG_SSVEP')
      data_ssvep_train, data_ssvep_test = get_data(data_ssvep_train), get_data(data_ssvep_test)

      if resample is not None:
        data_mi_train.resample(resample)
        data_mi_test.resample(resample)
        data_erp_train.resample(resample)
        data_erp_test.resample(resample)
        data_ssvep_train.resample(resample)
        data_ssvep_test.resample(resample)
      
      print(
            "data_mi_train: ", data_mi_train.get_data().shape, "\n",
            "data_mi_test: ", data_mi_test.get_data().shape, "\n",
            
            "data_erp_train: ", data_erp_train.get_data().shape, "\n",
            "data_erp_test: ", data_erp_test.get_data().shape, "\n",
            
            "data_ssvep_train: ", data_ssvep_train.get_data().shape, "\n",
            "data_ssvep_test: ", data_ssvep_test.get_data().shape, "\n",
            )
      
      mi_train = data_mi_train.get_data()
      mi_test = data_mi_test.get_data()
      mi_labels_train = data_mi_train.metadata['ids'].to_numpy()
      mi_labels_test = data_mi_test.metadata['ids'].to_numpy()


      if decrease_erp:
        randomlist = random.sample(range(0, data_erp_train.get_data().shape[0]), 200)
      else:
        randomlist = list(range(0, data_erp_train.get_data().shape[0]))
      
      erp_train = data_erp_train.get_data()[randomlist]
      erp_test = data_erp_test.get_data()[randomlist]
      erp_labels_train = data_erp_train.metadata['ids'].to_numpy()[randomlist]
      erp_labels_test = data_erp_test.metadata['ids'].to_numpy()[randomlist]

      ssvep_train = data_ssvep_train.get_data()
      ssvep_test = data_ssvep_test.get_data()
      ssvep_labels_train = data_ssvep_train.metadata['ids'].to_numpy()
      ssvep_labels_test = data_ssvep_test.metadata['ids'].to_numpy()

      print(
            "data_mi_train: ", mi_train.shape, "\n",
            "data_mi_test: ", mi_test.shape, "\n",
            
            "data_erp_train: ", erp_train.shape, "\n",
            "data_erp_test: ", erp_test.shape, "\n",
            
            "data_ssvep_train: ", ssvep_train.shape, "\n",
            "data_ssvep_test: ", ssvep_test.shape, "\n",
            )

      self.train_data = np.concatenate((mi_train, erp_train, ssvep_train), axis = 0)
      self.train_index = np.concatenate((mi_labels_train, erp_labels_train, ssvep_labels_train), axis = 0)
      self.train_index_paradigm = [0] * len(mi_train) + [1] * len(erp_train) + [2] * len(ssvep_train)

      test_data = np.concatenate((mi_test, erp_test, ssvep_test), axis = 0)
      test_index = np.concatenate((mi_labels_test, erp_labels_test, ssvep_labels_test), axis = 0)
      test_index_paradigm = [0] * len(mi_test) + [1] * len(erp_test) + [2] * len(ssvep_test)

      if concatenate_train_test:
        print("concat!!!!!!")
        self.train_data = np.concatenate((self.train_data, test_data),)
        self.train_index = np.concatenate((self.train_index, test_index),)
        self.train_index_paradigm = np.concatenate((self.train_index_paradigm, test_index_paradigm),)
        return
      elif typo == "train":
          print("train side!!!!")
          return 
      elif typo == "test":
          print("test side!!!!")
          self.train_data = test_data
          self.train_index = test_index
          self.train_index_paradigm = test_index_paradigm

    def __len__(self) :
        return len(self.train_data)

    def __getitem__(self, idx) :
        x, label_paradigm, label_ = self.train_data[idx], self.train_index_paradigm[idx], self.train_index[idx]
        return x, label_paradigm, label_



class BCI_pl_dataset(pl.LightningDataModule):
  def __init__(self, hparams):
    super().__init__()
    self.hparams = hparams

  def prepare_data(self):
    #here you download your dataset, from some site for example
    #or pytorch torchaudio etc.
    #or call torch.utils.data.Dataset type
    
    print("gg")
    

  def setup(self): 
    path_MI = os.path.join(self.hparams.path_to_files, self.hparams.path_mi)
    path_ERP =  os.path.join(self.hparams.path_to_files, self.hparams.path_erp)
    path_SSVEP =  os.path.join(self.hparams.path_to_files, self.hparams.path_ssvep)

    if not self.hparams.separate_test:
        print("MERGING !!!!!!!!!")
        dataset = BCI_Dataset(path_MI, path_ERP, path_SSVEP, self.hparams.resample, self.hparams.concatenate_train_test, self.hparams.decrease_erp, None)
        len_t = int(len(dataset) * 0.8)
        len_v = int((len(dataset) - len_t)/2)
        len_test = len(dataset) - len_t - len_v
        self.dataset_train, self.dataset_val, self.dataset_test = torch.utils.data.random_split(dataset, [len_t, len_v, len_test], generator=torch.Generator().manual_seed(42))
    else:
        print("deviding train/test !!!!!!!!!!!!")
        dataset_train = BCI_Dataset(path_MI, path_ERP, path_SSVEP, self.hparams.resample, False, self.hparams.decrease_erp, typo = "train")
        self.dataset_test = BCI_Dataset(path_MI, path_ERP, path_SSVEP, self.hparams.resample, False, self.hparams.decrease_erp, typo = "test")
        len_t = int(len(dataset_train) * 0.9)
        len_v = len(dataset_train) - len_t
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(dataset_train, [len_t, len_v], generator=torch.Generator().manual_seed(42))
    

  def train_dataloader(self):
    train_loader = DataLoader(dataset=self.dataset_train,
                                batch_size=self.hparams.batch_size,
                                shuffle=True,
                                num_workers = self.hparams.num_workers)
    return train_loader

  def val_dataloader(self):
    val_loader = DataLoader(dataset=self.dataset_val,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers = self.hparams.num_workers)
    return val_loader

  def test_dataloader(self):
    test_loader = DataLoader(dataset=self.dataset_test,
                                batch_size=self.hparams.batch_size,
                                shuffle=False,
                                num_workers = self.hparams.num_workers)
    return test_loader