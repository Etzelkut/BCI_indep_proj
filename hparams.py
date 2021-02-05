from argparse import Namespace

re_dict = {
    "path_to_files": "/content/data",
    "batch_size": 64,
    "num_workers": 4, 
    #'pin_memory': True,
    
    "path_mi": "sess01_subj02_EEG_MI.mat",
    "path_erp": "sess01_subj02_EEG_ERP.mat",
    "path_ssvep" : "sess01_subj02_EEG_SSVEP.mat",
    
    "resample": 100, 
    "concatenate_train_test": True,
    "decrease_erp": True,

    "pe_max_len": 405,
    #
    "num_classes": 3, 
    #
    "d_model_emb": 62,
    "d_ff": 248,
    "heads": 2,
    "dropout": 0.05,
    "encoder_number": 5,
    #!
    "local_heads": 2, #more that 0 enable local heads
    "add_sch": False,
    #!
    #
    "local_window_size": 256,
    "attention_type": "selfatt", #performer, selfatt, linear
    "feedforward_type": "glu", # classic, glu
    #
    "learning_rate": 3e-4,
    "epochs": 150, 
    #
    "separate_test": True, #also automatically disable concatenate_train_test

}

hyperparams = Namespace(**re_dict)
