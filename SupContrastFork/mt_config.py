import torch


class MT_Config:
    """                                                                                                                 
    Model configuration                                                                                                 
    """

    def __init__(self):
        self.status = "train"
        self.DATA_ROOT = "data"
        #self.MAXLEN = 1002
        self.MAXLEN = 2000
        #         self.short_AMINO='ARNDCQEGHILKMFPSTWYV'
        # self.short_AMINO = "ARNDCQEGHILKMFPSTWYVUOX"
        self.short_AMINO = "ACDEFGHIKLMNPQRSTVWYX"
        self.train_val_split = 0.8
        #     FUNCTION = 'mf'
        # save_model_path = 'save_model_path/'
        self.save_model_path = "saved_model/"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ResNet_out_channels = 128
        self.ResNet_kernel_size = 3
        self.ResNet_n_layers = 4
        self.use_gru = False
        self.emb_dropout = 0.1
        self.lr = 0.005

        self.LSTM_n_hidden_state = [64, 64, 64]

        self.lstm_dropout = 0
        self.n_lstm_layers = 1
        self.activation = "sigmoid"

        # self.epochs = 25
        self.epochs = 1
        #self.epochs = 2
        self.batch_size = 128
        self.workers = 4
        self.weight_decay = 0
        self.decay_epoch = 5
        self.improvement_epoch = 10
        self.print_freq = 100
        self.checkpoint = None
        self.best_model = None
        self.grad_clip = True
