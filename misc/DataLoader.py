import h5py
import torch
import numpy as np
import misc.utils as utils

class CDATA(torch.utils.data.Dataset): # Extend PyTorch's Dataset class
    def __init__(self, opt, train, transform=None, quiet=False):
        if not quiet:
            print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])
        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        if train:
            if not quiet:
                print('DataLoader loading h5 image train file: ' + opt['h5_img_file'])
            self.h5_img_file = h5py.File(opt['h5_img_file'], 'r')
            self.ques = h5_file['/ques_train']
            self.ques_len = h5_file['/ques_len_train']
            self.img_pos = h5_file['/img_pos_train']
            self.ques_id = h5_file['/ques_id_train']
            self.ans = h5_file['/answers']
            self.split = h5_file['/split_train']
        else:
            if not quiet:
                print('DataLoader loading h5 image test file: ' + opt['h5_img_file'])
            self.h5_img_file = h5py.File(opt['h5_img_file'], 'r')
            self.ques = h5_file['/ques_test']
            self.ques_len = h5_file['/ques_len_test']
            self.img_pos = h5_file['/img_pos_test']
            self.ques_id = h5_file['/ques_id_test']
            self.ans = h5_file['/ans_test']
            self.split = h5_file['/split_test']

        self.feature_type = opt['feature_type']
        self.train = train
        self.transform = transform

        if not quiet:
            print('DataLoader loading json file: %s'% opt['json_file'])
        json_file = utils.read_json(opt['json_file'])
        self.ix_to_word = json_file['ix_to_word']
        self.ix_to_ans = json_file['ix_to_ans']

        self.vocab_size = utils.count_key(self.ix_to_word)
        self.seq_length = self.ques.shape[1]

    def __len__(self):

        return self.split.shape[0]

    def __getitem__(self, idx):

        img_idx = self.img_pos[idx] - 1
        if self.h5_img_file:
            if self.train:
                if self.feature_type == 'VGG':
                    img = self.h5_img_file['/images_train'][img_idx, 0:14, 0:14, 0:512]  # [14, 14, 512]
                elif self.feature_type == 'Residual':
                    img = self.h5_img_file['/images_train'][img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
                else:
                    print("Error(train): feature type error")
            else:
                if self.feature_type == 'VGG':
                    img = self.h5_img_file['/images_test'][img_idx, 0:14, 0:14, 0:512] # [14, 14, 512]
                elif self.feature_type == 'Residual':
                    img = self.h5_img_file['/images_test'][img_idx, 0:14, 0:14, 0:2048] # [14, 14, 2048]
                else:
                    print("Error(test): feature type error")

        question = np.array(self.ques[idx], dtype=np.int32)                 # vector of size 26
        ques_len = self.ques_len[idx].astype(int) # scalar integer
        answer = self.ans[idx].astype(int) - 1    # scalar integer
        if self.transform is not None:
            img = self.transform(img)
            question = self.transform(question)

        return (img, question, ques_len, answer)

    def getVocabSize(self):
        return self.vocab_size

    def getSeqLength(self):
    return self.seq_length
