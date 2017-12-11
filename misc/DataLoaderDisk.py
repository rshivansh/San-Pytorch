import h5py
import torch
import numpy as np
from utils import getopt, read_json, count_key


class DataLoader(): # Extend PyTorch's Dataset class
    def __init__(self, opt):
        if opt['h5_img_file_train']:
            print('DataLoader loading h5 image train file: ' + opt['h5_img_file_train'])
            self.h5_img_file_train = h5py.File(opt['h5_img_file_train'], 'r')

        if opt['h5_img_file_test']:
            print('DataLoader loading h5 image test file: ' + opt['h5_img_file_test'])
            self.h5_img_file_test = h5py.File(opt['h5_img_file_test'], 'r')

        print('DataLoader loading h5 question file: ' + opt['h5_ques_file'])
        h5_file = h5py.File(opt['h5_ques_file'], 'r')
        self.ques_train = h5_file['/ques_train']
        self.ques_len_train = h5_file['/ques_len_train']
        self.img_pos_train = h5_file['/img_pos_train']
        self.ques_id_train = h5_file['/ques_id_train']
        self.ans_train = h5_file['/answers']
        self.split_train = h5_file['/split_train']

        self.ques_test = h5_file['/ques_test']
        self.ques_len_test = h5_file['/ques_len_test']
        self.img_pos_test = h5_file['/img_pos_test']
        self.ques_id_test = h5_file['/ques_id_test']
        self.ans_test = h5_file['/ans_test']
        self.split_test = h5_file['/split_test']

        h5_file.close()

        print('DataLoader loading json file: ', opt['json_file'])
        json_file = read_json(opt['json_file'])
        self.ix_to_word = json_file['ix_to_word']
        self.ix_to_ans = json_file['ix_to_ans']
        self.feature_type = opt['feature_type']

        # ------------------------------------------------------------------------------------------------------------------
        # need to double check this
        self.seq_length = self.ques_train.shape[1]
        # ------------------------------------------------------------------------------------------------------------------

        # count the vocabulary key!
        self.vocab_size = count_key(self['ix_to_word'])

        self.split_ix = {}
        self.iterators = {}

        for i in range(0, self.split_train.shape[0]):
            idx = self.split_train[i]
            idx = str(idx)
            if not self.split_ix[idx]:
                self.split_ix[idx] = []
                self.iterators[idx] = 0
            self.split_ix[idx].append(i)

        for i in range(0, self.split_test.shape[0]):
            idx = self.split_test[i]
            idx = str(idx)
            if not self.split_ix[idx]:
                self.split_ix[idx] = []
                self.iterators[idx] = 0
            self.split_ix[idx].append(i)

        for key, value in self.split_ix.iteritems():
            print("Assigned %d images to split %s", len(value), key)

    def resetIterator(split):
        self.iterators[str(split)] = 0

    def getVocabSize():
        return self.vocab_size

    def getSeqLength():
        return self.seq_length

    def getDataNum(split):
        return len(self.split_ix[str(split)])

    def getBatch(opt):
        split = getopt(opt, 'split')
        split = str(split)
        batch_zie = getopt(opt, 'batch_size', 128)

        split_ix_tmp = self.split_ix[split]
        assert(split_ix_tmp, 'split ' + str(split) + ' not found')

        max_index = len(split_ix_tmp) - 1
        ques_idx = torch.LongTensor(batch_size)
        img_idx = torch.LongTensor(batch_size)

        if self.feature_type == 'VGG':
            self.img_batch = torch.Tensor(batch_size, 14, 14, 512)
        elif self.feature_type == 'Residual':
            self.img_batch = torch.Tensor(batch_size, 14, 14, 2048)


        for i in range(0, batch_size):
            ri = self.iterators[split]
            ri_next = ri + 1
            if ri_next > max_index:
                ri_next = 1
            self.iterators[split] = ri_next
            if int(split) == 0:
                ix = split_ix_tmp[torch.randperm(max_index + 1)[0]]
            else:
                ix = split_ix_tmp[ri]

            assert(ix != None, 'Bug: split ' + split + ' was accessed out of bounds with ' + str(ri))
            ques_idx[i] = ix
            if int(split) == 0 or int(split) == 1:
                img_idx[i] = self.img_pos_train[ix]
                if self.h5_img_file_train != None:
                    if self.feature_type == 'VGG':
                        img = self.h5_img_file_train['/images_train'][img_idx[i]-1:img_idx[i], 0:14, 0:14, 0:512]
                        self.img_batch[i] = img
                    elif self.feature_type == 'Residual':
                        img = self.h5_img_file_train['/images_train'][img_idx[i]-1:img_idx[i], 0:14, 0:14, 0:2048]
                        self.img_batch[i] = img
                    else:
                        print("Error(train): feature type error")
            else:
                img_idx[i] = self.img_pos_test[ix]
                if self.h5_img_file_test != None:
                    if self.feature_type == 'VGG':
                        img = self.h5_img_file_test['/images_test'][img_idx[i]-1:img_idx[i], 0:14, 0:14, 0:512]
                        self.img_batch[i] = img
                    elif self.feature_type == 'Residual':
                        img = self.h5_img_file_test['/images_test'][img_idx[i]-1:img_idx[i], 0:14, 0:14, 0:2048]
                        self.img_batch[i] = img
                    else:
                        print("Error(test): feature type error")

        data = {}
        data['questions'] = []
        data['ques_id'] = []
        data['ques_len'] = []
        data['answer'] = []
        if int(split) == 0 or int(split) == 1:
            data['images'] = np.reshape(self.img_batch, (batch_size, 196, -1))
            for i in range(0, len(ques_idx)):
                data['questions'].append(self.ques_train[ques_idx[i]])
                data['ques_id'].append(self.ques_id_train[ques_idx[i]])
                data['ques_len'].append(self.ques_len_train[ques_idx[i]])
                data['answer'].append(self.ans_train[ques_idx[i]])
        else:
            data['images'] = np.reshape(self.img_batch, (batch_size, 196, -1))
            for i in range(0, len(ques_idx)):
                data['questions'].append(self.ques_test[ques_idx[i]])
                data['ques_id'].append(self.ques_id_test[ques_idx[i]])
                data['ques_len'].append(self.ques_len_test[ques_idx[i]])
                data['answer'].append(self.ans_test[ques_idx[i]])

    return data
