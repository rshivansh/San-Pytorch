import argparse
import numpy as np
import json
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from misc.DataLoader import CDATA
from misc.img_emb_net import ImageEmbedding
from misc.ques_emb_net import QuestionEmbedding
from misc.san import Attention


def adjust_learning_rate(optimizer, epoch, lr, learning_rate_decay_every):
    # Sets the learning rate to the initial LR decayed by 10 every learning_rate_decay_every epochs
    lr_tmp = lr * (0.5 ** (epoch // learning_rate_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_tmp
    return lr_tmp


def main(params):
    # Construct Data loader
    opt = {
            'feature_type': params['feature_type'],
            'h5_img_file' : params['input_img_train_h5'],
            'h5_ques_file': params['input_ques_h5'],
            'json_file'   : params['input_json']
            }
    train_dataset = CDATA(opt, train=True, quiet=( not params['print_params']))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=params['batch_size'],
                                               shuffle=False)

    # Construct NN models
    vocab_size = train_dataset.getVocabSize()
    question_model = QuestionEmbedding(vocab_size, params['emb_size'],
                                       params['hidden_size'], params['rnn_size'],
                                       params['rnn_layers'], params['dropout'],
                                       train_dataset.getSeqLength(), params['use_gpu'])

    image_model = ImageEmbedding(params['hidden_size'], params['feature_type'])

    attention_model = Attention(params['hidden_size'], params['att_size'],
                                params['img_seq_size'], params['output_size'],
                                params['dropout'])

    if params['use_gpu'] and torch.cuda.is_available():
        question_model.cuda()
        image_model.cuda()
        attention_model.cuda()

    if params['resume_from_epoch'] > 1:
        load_model_dir = os.path.join(params['checkpoint_path'], str(params['resume_from_epoch']-1))
        print('Loading model files from folder: %s' % load_model_dir)
        question_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'question_model.pkl')))
        image_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'image_model.pkl')))
        attention_model.load_state_dict(torch.load(
            os.path.join(load_model_dir, 'attention_model.pkl')))

    # Loss and optimizers
    criterion = nn.CrossEntropyLoss()

    optimizer_parameter_group = [
            {'params': question_model.parameters()},
            {'params': image_model.parameters()},
            {'params': attention_model.parameters()}
            ]
    if params['optim'] == 'sgd':
        optimizer = torch.optim.SGD(optimizer_parameter_group,
                                    lr=params['learning_rate'],
                                    momentum=params['momentum'])
    elif params['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(optimizer_parameter_group,
                                        lr=params['learning_rate'],
                                        alpha=params['optim_alpha'],
                                        eps=params['optim_epsilon'],
                                        momentum=params['momentum'])
    elif params['optim'] == 'adam':
	optimizer = torch.optim.Adam(optimizer_parameter_group,
				     eps=params['optim_epsilon'],
				     lr=params['learning_rate'])
    elif params['optim'] == 'rprop':
	optimizer = torch.optim.Rprop(optimizer_parameter_group,
                                     lr=params['learning_rate'])
    else:
        print('Unsupported optimizer: \'%s\'' % (params['optim']))
        return None

    # Start training
    all_loss_store = []
    loss_store = []
    lr_cur = params['learning_rate']
    for epoch in range(params['resume_from_epoch'], params['epochs']+1):

        if epoch > params['learning_rate_decay_start']:
            lr_cur = adjust_learning_rate(optimizer, epoch - 1 - params['learning_rate_decay_start'] + params['learning_rate_decay_every'],
                                          params['learning_rate'], params['learning_rate_decay_every'])
        print('Epoch: %d | lr: %f' % (epoch, lr_cur))

        running_loss = 0.0
        for i, (image, question, ques_len, ans) in enumerate(train_loader):
            image = Variable(image)
            question = Variable(question)
            ans = Variable(ans, requires_grad=False)
            if (params['use_gpu'] and torch.cuda.is_available()):
                image = image.cuda()
                question = question.cuda()
                ans = ans.cuda()

            optimizer.zero_grad()
            img_emb = image_model(image)
            ques_emb = question_model(question, ques_len)
            output = attention_model(ques_emb, img_emb)

            loss = criterion(output, ans)
            #  print('i: %d | LOSS: %.4f | lr: %f'%(i, loss.data[0], lr_cur))
            all_loss_store += [loss.data[0]]
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            if not (i+1) % params['losses_log_every']:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (
                    epoch, params['epochs'], i+1,
                    train_dataset.__len__()//params['batch_size'], loss.data[0]
                    ))

        print("Saving models")
        model_dir = os.path.join(params['checkpoint_path'], str(epoch))
        os.mkdir(model_dir)
        torch.save(question_model.state_dict(), os.path.join(model_dir, 'question_model.pkl'))
        torch.save(image_model.state_dict(), os.path.join(model_dir, 'image_model.pkl'))
        torch.save(attention_model.state_dict(), os.path.join(model_dir, 'attention_model.pkl'))
        loss_store += [running_loss]

        # torch.save(question_model.state_dict(), 'question_model'+str(epoch)+'.pkl')
    print("Saving all losses to file")
    np.savetxt(os.path.join(params['checkpoint_path'], 'all_loss_store.txt'), np.array(all_loss_store), fmt='%f')
    print(loss_store)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_img_train_h5', default='data/vqa_data_img_vgg_train.h5', help='path to the h5file containing the train image feature')
    parser.add_argument('--input_img_test_h5', default='data/vqa_data_img_vgg_test.h5', help='path to the h5file containing the test image feature')
    parser.add_argument('--input_ques_h5', default='data/vqa_data_prepro.h5', help='path to the json file containing additional info and vocab')

    parser.add_argument('--input_json', default='data/vqa_data_prepro.json', help='output json file')
    parser.add_argument('--start_from', default='', help='path to a model checkpoint to initialize model weights from. Empty = don\'t')
    parser.add_argument('--resume_from_epoch', default=1, type=int, help='load model from previous epoch')

    # Options
    parser.add_argument('--feature_type', default='VGG', help='VGG or Residual')
    parser.add_argument('--emb_size', default=500, type=int, help='the size after embeeding from onehot')
    parser.add_argument('--hidden_size', default=1024, type=int, help='the hidden layer size of the model')
    parser.add_argument('--rnn_size', default=1024, type=int, help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--att_size', default=512, type=int, help='size of sttention vector which refer to k in paper')
    parser.add_argument('--batch_size', default=200, type=int, help='what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
    parser.add_argument('--output_size', default=1000, type=int, help='number of output answers')
    parser.add_argument('--rnn_layers', default=2, type=int, help='number of the rnn layer')
    parser.add_argument('--img_seq_size', default=196, type=int, help='number of feature regions in image')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout ratio in network')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs to run')

    # Optimization
    parser.add_argument('--optim', default='rmsprop', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', default=4e-4, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--learning_rate_decay_start', default=10, type=int, help='at what epoch to start decaying learning rate?')
    parser.add_argument('--learning_rate_decay_every', default=10, type=int, help='every how many epoch thereafter to drop LR by 0.1?')
    parser.add_argument('--optim_alpha', default=0.99, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--optim_beta', default=0.995, type=float, help='beta used for adam')
    parser.add_argument('--optim_epsilon', default=1e-8, type=float, help='epsilon that goes into denominator in rmsprop')
    parser.add_argument('--max_iters', default=-1, type=int, help='max number of iterations to run for (-1 = run forever)')
    parser.add_argument('--iterPerEpoch', default=1250, type=int, help=' no. of iterations per epoch')

    # Evaluation/Checkpointing
    parser.add_argument('--save_checkpoint_every', default=500, type=int, help='how often to save a model checkpoint?')
    parser.add_argument('--checkpoint_path', default='train_model/', help='folder to save checkpoints into (empty = this folder)')

    # Visualization
    parser.add_argument('--losses_log_every', default=10, type=int, help='How often do we save losses, for inclusion in the progress dump? (0 = disable)')

    # misc
    parser.add_argument('--use_gpu', default=1, type=int, help='to use gpu or not to use, that is the question')
    parser.add_argument('--id', default='1', help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--backend', default='cudnn', help='nn|cudnn')
    parser.add_argument('--gpuid', default=2, type=int, help='which gpu to use. -1 = use CPU')
    parser.add_argument('--seed', default=1234, type=int, help='random number generator seed to use')
    parser.add_argument('--print_params', default=1, type=int, help='pass 0 to turn off printing input parameters')

    args = parser.parse_args()
    params = vars(args)                     # convert to ordinary dict
    if params['print_params']:
        print('parsed input parameters:')
        print json.dumps(params, indent = 2)
    main(params)
