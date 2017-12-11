# San-Pytorch
Let us try implementing SAN in pytorch from scratch
The original implementation of Stacked Attention Networks for Image Question Answering by Zichao Yang was in Theano . During my winter intern at IIT-K on VQA let me first implement this in pytorch and then begin my work .
Kindly refer to the [paper](https://arxiv.org/abs/1511.02274) and the original [theano](https://github.com/zcyang/imageqa-san) code before proceeding .
You could also refer to the [torch](https://github.com/JamesChuanggg/san-torch) implementation of SAN .
![image](/VQA.png)

Requirements :
The code is written in Python and requires [PyTorch](http://pytorch.org/). The preprocssinng code is in Python and Lua, and you need to install NLTK if you want to use NLTK to tokenize the question.

     

Download Dataset

We simply follow the steps provide by [HieCoAttenVQA](https://github.com/jiasenlu/HieCoAttenVQA) to prepare VQA data. The first thing you need to do is to download the data and do some preprocessing. Head over to the data/ folder and run

$ python vqa_preprocessing.py --download 1 --split 1

--download Ture means you choose to download the VQA data from the VQA website and --split 1 means you use COCO train set to train and validation set to evaluation. --split 2 means you use COCO train+val set to train and test set to evaluate. After this step, it will generate two files under the data folder. vqa_raw_train.json and vqa_raw_test.json

Download Image Model

We are using [VGG_ILSVRC_19_layers model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77) and [Deep Residual network implement model](https://github.com/facebook/fb.resnet.torch) by Facebook .

Generate Image/Question Features

Head over to the prepro folder and run

$ th prepro_img_vgg.lua -input_json ../data/vqa_data_prepro.json -image_root ../../coco/images/ -cnn_proto ../image_model/VGG_ILSVRC_19_layers_deploy.prototxt -cnn_model ../image_model/VGG_ILSVRC_19_layers.caffemodel

Before running this make sure you create a new folder called image_model and put the downloaded VGG caffe model ( slong with .prototxt file )and Deep residual net in that folder . For the image root give the path of coco dataset in your system .
You can change the -gpuid, -backend and -batch_size based on your gpu.

Train the model

We have everything ready to train the VQA. Back to the main folder and execute :
python train.py --use_gpu <0 or 1> --batch_size <batch size> --epochs <no. of epochs>
  
you can also change many other options. For a list of all options, see train.py

Evaluate the model :

In main folder run :
python eval.py --use_gpu <0 or 1>
you can also change many other options. For a list of all options, see eval.py

