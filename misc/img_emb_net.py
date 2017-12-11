import torch.nn as nn
from torch.autograd import Variable


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, feature_type='VGG'):
        super(ImageEmbedding, self).__init__() # Must call super __init__()

        if feature_type == 'VGG':
            self.img_features = 512
        elif feature_type == 'Residual':
            self.img_features = 2048
        else:
            print('Unsupported feature type: \'{}\''.format(feature_type))
            return None

        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.img_features, self.hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        # input: [batch_size, 14, 14, 512]

        intermed = self.linear(input.view(-1,self.img_features)).view(
                                    -1, 196, self.hidden_size)
    return self.dropout(self.tanh(intermed))
