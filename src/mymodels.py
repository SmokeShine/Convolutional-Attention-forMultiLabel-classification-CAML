import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from math import floor
from torch.nn.init import xavier_uniform
from torch.nn.init import xavier_uniform_
# https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim


class CNNAttn(nn.Module):
    def __init__(self, num_output_categories, weights_matrix, num_filters=50, kernel_size=10):
        super(CNNAttn, self).__init__()
        self.embeddings, num_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, non_trainable=True)
		# https://discuss.pytorch.org/t/how-to-keep-the-shapebaias-of-input-and-output-same-when-dilation-conv/14338
        padding=int(floor(kernel_size / 2))
        self.conv = nn.Conv1d(embedding_dim, num_filters,kernel_size, padding=padding)
        # removing bias for simple matrix multiplication
        self.upscaling = nn.Linear(num_filters, num_output_categories,bias=False)
        self.output = nn.Linear(num_filters, num_output_categories)

    def forward(self, input_tuple):
        seqs, lengths = input_tuple
        # [16, 2459, 1]
        
        x = self.embeddings(seqs.squeeze(2))
		# 1000 should never be reduced
        # [16, 1000,7]
        x = x.permute(0, 2, 1)
        # [16, 7, 1000]
        x = self.conv(x)
        # [16,50,1000]
        x = x.permute(0, 2, 1)
        # [16, 1000, 50
        x = torch.tanh(x)
        # [16, 1000, 50]
        # scale to all categories
        # https://stackoverflow.com/questions/61292150/breaking-down-a-batch-in-pytorch-leads-to-different-results-why
        # there is a slight numeric change in 5th floating point.
        x_branch = self.upscaling(x)
        x_branch = x_branch.permute(0,2,1)
        # tensor(156626.1562, device='cuda:0', grad_fn=<SumBackward0>)
        # [16, 8929, 1000]
        # calculating alpha
        
        alpha = torch.softmax(x_branch, dim=2)
        # (Pdb) alpha.size()
        # torch.Size([16, 8929, 1000])

		# torch.Size([16, 1000, 8929])
		# alpha.sum(axis=0)[0]
        # alpha.sum(axis=1)[0]
        # alpha.sum(axis=2)[0] 
        # Pdb) alpha.sum(axis=2)[0]
        # tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0
        #axis 2 has all 1.
        # axis 2 is lossy representation of n gram.
        # weighted sum on n grams
        # (Pdb) alpha.size()
        # torch.Size([16, 8929, 1000])
        # (Pdb) x.size()
        # torch.Size([16, 1000, 50])
        modified_n_gram = alpha @ x
        # torch.Size([16, 8929, 50])
        # back to original n graph but with weights
        # https://pytorch.org/docs/stable/generated/torch.mul.html
        # https://stackoverflow.com/questions/51980654/pytorch-element-wise-filter-layer
        weighted_output = (self.output.weight*modified_n_gram)
        # torch.Size([16, 8929, 50])
        # This can also be bypassed, as we already have the shape. 
        # Keeping it for added model capacity
        weighted_sum=weighted_output.sum(dim=2)
        # torch.Size([16, 8929])
		# weighted_sum.sum(axis=0)
        # weighted_sum.sum(axis=1)
        # weighted_sum.sum(axis=2)
        return weighted_sum


class testGRU(nn.Module):
    def __init__(self, weights_matrix, num_categories):
        # initialized everything from nn.module
        super(testGRU, self).__init__()

        self.embeddings, num_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, non_trainable=True)
        self.GRUCell = nn.GRU(input_size=weights_matrix.size()[
                              1], hidden_size=32, num_layers=1, batch_first=True, bidirectional=False)
        self.FullyConnectedOutput = nn.Linear(
            in_features=32, out_features=num_categories)

    def forward(self, input_tuple):
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-3-cnn-model-7bb30712abd7
        # https://stackoverflow.com/questions/47205762/embedding-3d-data-in-pytorch
        # similar to our data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seqs, lengths = input_tuple
        # torch.Size([2, 1688, 1])
        # 2 batches. each batch contains a matrix of size (1688,1)
        # may be incorrect - how did this work in embedding layer
        x = self.embeddings(seqs.squeeze(2))
        # torch.Size([2, 1688, 1, 7])

        # torch.Size([2, 1688, 7])
        # https://github.com/pytorch/pytorch/issues/43227
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True)
        x, hn = self.GRUCell(pack)
        x_unpacked, x_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        x_unpacked_rearranged = torch.zeros(
            len(x_unpacked), x_unpacked.shape[-1])

        for i, np_array in enumerate(x_unpacked):
            temp = np_array[int(lengths[i])-1, :]
            x_unpacked_rearranged[i] = temp
        # RuntimeError: Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for addmm)
        x_unpacked_rearranged = x_unpacked_rearranged.to(device)
        x = self.FullyConnectedOutput(x_unpacked_rearranged)

        return x

class testLSTM(nn.Module):
    def __init__(self, weights_matrix, num_categories):
        # initialized everything from nn.module
        super(testLSTM, self).__init__()

        self.embeddings, num_embeddings, embedding_dim = create_emb_layer(
            weights_matrix, non_trainable=True)
        self.LSTMCell = nn.LSTM(input_size=weights_matrix.size()[
                              1], hidden_size=32, num_layers=1, batch_first=True, bidirectional=False)
        self.FullyConnectedOutput = nn.Linear(
            in_features=32, out_features=num_categories)

    def forward(self, input_tuple):
        # https://galhever.medium.com/sentiment-analysis-with-pytorch-part-3-cnn-model-7bb30712abd7
        # https://stackoverflow.com/questions/47205762/embedding-3d-data-in-pytorch
        # similar to our data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seqs, lengths = input_tuple
        # torch.Size([2, 1688, 1])
        # 2 batches. each batch contains a matrix of size (1688,1)
        # may be incorrect - how did this work in embedding layer
        x = self.embeddings(seqs.squeeze(2))
        # torch.Size([2, 1688, 1, 7])

        # torch.Size([2, 1688, 7])
        # https://github.com/pytorch/pytorch/issues/43227
        pack = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True)
        x, hn = self.LSTMCell(pack)
        x_unpacked, x_unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)
        x_unpacked_rearranged = torch.zeros(
            len(x_unpacked), x_unpacked.shape[-1])

        for i, np_array in enumerate(x_unpacked):
            temp = np_array[int(lengths[i])-1, :]
            x_unpacked_rearranged[i] = temp
        # RuntimeError: Tensor for argument #2 'mat1' is on CPU, but expected it to be on GPU (while checking arguments for addmm)
        x_unpacked_rearranged = x_unpacked_rearranged.to(device)
        x = self.FullyConnectedOutput(x_unpacked_rearranged)

        return x