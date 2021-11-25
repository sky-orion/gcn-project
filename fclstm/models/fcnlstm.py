import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from models import layer
import math

class FCLSTM(nn.Module):
    def __init__(self,
                 cuda="cuda:0",
                 window=120,
                 feature=17,
                 hidRNN=100,
                 hidden_state_features=120,
                 outfeature=1,
                 hidSkip=5,
                 CNN_kernel=1,
                 skip=24,
                 highway_window=24,
                 num_layers_lstm=1,
                 dropout=0.5,
                 output_fun="sigmoid",
                 batch_size=1):
        super(FCLSTM, self).__init__()
        # self.use_cuda = args.cuda
        self.window_length = window;  # window, read about it after understanding the flow of the code...What is window size? --- temporal window size (default 24 hours * 7)
        self.original_columns = feature  # the number of columns or features
        self.hidR = hidRNN;
        self.hidden_state_features = hidden_state_features
        self.hidC = outfeature;
        self.hidS = hidSkip;
        self.Ck = CNN_kernel;  # the kernel size of the CNN layers
        self.skip = skip;
        self.pt = (self.window_length - self.Ck) // self.skip
        self.hw = highway_window
        self.num_layers_lstm = num_layers_lstm
        self.lstm = nn.LSTM(input_size=self.original_columns, hidden_size=self.hidden_state_features,
                            num_layers=self.num_layers_lstm,
                            bidirectional=False);
        self.compute_convolution = nn.Conv2d(1, self.hidC, kernel_size=(
            self.Ck, self.hidden_state_features))  # hidC are the num of filters, default value of Ck is one
        self.attention_matrix = nn.Parameter(
            torch.ones(batch_size, self.hidC, self.hidden_state_features, requires_grad=True))
        self.context_vector_matrix = nn.Parameter(
            torch.ones(batch_size, self.hidden_state_features, self.hidC, requires_grad=True))
        self.final_state_matrix = nn.Parameter(
            torch.ones(batch_size, self.hidden_state_features, self.hidden_state_features, requires_grad=True))
        self.final_matrix = nn.Parameter(
            torch.ones(batch_size, self.original_columns, self.hidden_state_features, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.attention_matrix)
        torch.nn.init.xavier_uniform_(self.context_vector_matrix)
        torch.nn.init.xavier_uniform_(self.final_state_matrix)
        torch.nn.init.xavier_uniform_(self.final_matrix)
        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.original_columns));  # kernel size is size for the filters
        self.GRU1 = nn.GRU(self.hidC, self.hidR);
        self.dropout = nn.Dropout(p=dropout);
        if (self.skip > 0):
            self.GRUskip = nn.GRU(self.hidC, self.hidS);
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.original_columns);
        else:
            self.linear1 = nn.Linear(self.hidR, self.original_columns);
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1);
        self.output = None;
        if (output_fun == 'sigmoid'):
            self.output = torch.sigmoid;
        if (output_fun == 'tanh'):
            self.output = torch.tanh;

    def forward(self, input):
        batch_size = input.size(0);
        # if (self.use_cuda):
        #     x = input.cuda()
        x=input
        """
           Step 1. First step is to feed this information to LSTM and find out the hidden states 

            General info about LSTM:

            Inputs: input, (h_0, c_0)
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence.
          The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - **c_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial cell state for each element in the batch.

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.


    Outputs: output, (h_n, c_n)
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features `(h_t)` from the last layer of the LSTM,
          for each t. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.

          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.
          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`.

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)`` and similarly for *c_n*.
        - **c_n** (num_layers * num_directions, batch, hidden_size): tensor
          containing the cell state for `t = seq_len`

        """
        input_to_lstm = x.permute(1, 0, 2).contiguous()  # input to lstm is of shape (seq_len, batch, input_size) (x shape (batch_size, seq_length, features))
        lstm_hidden_states, (h_all, c_all) = self.lstm(input_to_lstm)
        hn = h_all[-1].view(1, h_all.size(1), h_all.size(2))

        """
            Step 2. Apply convolution on these hidden states. As in the paper TPA-LSTM, these filters are applied on the rows of the hidden state
        """
        output_realigned = lstm_hidden_states.permute(1, 0, 2).contiguous()
        # print(output_realigned.size())
        hn = hn.permute(1, 0, 2).contiguous()
        # cn = cn.permute(1, 0, 2).contiguous()
        input_to_convolution_layer = output_realigned.view(-1, 1, self.window_length, self.hidden_state_features);
        convolution_output = F.relu(self.compute_convolution(input_to_convolution_layer));
        convolution_output = self.dropout(convolution_output);
        # print(convolution_output.size())

        """
            Step 3. Apply attention on this convolution_output
        """
        convolution_output = convolution_output.squeeze(3)

        """
                In the next 10 lines, padding is done to make all the batch sizes as the same so that they do not pose any problem while matrix multiplication
                padding is necessary to make all batches of equal size
        """
        final_hn = torch.zeros(self.attention_matrix.size(0), 1, self.hidden_state_features)
        final_convolution_output = torch.zeros(self.attention_matrix.size(0), self.hidC, self.window_length)
        diff = 0
        if (hn.size(0) < self.attention_matrix.size(0)):
            final_hn[:hn.size(0), :, :] = hn
            final_convolution_output[:convolution_output.size(0), :, :] = convolution_output
            diff = self.attention_matrix.size(0) - hn.size(0)
        else:
            final_hn = hn
            final_convolution_output = convolution_output

        """
           final_hn, final_convolution_output are the matrices to be used from here on
        """
        convolution_output_for_scoring = final_convolution_output.permute(0, 2, 1).contiguous()
        final_hn_realigned = final_hn.permute(0, 2, 1).contiguous()
        convolution_output_for_scoring = convolution_output_for_scoring
        print(convolution_output_for_scoring.size())
        # final_hn_realigned = final_hn_realigned
        # mat1 = torch.bmm(convolution_output_for_scoring, self.attention_matrix)
        # print(mat1.size())
        # scoring_function = torch.bmm(mat1, final_hn_realigned)
        # alpha = torch.nn.functional.sigmoid(scoring_function)
        # context_vector = alpha * convolution_output_for_scoring
        # context_vector = torch.sum(context_vector, dim=1)
        #
        # """
        #    Step 4. Compute the output based upon final_hn_realigned, context_vector
        # """
        # context_vector = context_vector.view(-1, self.hidC, 1)
        # h_intermediate = torch.bmm(self.final_state_matrix, final_hn_realigned) + torch.bmm(self.context_vector_matrix, context_vector)
        # result = torch.bmm(self.final_matrix, h_intermediate)
        # # print(result.size())
        # result = result.permute(0, 2, 1).contiguous()
        # # print(result.size())
        # result = result.squeeze()
        #
        # """
        #    Remove from result the extra result points which were added as a result of padding
        # """
        # final_result = result[:result.size(0) - diff]
        # # print(final_result.size())
        # """
        # Adding highway network to it
        # """
        #
        # if (self.hw > 0):
        #     z = x[:, -self.hw:, :];
        #     z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
        #     z = self.highway(z);
        #     z = z.view(-1, self.original_columns);
        #     res = final_result + z;

        return torch.sigmoid(convolution_output_for_scoring)

class FCN_model(nn.Module):

    def __init__(self, NumClassesOut, N_time, N_Features, N_LSTM_Out=128, N_LSTM_layers=1
                 ,Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP = 0.8, FC_DropP = 0.3):
        super(FCN_model, self).__init__()
        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(self.N_Features, self.N_LSTM_Out, self.N_LSTM_layers)
        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 8)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 5)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(self.Conv3_NF + self.N_LSTM_Out, self.NumClassesOut)


    def init_hidden(self):
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out)
        return h0, c0


    def forward(self, x):
        # input x should be in size [B,T,F] , where B = Batch size
        #                                         T = Time sampels
        #                                         F = features

        h0, c0 = self.init_hidden()
        x1, (ht, ct) = self.lstm(x, (h0, c0))
        # for _ in range(x1.size(1)):
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)
        x_all = torch.cat((x1, x2), dim=1)
        print(x1.size())
        print(x2.size(),x_all.size())
        x_out = self.FC(x_all)
        return x_out
# in1 = torch.zeros((32,120,17)).cuda()
# lstm=FCLSTM()
# lstm=lstm.to("cuda:0")
# out=lstm(in1)
# print(out.size())