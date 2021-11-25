import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import  numpy

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Linear, self).__init__()
        self.mlp = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.

    :param in_c: int, number of input channels.
    :param out_c: int, number of output channels.
    :param K: int, the order of Chebyshev Polynomial.
    """
    def __init__(self, in_c, out_c, K=2, drop_rate=0,bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.drop_rate=drop_rate
        # self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c,
        # ]
        # init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_c, 1, 1))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1
        self.linear = Linear(in_c, out_c)

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        # batch,seqlen,number,feature=inputs.size()

        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L)  # [K, N, N]
        results=[]
        for i in range(self.K):
            result = torch.einsum('mn,bsni->bsmi', mul_L[i], inputs).contiguous()
            results.append(result.unsqueeze(0))
        results=torch.cat(results, dim=0)#[K,batch,seqlen,number,feature]
        results = torch.sum(results, dim=0) #[batch,seqlen,number,feature]
        x_gc = self.linear(results)
        output = F.dropout(x_gc, self.drop_rate, training=self.training)+self.bias#[batch,seqlen,number,feature]
        return output

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:

            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, K):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c, K=K)
        self.conv2 = ChebConv(in_c=hid_c, out_c=out_c, K=K)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]

        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_2 = self.act(self.conv2(output_1, graph_data))

        return output_2.unsqueeze(2)

class DiffGraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, drop_rate, adj_len, orders=2, enable_bias=True):
        super(DiffGraphConv, self).__init__()
        self.in_channels = in_channels*(1+orders*adj_len)
        self.out_channels = out_channels
        self.enable_bias = enable_bias
        self.orders = orders
        self.drop_rate = drop_rate

        self.linear = Linear(self.in_channels, out_channels)

    def forward(self, x, adj_mats):
        output = [x]
        for adj in adj_mats:
            x_mul = torch.einsum('mn,bsni->bsmi', adj, x).contiguous()
            output.append(x_mul)
            for k in range(2, self.orders + 1):
                x_mul_k = torch.einsum('mn,bsni->bsmi', adj, x_mul).contiguous()
                output.append(x_mul_k)
                x_mul = x_mul_k

        x_gc = self.linear(torch.cat(output, dim=1))
        output = F.dropout(x_gc, self.drop_rate, training=self.training)
        return output


class OutputLayer(nn.Module):
    def __init__(self, skip_channels, end_channels, out_channels):
        super(OutputLayer, self).__init__()

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, (1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, out_channels, (1, 1), bias=True)

    def forward(self, x):
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x