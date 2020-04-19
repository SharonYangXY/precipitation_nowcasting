from torch import nn
import torch
from nowcasting.utils import make_layers
import math
import torch.nn.functional as F
from experiments.config import cfg


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.3, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output



class Encoder(nn.Module):
    def __init__(self, subnets, rnns):
        super().__init__()
        assert len(subnets)==len(rnns)

        self.blocks = len(subnets)

        for index, (params, rnn) in enumerate(zip(subnets, rnns), 1):
            setattr(self, 'stage'+str(index), make_layers(params))
            setattr(self, 'rnn'+str(index), rnn)

        # init the GCN.
        self.gcn1 = GraphConvolution(8, 8)
        self.gcn2 = GraphConvolution(192, 192)
        self.gcn3 = GraphConvolution(192, 192)
        
#         self.fi_1 = nn.Parameter(torch.FloatTensor(8, 8)) #.to(cfg.GLOBAL.DEVICE)
#         self.fi_2 = nn.Parameter(torch.FloatTensor(192, 192))#.to(cfg.GLOBAL.DEVICE)
#         self.fi_3 = nn.Parameter(torch.FloatTensor(192, 192))#.to(cfg.GLOBAL.DEVICE)
        
#         self.fi_1_a = nn.Parameter(torch.FloatTensor(8, 8))#.to(cfg.GLOBAL.DEVICE)
#         self.fi_2_a = nn.Parameter(torch.FloatTensor(192, 192))#.to(cfg.GLOBAL.DEVICE)
#         self.fi_3_a = nn.Parameter(torch.FloatTensor(192, 192))#.to(cfg.GLOBAL.DEVICE)
        
#         conv1 = nn.Conv2d(16, 8, 1)
#         conv2 = nn.Conv2d(192*2, 192*1, 1)
#         conv3 = nn.Conv2d(192*2, 192*1, 1)
        
        self.embedding_layer1 = nn.Linear(8, 8, bias=False)
        self.embedding_layer2 = nn.Linear(192, 192, bias=False)
        self.embedding_layer3 = nn.Linear(192, 192, bias=False)
        
        self.gcn_list = nn.ModuleList([self.gcn1, self.gcn2, self.gcn3])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_layers = nn.ModuleList([self.embedding_layer1, self.embedding_layer2, self.embedding_layer3])
#         self.conv_layers = nn.ModuleList([conv1, conv2, conv3])
#         self.fi_list = [self.fi_1, self.fi_2, self.fi_3] #nn.ModuleList([fi_1, fi_2, fi_3])
#         self.fi_list_a = [self.fi_1_a, self.fi_2_a, self.fi_3_a] #nn.ModuleList([fi_1_a, fi_2_a, fi_3_a])
        
#         self.reset_parameters()
            
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fi_1.size(1))
        self.fi_1.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fi_2.size(1))
        self.fi_2.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fi_3.size(1))
        self.fi_3.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fi_1_a.size(1))
        self.fi_1_a.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fi_2_a.size(1))
        self.fi_2_a.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.fi_3_a.size(1))
        self.fi_3_a.data.uniform_(-stdv, stdv)
            
    def forward_by_stage(self, input, subnet, rnn, gcn_masks, i):
        seq_number, batch_size, input_channel, height, width = input.size()
        input = torch.reshape(input, (-1, input_channel, height, width))
        input = subnet(input)
        input = torch.reshape(input, (seq_number, batch_size, input.size(1), input.size(2), input.size(3)))
        
        #torch.Size([6, 2, 8, 120, 100])
        #torch.Size([6, 2, 192, 60, 50])
        #torch.Size([6, 2, 192, 30, 25])
        h, w = input.size(3), input.size(4)
        
        # Unified GCN..........
        max_typhoon_number = gcn_masks.size()[2]
        residual_feature_maps = torch.randn(seq_number, batch_size, input.size()[2], input.size()[3], input.size()[4]).cuda()
        gcn_inputs = torch.randn(seq_number*max_typhoon_number, batch_size, input.size()[2]).cuda() # [t, b, c]
        # Step 1. crop the typhoon region feature map and then perform global pooling to get and save the node feature.
        for b_idx in range(0, batch_size):
            node_number = 0
            # get the gcn_masks and its feature maps.
            b_gcn_masks = gcn_masks[b_idx] # (6, 3, 600, 500)
            b_feature_maps = input[b_idx]
            resized_b_gcn_masks = F.interpolate(b_gcn_masks, size=(h, w), mode='nearest')
            # Fix the node numbers to max_typhoon*6.
            for t_idx in range(0, 6):
                temp_masks = resized_b_gcn_masks[t_idx] # (3, 600, 500)
                for resized_temp_mask in temp_masks:
                    curr_feature_maps = input[t_idx][b_idx]
                    before_pool_feature = curr_feature_maps * resized_temp_mask.float()
                    after_pool_feature = self.avg_pool(before_pool_feature).squeeze().squeeze() # [192]
                    gcn_inputs[node_number][b_idx] = after_pool_feature
                    node_number += 1
        
        # Step 2. build the reasoning graph with size node_numbers x node_numbers.
        for b_idx in range(0, batch_size):
            b_gcn_masks = gcn_masks[b_idx] # (6, 3, 600, 500)
            resized_b_gcn_masks = F.interpolate(b_gcn_masks, size=(h, w), mode='nearest') # (6, 3, 600, 500)
            
            node_feature = gcn_inputs[:,b_idx,:] # [18, 192]
#             node_feature = torch.mm(node_feature, self.fi_list[i-1])
            node_feature_T = torch.transpose(node_feature, 1, 0)
#             node_feature_T = torch.mm(self.fi_list_a[i-1], node_feature_T)
            adj = F.softmax(node_feature.mm(node_feature_T)) # node_numbers x node_numbers.
            # Step 3. perform information propagtion on the graph.
            aaa = self.embedding_layers[i-1](node_feature)
            after_gcn_feature = self.gcn_list[i-1](aaa, adj) # [18,192]
            after_gcn_feature = after_gcn_feature.view(after_gcn_feature.size()[0], after_gcn_feature.size()[1], 1, 1).cuda() # (18, 192, 1, 1)
            #unpooling
            after_gcn_feature = F.upsample(after_gcn_feature, size=(input.size()[3], input.size()[4]), mode='nearest') # (18, 192, h, w)
            for t_idx in range(0, 6):
                if max_typhoon_number == 2:
                    residual_feature_maps[t_idx,b_idx,:] = after_gcn_feature[t_idx*2+0] * resized_b_gcn_masks[t_idx][0].float() + after_gcn_feature[t_idx*2+1] * resized_b_gcn_masks[t_idx][1].float()
                elif max_typhoon_number == 3:
                    residual_feature_maps[t_idx,b_idx,:] = after_gcn_feature[t_idx*3+0] * resized_b_gcn_masks[t_idx][0].float() + after_gcn_feature[t_idx*3+1] * resized_b_gcn_masks[t_idx][1].float() + after_gcn_feature[t_idx*3+2] * resized_b_gcn_masks[t_idx][2].float()
                else:
                    assert 0 == 1
#                     + after_gcn_feature[t_idx*3+1] * resized_b_gcn_masks[t_idx][1].float() + after_gcn_feature[t_idx*3+2] * resized_b_gcn_masks[t_idx][2].float()
#                     residual_feature_maps[t_idx,b_idx,:] += after_gcn_feature[t_idx*max_typhoon_number+ty_n] * resized_b_gcn_masks[t_idx][ty_n].float()
# 
        input = input + residual_feature_maps
        outputs_stage, state_stage = rnn(input, None)

        return outputs_stage, state_stage

    # input: 5D S*B*I*H*W
    def forward(self, input, gcn_masks):
        input = input.transpose(1, 0)
        #input = F.upsample(input, size_new=(96, 96), mode='bilinear')
        hidden_states = []
        #logging.debug()
        for i in range(1, self.blocks+1):
            input, state_stage = self.forward_by_stage(input, getattr(self, 'stage'+str(i)), getattr(self, 'rnn'+str(i)), gcn_masks, i)
            hidden_states.append(state_stage)
        return tuple(hidden_states)


    
           # performing GCN here.
#         node_numbers = 0
#         node_features = []
#         print ("gcn_masks.size()", gcn_masks.size()) # torch.Size([2, 6, 3, 600, 500])
        
#         # iter every sample in one mini-batch.
#         for b_idx in range(0, gcn_masks.size()[0]):
#             node_numbers = 0
#             node_features = []
#             position_mark = []
#             # get the gcn_masks and its feature maps.
#             b_gcn_masks = gcn_masks[b_idx]
#             b_feature_maps = input[b_idx]
            
#             for t_idx in range(0, 6):
#                 temp_masks = b_gcn_masks[t_idx]
#                 for y in range(0, 3):
#                     cur_mask = temp_masks[y]
#                     if torch.sum(cur_mask) > 50:
#                         # It has typhoon regions.
#                         position_mark.append(1)
#                         node_numbers += 1
#                         # To get the typhoon region feature map.
#                         cur_feature_map = F.upsample(cur_mask, size=(h, w), mode='nearest') * input[idx]
#                         gcn_input_feature = self.avg_pool(cur_feature_map)
#                         node_features.append(gcn_input_feature)
#                     else:
#                         position_mark.append(0)
        
        
       