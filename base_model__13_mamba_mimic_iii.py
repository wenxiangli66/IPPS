import torch.nn as nn
import torch
import torch.nn.functional as F
# from model_mamba import *
from model_mamba import *
def query_attention(query, keys_values):
    # 计算查询向量与键向量之间的点积，得到注意力分数
    attention_scores = torch.matmul(query, keys_values.transpose(1, 2))

    # 使用 softmax 函数计算注意力权重
    attention_weights = F.softmax(attention_scores, dim=-1)

    # 使用注意力权重加权平均计算注意力向量
    attention_output = torch.matmul(attention_weights, keys_values)

    return attention_output, attention_weights


class Base_model(nn.Module):
    def __init__(self, dropout_ratio=0., input_dim=0, embedding_dim=3, hidden_size=16,output_dim=1,mamba_model=None):
    # def __init__(self, dropout_ratio=0., input_dim=0, embedding_dim=3, hidden_size=16, output_dim=1):
        super().__init__()
        # self.embedding = nn.Embedding(num_embeddings=10, embedding_dim=embedding_dim)
        # self.transformer=transformer_model

        self.mamba1 = mamba_model
        self.mamba2 = mamba_model
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(dropout_ratio)
        self.base_gru = nn.GRU(embedding_dim, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0.0,
                               bidirectional=False)
        self.linear=nn.Linear(hidden_size,output_dim)

    def forward(self, input_seq, s1, s2, s3):
        x = self.embedding(input_seq)
        x_mamba = self.mamba1(x)
        x = self.dropout(x_mamba)
        output, hidden = self.base_gru(x)
        output = self.dropout(output)
        # output = self.dropout(output)
        # output=self.linear(output)

        # x_transformed_1 = self.transformer(s1)
        # x_transformed_1 = self.dropout(x_transformed_1)
        # output_1, hidden_1 = self.base_gru(x_transformed_1)
        x_mamba_1 = self.mamba2(s1)
        x_mamba_1 = self.dropout(x_mamba_1)
        output_1, hidden_1 = self.base_gru(x_mamba_1)
        output_1 = self.dropout(output_1)

        x_mamba_2 = self.mamba2(s2)
        x_mamba_2 = self.dropout(x_mamba_2)
        output_2, hidden_1 = self.base_gru(x_mamba_2)
        output_2 = self.dropout(output_2)

        x_mamba_3 = self.mamba2(s3)
        x_mamba_3 = self.dropout(x_mamba_3)
        output_3, hidden_1 = self.base_gru(x_mamba_3)
        output_3 = self.dropout(output_3)


        # print("hidden1[:][-1][:]", torch.cat((output[:,-1,:], hidden[0,:,:]), dim=1).shape)
        x2 = torch.stack((output_1, output_2, output_3), dim=2).mean(dim=2)
        x = output + x2
        # x = torch.cat((output, output_1, output_2, output_3), dim=2)
        # print("hidden1[:][-1][:]", torch.cat((output[:,-1,:], hidden[0,:,:]), dim=1).shape)
        output = self.linear(x)
        # output=self.linear_1(output[:,-1,:])
        # output=self.relu(output)
        # output=self.linear(output)
        # output=self.linear(output)

        return output


class SeqMoE(nn.Module):
    def __init__(
            self, input_size, embed_size, output_size, num_experts, hidden_size,
            noisy_gating=True, k=4, dropout=0.2, gate_type='gru',
            residual=False, base_gru=True, use_zero_expert=False, feed_error=False,
            incl_base_to_expert=True,mamba_model=None
    ):
        super(SeqMoE, self).__init__()
        # self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.k = k
        self.feed_error = feed_error
        self.incl_base_to_expert = incl_base_to_expert
        self.mamba = mamba_model
        self.base_gru = base_gru

        self.embed_layer = nn.Linear(input_size, embed_size, bias=False)
        self.embed_layer2 = nn.Linear(input_size, embed_size, bias=False)
        # if self.base_gru:
        #     # lock base gru parameters  #
        #     self.embed_layer.weight = base_gru.embed_input.weight  # 是共享嵌入层的权重，意味着两个层将使用相同的嵌入。这可以确保嵌入保持一致。
        #     self.base_gru.embed_input.weight.requires_grad = False
        #     for param in self.base_gru.rnn.parameters():
        #         param.requires_grad = False
        #     for param in self.base_gru.fc_out.parameters():
        #         param.requires_grad = False
        # 反向传播期间不更新 GRU的循环部分（rnn）和全连接输出层（fc_out）的参数

        # instantiate experts
        # MLP(self.input_size, self.output_size, self.hidden_size)
        inp_dim_expert = embed_size
        if self.feed_error:
            inp_dim_expert += output_size

        self.experts = nn.ModuleList(
            [GRUPredictor1(inp_dim_expert, self.hidden_size, self.output_size, dropout,mamba_model = self.mamba)
             for i in range(self.num_experts)])  # 专家个数循环50
        # transformer_model = self.transformer
        num_experts = self.num_experts
        # if use_zero_expert:
        #     num_experts += 1
        #     self.experts.append(ZeroModule(self.output_size))

        # if self.incl_base_to_expert:#没有用到这个参数
        #     num_experts += 1

        # curren time step embed => number of experts
        if gate_type == 'mlp':
            self.gate = nn.Linear(embed_size, num_experts)
        elif gate_type == 'gru':
            self.gate = GRUPredictor1(
                embed_size, self.hidden_size, num_experts, dropout,mamba_model = self.mamba)

    # 门的选择

    # 门的前向传播
    def forward(self, inp, data_three_cos, trg):

        device = inp.device # 获取输入张量 inp 所在的设备 (GPU 或 CPU)。
        data_three_cos.to(device)
        trg.to(device)
        batch_size = inp.size(0)  # 获取输入张量 inp 的批大小（batch size）

        inp_embed = self.embed_layer(inp)  # 使用一个嵌入层 (embed_layer) 将输入数据 inp 转换为嵌入向量
        inp_embed_pro = self.embed_layer2(data_three_cos)
        # inp_embed_pro = self.embed_layer(data_three_cos)


        attention_output1, attention_weights1 = query_attention(inp_embed, inp_embed_pro[0:300:3])
        # 计算 preds 和 preds2 之间的注意力权重
        attention_output2, attention_weights2 = query_attention(inp_embed, inp_embed_pro[1:300:3])
        # 计算 preds 和 preds3 之间的注意力权重
        attention_output3, attention_weights3 = query_attention(inp_embed, inp_embed_pro[2:300:3])

        # print(inp_embed_pro[0:300:3].shape)
        # print(inp_embed.shape)

        s1 = attention_output1
        s2 = attention_output2
        s3 = attention_output3

        # inp1_embed = self.embed_layer(inp1)
        # inp2_embed = self.embed_layer(inp2)
        # inp3_embed = self.embed_layer(inp3)

        gate_val = self.gate(inp_embed, s1, s2, s3)  # 计算门控值
        # x_transformed = self.transformer(inp)
        # gate_val = self.gate(x_transformed)
        gate_val = F.softmax(gate_val, dim=2)  # softmax 操作,第三个维度上执行
        # gate_val: n_batch x n_seq x n_gate

        n_batch, n_seq, _ = inp.shape  # 获取输入张量 inp 的形状信息




        # basemodel的前向传播
        if self.base_gru:
            # base_init_hidden = GRUPredictor.init_hidden(
            #     batch_size, self.base_gru.hidden_dim
            # ).to(device)#初始化 GRU 模型的隐藏状态

            preds = self.base_gru(inp, s1, s2, s3)#初始化base_model
            #
            # attention_output1, attention_weights1 = query_attention(preds.unsqueeze(1), preds1)
            # # 计算 preds 和 preds2 之间的注意力权重
            # attention_output2, attention_weights2 = query_attention(preds.unsqueeze(1), preds2)
            # # 计算 preds 和 preds3 之间的注意力权重
            # attention_output3, attention_weights3 = query_attention(preds.unsqueeze(1), preds3)
            #
            # # 对注意力权重进行归一化
            # total_attention_weights = attention_weights1 + attention_weights2 + attention_weights3
            # normalized_attention_weights1 = attention_weights1 / total_attention_weights
            # normalized_attention_weights2 = attention_weights2 / total_attention_weights
            # normalized_attention_weights3 = attention_weights3 / total_attention_weights
            #
            preds = torch.sigmoid(preds)
            # preds_total=preds+preds1*normalized_attention_weights1+preds2*normalized_attention_weights2+preds3*normalized_attention_weights3
        # -----------------------------
        # base model 的输入和输出 -----------------------------

        else:
            preds = torch.zeros(n_batch, n_seq, self.output_size).to(device)

        if self.incl_base_to_expert and self.base_gru:
            gate_exp_val = gate_val[:, :, -1].unsqueeze(2)
            # gate_exp_val: n_batch x n_seq x 1
            preds = preds * gate_exp_val  # 图中未标记的g

        if self.feed_error:
            if self.base_gru == None:
                raise RuntimeError("should have base_gru to run feed_error")
            errors = F.l1_loss(preds, trg, reduction='none')

            # shift timesteps +1, and add zero padding for the first one
            errors = errors[:, :-1, :]  # [b, seq, dim]
            zero_pad = torch.zeros(batch_size, 1, self.output_size)
            zero_pad = zero_pad.to(errors.device)
            prev_errors = torch.cat((zero_pad, errors), dim=1)
        experts_list=[]
        # 50个专家模型的前向传播 ---遍历每个
        for exp_idx, expert in enumerate(self.experts):

            if self.feed_error:
                inp_expert = torch.cat((inp_embed, prev_errors), dim=2)
            else:
                inp_expert = inp_embed
            # 前向传播
            pred_exp = expert(inp_expert, s1, s2, s3)

            # pred_exp: n_batch x n_seq x n_events
            experts_list.append(pred_exp)
            gate_exp_val = gate_val[:, :, exp_idx].unsqueeze(2)
            # gate_exp_val: n_batch x n_seq x 1
            # 最终输出公式----------------------------------------
            preds = preds + gate_exp_val * pred_exp

        preds = preds.clamp(0, 1)
        return preds, gate_val,experts_list

    def init_hidden(self, batch_size):
        return GRUPredictor.init_hidden(batch_size, self.hidden_size)


class GRUPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout=0.2,transformer_model=None):
        super(GRUPredictor, self).__init__()
        self.transformer=transformer_model
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embed_dim, hidden_dim,
                          dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    # 专家的前向传播
    def forward(self, embedded_inp, s1, s2, s3):
        batch_size = embedded_inp.size(0)
        device = embedded_inp.device
        s1, s2, s3=s1.to(device), s2.to(device), s3.to(device)
        # hidden = self.init_hidden(batch_size, self.hidden_dim).to(device)
        # hidden2 = self.init_hidden(batch_size, data_three_cos.shape[1]).to(device)

        transformed_input = self.transformer(embedded_inp)
        transformed_input_1 = self.transformer(s1)
        transformed_input_2 = self.transformer(s2)
        transformed_input_3 = self.transformer(s3)
  
        output, hidden = self.gru(transformed_input)
        output_1, hidden = self.gru(transformed_input_1)
        output_2, hidden = self.gru(transformed_input_2)
        output_3, hidden = self.gru(transformed_input_3)

  
        #[100, hidden_dim*4]
        # prediction = self.proj_out(torch.cat((output[:,-1,:], output_1[:,-1,:], output_2[:,-1,:], output_3[:,-1,:]), dim=1))
        preds = torch.sigmoid(hidden)
        prediction = self.proj_out(preds)
        prediction = torch.sigmoid(prediction)
        return prediction

    @staticmethod
    def init_hidden(batch_size, hidden_dim):
        init = 0.1
        h0 = torch.randn(batch_size, hidden_dim)
        h0.data.uniform_(-init, init)
        return h0.unsqueeze(0)

# 定义只包含编码器的Transformer模型
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.transformer_encoder(src)
        src = self.fc(src)
        return src

# 数据预处理,准备训练集、测试集
# GPU-device
# train 函数
# test函数
# 预测函数
# dataset、dataloader
# 损失函数和优化器
# 评估


# if __name__ == '__main__':
#     # 2 100 964  --- 100 2 964   看batch_first 参数 【, batch , ?】
#     # 4 3 20 ---> 100 2 942 ---- 输入和输出怎么改
#     # 看今天这个
#     inputs = torch.ones(4, 3, 20, dtype=torch.float)
#     trg = torch.ones_like(inputs, dtype=torch.float)
#     base_gru1 = Base_model(dropout_ratio=0., embedding_dim=3, hidden_size=16, input_dim=20,output_dim=1)  # 这个10 是输入的第三个维度数字，为什么需要
#
#     # base_gru = base_gru1(inputs)
#     model = SeqMoE(input_size=20, embed_size=50, output_size=1, num_experts=50, hidden_size=16, base_gru=base_gru1)  # 模型实例化
#
#
#     # dataset  yiji dataloader 类
#     out = model(inputs, trg)
class GRUPredictor1(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, dropout=0.2,mamba_model=None):
        super(GRUPredictor1, self).__init__()
        self.mamba1=mamba_model
        self.mamba2 = mamba_model
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(embed_dim, hidden_dim,
                          dropout=dropout, batch_first=True)
        self.proj_out = nn.Linear(hidden_dim, output_dim)

    # 专家的前向传播
    def forward(self, embedded_inp, s1, s2, s3):
        batch_size = embedded_inp.size(0)
        device = embedded_inp.device
        hidden = self.init_hidden(batch_size, self.hidden_dim).to(device)
        mamba_input = self.mamba1(embedded_inp)
        hidden_1, _ = self.gru(mamba_input, hidden)

        mamba_input_1 = self.mamba2(s1)
        mamba_input_2 = self.mamba2(s2)
        mamba_input_3 = self.mamba2(s3)

        # output, hidden = self.gru(transformed_input)
        output_1, hidden = self.gru(mamba_input_1)
        output_2, hidden = self.gru(mamba_input_2)
        output_3, hidden = self.gru(mamba_input_3)
        # print(hidden.shape)
        # 对注意力权重进行归一化
        x2 = torch.stack((output_1, output_2, output_3), dim=2).mean(dim=2)
        x = hidden_1 + x2
        prediction = self.proj_out(x)
        # # prediction = self.proj_l1(output[:,-1,:])
        # prediction = torch.sigmoid(prediction)
        # prediction = self.proj_out(hidden)
        prediction = torch.sigmoid(prediction)
        # # prediction = self.proj_l1(output[:,-1,:])
        # prediction = torch.sigmoid(prediction)
        # prediction = self.proj_out(hidden)
        #prediction = torch.sigmoid(prediction)
        # print('gate', prediction.shape)
        return prediction

    @staticmethod
    def init_hidden(batch_size, hidden_dim):
        init = 0.1
        h0 = torch.randn(batch_size, hidden_dim)
        h0.data.uniform_(-init, init)
        return h0.unsqueeze(0)

# 定义只包含编码器的Transformer模型
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, num_layers):
        super(TransformerEncoderModel, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = self.transformer_encoder(src)
        src = self.fc(src)
        return src
#
# args = ModelArgs(
#     d_model=512,
#     n_layer=12,
#     vocab_size=1613  # 例如，BERT使用的词汇表大小
# )
# mamba_model = ResidualBlock(args)