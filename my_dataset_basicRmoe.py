from base_model__13 import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

_TEST_RATIO = 0.2
_VALIDATION_RATIO = 0.1


def load_data_simple(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')))  # 加载序列数据
    labels = np.array(pickle.load(open(labelFile, 'rb')))  # 加载标签数据
    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    dataSize = len(labels)
    np.random.seed(0)
    ind = np.random.permutation(dataSize)  # 对数据进行随机排列的索引数组，以便后续划分数据集
    nTest = int(_TEST_RATIO * dataSize)  # 计算测试集和验证集的样本数量。
    nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = ind[:nTest]  # 使用索引数组划分测试集、验证集和训练集。
    valid_indices = ind[nTest:nTest + nValid]
    train_indices = ind[nTest + nValid:]

    train_set_x = sequences[train_indices]  # 通过索引获取对应数据集的序列和标签。
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    valid_set_x = sequences[valid_indices]
    valid_set_y = labels[valid_indices]
    train_set_t = None
    valid_set_t = None
    test_set_t = None

    if len(timeFile) > 0:
        train_set_t = pickle.load(open(timeFile + '.train', 'rb'))
        valid_set_t = pickle.load(open(timeFile + '.valid', 'rb'))
        test_set_t = pickle.load(open(timeFile + '.test', 'rb'))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # 使用 len_argsort 函数对训练集、验证集和测试集中的序列数据进行排序
    train_sorted_index = len_argsort(train_set_x)
    train_set_x = [train_set_x[i] for i in train_sorted_index]
    train_set_y = [train_set_y[i] for i in train_sorted_index]
    # 使用 train_sorted_index 对训练集的输入数据 train_set_x 进行重新排序
    valid_sorted_index = len_argsort(valid_set_x)
    valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    valid_set_y = [valid_set_y[i] for i in valid_sorted_index]

    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]

    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_t)
    valid_set = (valid_set_x, valid_set_y, valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_t)

    return train_set, valid_set, test_set


def padMatrixWithoutTime(seqs, inputDimSize=1613):  # inputdimsize这儿设的值是错的
    lengths = np.array([len(seq) for seq in seqs]).astype('int32')
    # 计算每个序列的长度，并将长度转换为整数类型，存储在 lengths 中。
    n_samples = len(seqs)
    maxlen = np.max(lengths)

    x = np.zeros((maxlen, n_samples, inputDimSize))  # 函数创建一个全零张量，用于存储填充后的序列数据
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.
    # [20 30 40 ]  [0 00 0 1 0 0 00 -1 ----00 010]
    return x, lengths


# 通过遍历序列数据 seqs，将每个子序列中的整数值用 one-hot 编码的形式在 x 中标记为 1。

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.seq = self.dataset[0]
        self.label = self.dataset[1]

    # __getitem__方法也是一个特殊方法，用于按索引获取数据
    def __getitem__(self, item):
        return self.seq[item], self.label[item]

    # __len__ 方法同样是一个特殊方法，在需要确定数据集大小时被调用
    def __len__(self):
        return len(self.seq)


def my_collate_fun(batch):
    texts, labels = zip(*batch)
    new_texts, length = padMatrixWithoutTime(texts)
    new_texts = torch.tensor(new_texts).float().permute(1, 0, 2)
    return torch.tensor(new_texts).float(), torch.tensor(labels).float(), np.maximum(length, 1)


trainSet, validSet, testSet = load_data_simple(r"D:\python project\third paper\LSTM\output\output_filename.3digitICD9.seqs",
                                               r"D:\python project\third paper\LSTM\output/output_filename.morts", timeFile="")
train_dataset = MyDataset(trainSet)
test_dataset = MyDataset(testSet)
print(len(train_dataset))
a = train_dataset[0][0]
b = train_dataset[1]
c = train_dataset[2]
train_dataloader = DataLoader(train_dataset, batch_size=100, collate_fn=my_collate_fun, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, collate_fn=my_collate_fun, shuffle=True, drop_last=True)

for i, data in enumerate(train_dataloader):
    print(data)
  #  print(data.shape())
# i是批次序号，而 data 则是从 train_dataloader 中获取的一个批次数据，包含了填充后的文本数据、对应的标签数据以及文本长度信息


# 实例化模型
if __name__ == '__main__':

    base_gru1= Base_model(dropout_ratio=0., embedding_dim=50, hidden_size=16, input_dim=942,
                           output_dim=1)  # 这个20 是输入的第三个维度数字，为什么需要

    # base_gru = base_gru1(inputs)
    model = SeqMoE(input_size=942, embed_size=50, output_size=1, num_experts=50, hidden_size=16,
                   base_gru=base_gru1)  # 模型实例化

    model.base_gru.load_state_dict(torch.load("./base_model1.pth"))
    for param in model.base_gru.parameters():
        param.requires_grad = False

    # #损失函数、优化器
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100  # 设定训练轮数
    best_auroc=0.0
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(train_dataloader):
            inputs, labels, _ = data  # 获取输入数据和标签

            optimizer.zero_grad()  # 梯度清零

            # 前向传播
            outputs, _ = model(inputs, labels)
            # outputs = model(inputs)
            # outputs = outputs.squeeze(2)  # [100 7 1]
            outputs1 = outputs[:, -1, 0]  # outputs1 是从模型输出中提取出了在最后一个时间步（序列的末尾）的第一个特征或维度的值。
            # print(outputs1.size())
            # print(labels)
            # print(outputs[1].size())
            #         # 计算损失
            # print(outputs.size())
            # labels = labels.long()
            loss = loss_fn(outputs1, labels)
            #
            #         # 反向传播和优化
            loss.backward()
            optimizer.step()
            #
            #         # 统计损失
            running_loss += loss.item()
            if i % 10 == 9:  # 每10个batch打印一次损失
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10}')
                running_loss = 0.0

        # 在每个epoch结束后，用测试集评估模型并计算AUROC
        model.eval()  # 切换到评估模式
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for test_data in test_dataloader:
                test_inputs, test_labels, _ = test_data
                test_outputs, _ = model(test_inputs, test_labels)
                # test_outputs = model(test_inputs)
                test_outputs = test_outputs[:, -1, 0]
                all_predictions.append(test_outputs.detach().cpu().numpy())
                all_labels.append(test_labels.detach().cpu().numpy())

        # 将预测和标签转换为numpy数组
        all_predictions = np.concatenate(all_predictions)
        all_labels = np.concatenate(all_labels)

        # 计算AUROC
        auroc = roc_auc_score(all_labels, all_predictions)
        print(f'Epoch [{epoch + 1}/{epochs}], Test AUROC: {auroc}')
        # torch.save(model.state_dict(), "base_model1.pth")
        if auroc > best_auroc:
            best_auroc = auroc
    print(f'Best AUROC: {best_auroc}')  # 打印训练过程中的最佳 AUROC 值

# for i, data in enumerate(dataloader):
# 	update_cnt = 0
#
# 	inp, trg, len_inp, len_trg, inp_time, trg_time, hadm_ids \
# 		= process_batch(self, data, epoch)
#
# 	if self.moe:
# 		pred, gating_score = self.model(inp, len_inp, trg)
#
# 		loss = self.loss_fn(pred, trg.float(),
# 							len_trg,
# 							self.use_bce_logit,
# 							self.use_bce_stable,
# 							pos_weight=self.pos_weight,
# 							event_weight=self.event_weights
# 							)
# 		t_loss += loss.item()
#
# 		loss.backward()
# 		optimizer.step()
# 		self.model.zero_grad()
