#读入数据
import numpy
from torch.utils.tensorboard import SummaryWriter
from base_model__13_mamba_mimic_iii import *
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
import random
from sklearn.metrics import accuracy_score
manualSeed = 1
random.seed(manualSeed)
np.random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from model_mamba import *
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，则会报错，可以验证代码中有没有cuda没有确定性实现的代码
# torch.use_deterministic_algorithms(True)
# 这么设置使用确定性算法，如果代码中有算法cuda没有确定性实现，也不会报错
torch.use_deterministic_algorithms(True, warn_only=True)

# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
_TEST_RATIO = 0.3
# _VALIDATION_RATIO = 0.1

writer =  SummaryWriter('./log')
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')  # 保存模型
        self.val_loss_min = val_loss
early_stopping = EarlyStopping(patience=10, verbose=True)
def one_hot(samples):
    labels = list(set(label for sample in samples for label in sample))
    # Create a dictionary to map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    # Convert samples to one-hot encoding
    one_hot_samples = []
    for sample in samples:
        one_hot_sample = [0] * len(labels)
        for label in sample:
            one_hot_sample[label_to_index[label]] = 1
        one_hot_samples.append(one_hot_sample)
    # Convert the one-hot samples to a PyTorch tensor
    tensor_samples = torch.tensor(one_hot_samples, dtype=torch.float)
    return tensor_samples

def delete_null(data,lable):
    valid_indexes = [i for i, sublist in enumerate(data) if len(sublist) > 0]
    data = [data[i] for i in valid_indexes]
    lable = [lable[i] for i in valid_indexes]

    return np.array(data, dtype=object), np.array(lable)
def load_data_simple(seqFile, labelFile, timeFile=''):
    sequences = np.array(pickle.load(open(seqFile, 'rb')), dtype=object)  # 加载序列数据
    labels = np.array(pickle.load(open(labelFile, 'rb')))  # 加载标签数据
    # labels2 = np.array(pickle.load(open("./output/output_filename.4digit_label.seqs", "rb")), dtype=object)
    labels2 = np.array(pickle.load(open("./output/output_filename.4digit_label.seqs", "rb")))

    labels2 = one_hot(labels2)
    sequences, labels = delete_null(sequences, labels)

    if len(timeFile) > 0:
        times = np.array(pickle.load(open(timeFile, 'rb')))

    dataSize = len(labels)
    np.random.seed(0)
    nid = np.random.permutation(dataSize)  # 对数据进行随机排列的索引数组，以便后续划分数据集
    nTest = int(_TEST_RATIO * dataSize)  # 计算测试集和验证集的样本数量。
    # nValid = int(_VALIDATION_RATIO * dataSize)

    test_indices = nid[:nTest]  # 使用索引数组划分测试集、验证集和训练集。
    # valid_indices = nid[nTest:nTest + nValid]
    train_indices = nid[nTest:]

    train_set_x = sequences[train_indices]  # 通过索引获取对应数据集的序列和标签。
    train_set_y = labels[train_indices]
    test_set_x = sequences[test_indices]
    test_set_y = labels[test_indices]
    # valid_set_x = sequences[valid_indices]
    # valid_set_y = labels[valid_indices]
    train_set_t = None
    valid_set_t = None
    test_set_t = None


    train_set_y_2 = labels2[train_indices]
    test_set_y_2= labels2[test_indices]
    # valid_set_y_2 = labels2[valid_indices]




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
    # train_set_x,train_set_y=delete_null(train_set_x, train_set_y)
    train_set_y_2 = [train_set_y_2[i] for i in train_sorted_index]

    # valid_sorted_index = len_argsort(valid_set_x)
    # valid_set_x = [valid_set_x[i] for i in valid_sorted_index]
    # valid_set_y = [valid_set_y[i] for i in valid_sorted_index]
    # valid_set_y_2 = [valid_set_y_2[i] for i in valid_sorted_index]


    test_sorted_index = len_argsort(test_set_x)
    test_set_x = [test_set_x[i] for i in test_sorted_index]
    test_set_y = [test_set_y[i] for i in test_sorted_index]
    test_set_y_2 = [test_set_y_2[i] for i in test_sorted_index]
    # test_set_x, test_set_y = delete_null(test_set_x, test_set_y)
    if len(timeFile) > 0:
        train_set_t = [train_set_t[i] for i in train_sorted_index]
        # valid_set_t = [valid_set_t[i] for i in valid_sorted_index]
        test_set_t = [test_set_t[i] for i in test_sorted_index]

    train_set = (train_set_x, train_set_y, train_set_y_2,train_set_t)
    # valid_set = (valid_set_x, valid_set_y, valid_set_y_2,valid_set_t)
    test_set = (test_set_x, test_set_y, test_set_y_2,test_set_t)

    return train_set,  test_set


def padMatrixWithoutTime(seqs, inputDimSize=942):  # inputdimsize这儿设的值是错的

    length = np.array([len(seq) for seq in seqs]).astype('int32')
    maxlen = np.max(length)

    # 计算每个序列的长度，并将长度转换为整数类型，存储在 lengths 中。
    n_samples = len(seqs)

    x = np.zeros((maxlen, n_samples, inputDimSize))  # 函数创建一个全零张量，用于存储填充后的序列数据
    for idx, seq in enumerate(seqs):
        for xvec, subseq in zip(x[:, idx, :], seq):
            xvec[subseq] = 1.
    # [20 30 40 ]  [0 00 0 1 0 0 00 -1 ----00 010]
    return x, length


# 通过遍历序列数据 seqs，将每个子序列中的整数值用 one-hot 编码的形式在 x 中标记为 1。

class MyDataset(Dataset):
    def __init__(self, dataset, cos_index_path=None):
        self.dataset = dataset
        self.seq = self.dataset[0]
        self.label = self.dataset[1]
        self.label2=self.dataset[2]

        self.cos_index = np.load(cos_index_path)
    # __getitem__方法也是一个特殊方法，用于按索引获取数据
    def __getitem__(self, item):
        return self.seq[item], self.label[item], self.label2[item], self.cos_index[item]

    # __len__ 方法同样是一个特殊方法，在需要确定数据集大小时被调用
    def __len__(self):
        return len(self.seq)


def my_collate_fun1(batch):
    texts, labels, label2, cos_index = zip(*batch)
    new_texts, length = padMatrixWithoutTime(texts)
    new_texts = torch.tensor(new_texts).float().permute(1, 0, 2)
    label2 = torch.stack(label2)
    new_cos_index = torch.tensor(cos_index)
    return torch.tensor(new_texts).float(), torch.tensor(labels).float(), torch.tensor(label2).float(), torch.tensor(length), torch.tensor(new_cos_index)


trainSet, testSet = load_data_simple("./output/output_filename.3digitICD9.seqs", "./output/output_filename.morts", timeFile="")
# train_dataset = MyDataset(trainSet)
# test_dataset = MyDataset(testSet)

train_dataset = MyDataset(trainSet, cos_index_path="cos_train_max_thress_index.npy")
test_dataset = MyDataset(testSet, cos_index_path="cos_test_max_thress_index.npy")
print(len(train_dataset))
a = train_dataset[0][0]
b = train_dataset[1]
c = train_dataset[2]
train_dataloader = DataLoader(train_dataset, batch_size=100, collate_fn=my_collate_fun1, shuffle=False, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=100, collate_fn=my_collate_fun1, shuffle=False, drop_last=True)
#
# for i, (data, label, label2, _) in enumerate(train_dataloader):
    # print(label2)
# i是批次序号，而 data 则是从 train_dataloader 中获取的一个批次数据，包含了填充后的文本数据、对应的标签数据以及文本长度信息

#统计数据集中每个病人，最后一个病历数据中出现的疾病数量以及对应的1-level CCS Code数量分布
    #确定最终的Expert个数 N
    #确定要分类的疾病数量 K


#读入3-level CCS Code -- 1-level CCS Code映射，构建Dict

#遍历数据集中的每个病人
    #按照病例序列中的最后一个病例数据 查询Dict 生成最后一个病例数据对应的 1-level标签
    #将病人最后一个病例数据中的疾病，映射到0~K之间，作为待预测的label

#将每个病人的标签序列加入到数据集中

#模型代码
    #单个Expert
        #输入为病人病例序列
        #输出为1-level对应的3-level疾病

    #MOE
        #设置N个Expert，每个对应一个1-level标签
        #将病人序列输入到每个Expert
        #按照Expert与 3-level CCS code的映射关系，将专家的预测结果合并为一个K维的向量，这个K维的向量作为用户下一次患病情况的预测输出，O1
        #设置一个Gate net，输入为Expert输出concat，输出为一个值，对应是否死亡，O2
    #Loss
        #将O1与用户最后一个病例中的预测Label计算 Loss1
        #将O2与用户是否死亡计算 Loss2
        #采用加权和的方式和并Loss1 与 Loss2
import warnings
warnings.filterwarnings("ignore",category=UserWarning)
#训练模型
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    # transformer_encoder_model = TransformerEncoderModel(input_dim=80, output_dim=80, num_heads=4, num_layers=2)
    # base_gru1 = Base_model(dropout_ratio=0.3, embedding_dim=80, hidden_size=16, input_dim=1613, output_dim=1, mamba_model=mamba_model_)  # 这个20 是输入的第三个维度数字，为什么需要
    #
    # # base_gru = base_gru1(inputs)
    # model = SeqMoE(input_size=1613, embed_size=80, output_size=1 + 19, num_experts=19, hidden_size=32,
    #                base_gru=base_gru1,mamba_model=mamba_model_).to(device)  # 模型实例化
    # #
    # base_gru1= Base_model(dropout_ratio=0., embedding_dim=50, hidden_size=16, input_dim=1613,
    #                            output_dim=1)  # 这个20 是输入的第三个维度数字，为什么需要
    #
    # model = SeqMoE(input_size=1613, embed_size=50, output_size=1 + 25, num_experts=25, hidden_size=16,
    #                    base_gru=base_gru1).to(device)  # 模型实例化



    # model.base_gru.load_state_dict(torch.load("./base_model1.pth"))
    # for param in model.base_gru.parameters():
    #     param.requires_grad = False

    # #损失函数、优化器
    # loss_fn = nn.BCELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 10# 设定训练轮数
    runs = 3

    accuracy_values = []
    recall_values = []
    precision_values = []
    f1_values = []
    auroc_values = []
    auprc_values = []
    for run in range(runs):
        best_auprc = 0
        args = ModelArgs(
            d_model=128,
            n_layer=2,
            vocab_size=942
        )
        #
        mamba_model_ = Mamba(args)
        base_gru1 = Base_model(dropout_ratio=0.1, embedding_dim=128, hidden_size=64, input_dim=942, output_dim=1,
                               mamba_model=mamba_model_)  # 这个20 是输入的第三个维度数字，为什么需要

        # base_gru = base_gru1(inputs)
        model = SeqMoE(input_size=942, embed_size=128, output_size=1 + 19, num_experts=19, hidden_size=64,
                       base_gru=base_gru1, mamba_model=mamba_model_).to(device)  # 模型实例化
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        for epoch in range(epochs):
            running_loss = 0.0
            model.train()

            train_label = []
            train_pre_label = []
            for j, data in enumerate(train_dataloader):
                weight = np.zeros(len(data[2]))
                for i in range(len(data[2])):
                    if data[1][i] == 0:
                        weight[i] = 1.5
                    else:
                        weight[i] = 1.0
                weight = torch.Tensor(weight).to(device)
                loss_fn = nn.BCELoss(weight=weight)
                inputs, labels,labels2, lengths, new_cos_index = data  # 获取输入数据和标签
                inputs, labels, labels2 ,new_cos_index = inputs.to(device), labels.to(device), labels2.to(device),new_cos_index.to(device)
                # print(inputs.shape)

                #取相似病人的数据
                data_three_cos = []
                for k in range(new_cos_index.shape[0]):
                    # data_three_cos.extend([train_dataset[i][0] for i in new_cos_index[k]])
                    for kk in new_cos_index[k]:
                        data_three_cos.append(train_dataset[kk][0])
                data_three_cos = padMatrixWithoutTime(data_three_cos)
                data_three_cos = torch.tensor(data_three_cos[0]).float().permute(1, 0, 2)
                data_three_cos = [data_three_cos]
                data_three_cos = torch.stack(data_three_cos)
                # print(data_three_cos.shape())
                data_three_cos = data_three_cos.squeeze()
                data_three_cos = data_three_cos.to(device)
                optimizer.zero_grad()  # 梯度清零
                # 前向传播
                outputs, _,experts_list = model(inputs,data_three_cos,labels2)
                # outputs = model(inputs)
                # outputs = outputs.squeeze(2)  # [100 7 1]
                #变成4维的输出，把专家的输出结果拼接到后面
                outputs1 = outputs[:, -1, 0]  # outputs1 是从模型输出中提取出了在最后一个时间步（序列的末尾）的第一个特征或维度的值。
                # print(outputs1.size())
                # print(labels)
                # print(outputs[1].size())
                #         # 计算损失
                # print(outputs.size())
                # labels = labels.long()
                #for 循环 25个loss

    #每个专家最后输出的向量都是（100，长度，22）

                experts_loss_all = 0
                experts_predict_all = []
                for i, experts_predict in enumerate(experts_list):
                    loss_expert_i = loss_fn(experts_predict[:, -1, i+1], labels2[:, i])
                    experts_loss_all += loss_expert_i
                    expert_predict = experts_predict[:, -1, i + 1]
                    experts_predict_all.append(expert_predict.cpu())#将当前专家模型最后一次的预测结果添加到列表中
                expert_predict_last = torch.stack(experts_predict_all).transpose(0, 1)#预测结果堆叠起来，并进行维度变换，使得每行对应一个样本的所有专家模型的最后一次预测结果。（100,21）

                # loss_expert_sum = torch.sum(experts_loss_all)

                # experts_last_step = [expert[:, -1, i + 1] for i, expert in enumerate(experts_list)]
                # experts_last_step = torch.stack(experts_last_step).transpose(0, 1)
                # experts_loss = loss_fn(experts_last_step, labels2)
                loss = loss_fn(outputs1, labels)+experts_loss_all
                writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + j)
                # loss = loss_fn(outputs1, labels)
                with open("train.txt", "a") as f:
                    f.write(str(loss.item()) + "\n")

                # auroc_experts = []
                # for i, labels2_predict in enumerate(expert_predict_last):
                #     auroc_experts1=roc_auc_score(labels2_predict[i],labels2[:,i])
                #     auroc_experts.append(auroc_experts1)
                # average_auroc_experts = np.mean(auroc_experts)
                # print("平均AUROC值（所有专家）:", average_auroc_experts)
                labels2_predict = expert_predict_last.float()#算AUC的时候不需要
                train_label.append(labels2.cpu().tolist())
                train_pre_label.append(labels2_predict.cpu().tolist())

                # experts_loss_all.backward()
                #         # 反向传播和优化
                loss.backward()
                optimizer.step()
                #
                #         # 统计损失
                running_loss += loss.item()
                if j % 10== 9:  # 每10个batch打印一次损失
                    print(
                        f'Epoch [{epoch + 1}/{epochs}], Step [{j + 1}/{len(train_dataloader)}], Loss: {running_loss / 10}')
                    running_loss = 0.0
            train_label = numpy.array(train_label).reshape(-1)
            train_pre_label = numpy.array(train_pre_label).reshape(-1)
            accuracy = roc_auc_score(train_label, train_pre_label)
            print("train labels2 auroc:",accuracy)
            # print(labels2_predict[:20, :])

            # 在每个epoch结束后，用测试集评估模型并计算AUROC
            model.eval()  # 切换到评估模式
            all_predictions = []
            all_labels = []
            val_loss = 0.0
            test_labels2_list = []
            # test_pre_label = []
            test_labels2_predict = []
            with torch.no_grad():
                for id,test_data in enumerate(test_dataloader):
                    test_inputs, test_labels,test_labels2,test_lengths,test_cos_index = test_data
                    test_inputs, test_labels, test_labels2,test_cos_index = test_inputs.to(device), test_labels.to(device), test_labels2.to(device),test_cos_index.to(device)

                    test_data_three_cos=[]
                    for m in range(test_cos_index.shape[0]):
                        for n in test_cos_index[m]:
                            test_data_three_cos.append(test_dataset[n][0])
                    test_data_three_cos=padMatrixWithoutTime(test_data_three_cos)
                    test_data_three_cos=torch.tensor(test_data_three_cos[0]).float().permute(1,0,2)
                    test_data_three_cos = [test_data_three_cos]
                    test_data_three_cos=torch.stack(test_data_three_cos)
                    test_data_three_cos = test_data_three_cos.squeeze()
                    test_data_three_cos=test_data_three_cos.to(device)


                    test_outputs, _, experts_list_test = model(test_inputs, test_data_three_cos, test_labels)
                    test_outputs = test_outputs[:, -1, 0]

                    experts_loss_all1 = 0
                    experts_predict_all_test = []
                    for i, experts_predict in enumerate(experts_list_test):
                        loss_expert_i = loss_fn(experts_predict[:, -1, i + 1], labels2[:, i])
                        experts_loss_all1 += loss_expert_i
                        expert_predict = experts_predict[:, -1, i + 1]
                        experts_predict_all_test.append(expert_predict)
                    expert_predict_test = torch.stack(experts_predict_all_test).transpose(0, 1)
                    # test_labels2_list.append(expert_predict_test)
                    loss=loss_fn(test_outputs,test_labels)+experts_loss_all1
                    writer.add_scalar('Loss/test', loss.item(), epoch * len(test_dataloader) + id)
                    with open("test.txt", "a") as f:
                        f.write(str(loss.item()) + "\n")
                    labels2_predict = expert_predict_test.float()  # 算AUC的时候不需要
                    test_labels2_list.append(test_labels2.cpu().tolist())
                    test_labels2_predict.append(expert_predict_test.cpu().tolist())

                    val_loss += loss.item()

                    # test_outputs = model(test_inputs)
                    # test_outputs = test_outputs[:, -1, 0]
                    all_predictions.append(test_outputs.detach().cpu().numpy())
                    all_labels.append(test_labels.detach().cpu().numpy())

            # 计算验证集上的平均损失
            # val_loss /= len(test_dataloader)
            #
            # print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}')
            #
            # # Early stopping check
            # early_stopping(val_loss, model)
            #
            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     break



            test_labels2_list = numpy.array(test_labels2_list).reshape(-1)
            test_labels2_predict = numpy.array(test_labels2_predict).reshape(-1)
            accuracy = roc_auc_score(test_labels2_list, test_labels2_predict)
            print("test labels2 auroc:", accuracy)

            # 将预测和标签转换为numpy数组
            all_predict = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)
            precision, recall, _ = precision_recall_curve(all_labels, all_predict)

            all_predictions = [1 if i > 0.39 else 0 for i in all_predict]
            accuracy = accuracy_score(all_labels, all_predictions)
            # 计算 Recall
            recall = recall_score(all_labels, all_predictions, zero_division=1)
            # 计算 Precision
            precision = precision_score(all_labels, all_predictions, zero_division=1)
            # 计算 F1-Score
            f1 = f1_score(all_labels, all_predictions, zero_division=1)
            # 计算AUPRC
            # auprc = auc(recall, precision)

            #precision1, recall1, _ = precision_recall_curve(all_labels, all_predict, pos_label=1)
            auprc = average_precision_score(all_labels, all_predict)

            # 计算AUROC
            auroc = roc_auc_score(all_labels, all_predict)

            print(f'Epoch [{epoch + 1}/{epochs}], Test Accuracy: {accuracy}')
            print(f'Epoch [{epoch + 1}/{epochs}], Test Recall: {recall}')
            print(f'Epoch [{epoch + 1}/{epochs}], Test Precision: {precision}')
            print(f'Epoch [{epoch + 1}/{epochs}], Test F1 Score: {f1}')
            print(f'Epoch [{epoch + 1}/{epochs}], Test AUROC: {auroc}')
            print(f'Epoch [{epoch + 1}/{epochs}], Test AUPRC: {auprc}')

            if auprc > best_auprc:
                if run == 0:
                    best_predict = all_predict
                    best_labels = all_labels
                best_auroc = auroc
                # best_auroc = auroc
                best_recall = recall
                best_precision = precision
                best_f1 = f1
                best_auprc = auprc
                best_accuracy = accuracy

        accuracy_values.append(best_accuracy)
        recall_values.append(best_recall)
        precision_values.append(best_precision)
        f1_values.append(best_f1)
        auroc_values.append(best_auroc)
        auprc_values.append(best_auprc)
        print(f'Run {run + 1} Best AUROC: {best_auroc}')
        # torch.save(model.state_dict(), "base_model1.pth")
        #     if auroc > best_auroc:
        #         best_auroc = auroc
        # print(f'Best AUROC: {best_auroc}')  # 打印训练过程中的最佳 AUROC 值

        # 计算所有epoch的平均值
    mean_accuracy = sum(accuracy_values) / len(accuracy_values)
    mean_recall = sum(recall_values) / len(recall_values)
    mean_precision = sum(precision_values) / len(precision_values)
    mean_f1 = sum(f1_values) / len(f1_values)
    mean_auroc = sum(auroc_values) / len(auroc_values)
    mean_auprc = sum(auprc_values) / len(auprc_values)
    # 输出最终的平均值
    print(f'Average Accuracy: {mean_accuracy}')
    print(f'Average Recall: {mean_recall}')
    print(f'Average Precision: {mean_precision}')
    print(f'Average F1 Score: {mean_f1}')
    print(f'Average AUROC: {mean_auroc}')
    print(f'Average AUPRC: {mean_auprc}')
    np.savetxt('mamba-mimicIII-pre.txt', best_predict)
    np.savetxt('mamba-mimicIII-label.txt', best_labels)