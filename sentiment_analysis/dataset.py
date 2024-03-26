from torch.utils.data import Dataset, DataLoader
import numpy as np
from configs import Config


# 定义读取词库操作
def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item = item.split(",")
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict


# 定义读取预处理的数据的操作
def load_data(data_path):
    ff = open(data_path, encoding='gbk')
    datalist = ff.readlines()
    max_len_seq = int(datalist[0])
    train_data_str = datalist[1:]
    train_data = []
    for idx, item in enumerate(train_data_str):
        train_data.append([int(item[0]), item.strip()[4:-2].split("', '")])
    return train_data, max_len_seq


# 定义Dataset类
class Text_Cls(Dataset):
    def __init__(self, voc_dict_path, train_data_path, max_len_seq=None):
        self.train_data_path = train_data_path
        self.voc_dict = read_dict(voc_dict_path)
        self.train_data, self.max_seq_len = load_data(self.train_data_path)
        if max_len_seq is not None:
            self.max_seq_len = max_len_seq

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item):
        train_data = self.train_data[item]
        label = int(train_data[0])
        word_list = train_data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])

        if len(input_idx) < self.max_seq_len:
            input_idx += [self.voc_dict["<PAD>"]
                        for _ in range(self.max_seq_len - len(input_idx))]
        train_data = np.array(input_idx)
        return label, train_data


def data_loader(dataset, config):
    return DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)

#测试
if __name__ == "__main__":
    train_data_path = "./sources/trainset.csv"
    voc_dict_path = "./sources/dict.csv"
    config = Config()
    dataset = Text_Cls(voc_dict_path, train_data_path)
    train_dataloader = data_loader(dataset, config=config)
    for i, batch in enumerate(train_dataloader):
        print(batch[1].size())
