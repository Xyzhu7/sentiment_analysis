import torch
from model import SA_LSTM
from dataset import Text_Cls, data_loader
from configs import Config


cfg = Config()
# dataset
test_data_path = "./sources/testset.csv"
voc_dict_path = "./sources/dict.csv"
dataset = Text_Cls(voc_dict_path, test_data_path)
test_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_seq_len
model_text_cls = SA_LSTM(cfg)
model_text_cls.to(cfg.devices)
model_text_cls.load_state_dict(torch.load("./models/90.pth"))

out_sum = 0
for i, batch in enumerate(test_dataloader):
    label, data = batch
    data = torch.tensor(data).to(cfg.devices)
    label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
    pred_softmax = model_text_cls.forward(data)
    pred = torch.argmax(pred_softmax, dim=1)
    out = torch.eq(pred, label)
    accuracy = out.sum() * 100.0 / pred.size()[0]
    print("For batch {}, accuracy is {}%".format(i,accuracy))
    out_sum += out.sum()
accuracy_t = out_sum *100.0 / dataset.__len__()
print("Total accuracy is {}%".format(accuracy_t))