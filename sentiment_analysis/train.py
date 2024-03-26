import torch
import torch.nn as nn
from torch import optim
from model import SA_LSTM
from dataset import Text_Cls, data_loader
from configs import Config

cfg = Config()
# dataset
train_data_path = "./sources/trainset.csv"
voc_dict_path = "./sources/dict.csv"
dataset = Text_Cls(voc_dict_path, train_data_path)
train_dataloader = data_loader(dataset, cfg)
# model
cfg.pad_size = dataset.max_seq_len
model_text_cls = SA_LSTM(cfg)
model_text_cls.to(cfg.devices)
# loss
loss_func = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

# train
print("Start Training...\n")
for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.devices)
        label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        print("epoch is {}, ite is {}, loss is {}".format(epoch, i, loss_val))
        loss_val.backward()
        optimizer.step()
    scheduler.step()
    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), "./models/{}.pth".format(epoch))
