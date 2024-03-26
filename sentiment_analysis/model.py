import torch
import torch.nn as nn
import torch.nn.functional as F


# 基于LSTM的SA model
class SA_LSTM(nn.Module):
    def __init__(self, config):
        super(SA_LSTM, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=config.num_vocab,
                                      embedding_dim=config.embed_size,
                                      padding_idx=config.num_vocab - 1)
        self.lstm = nn.LSTM(input_size=config.embed_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.maxpooling = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size, config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded_code = self.embedding(x)  # [batchsize, len_seq, embed_size]
        out, _ = self.lstm(embedded_code)  # [batchsize, len_seq, hidden_size*2]
        out = torch.cat((embedded_code, out), 2)  # [batchsize, len_seq, hidden_size*2+embed_size]
        out = F.relu(out)  # activation function
        out = out.permute(0, 2, 1)  # [batchsize,hidden_size*2+embed_size, len_seq]
        out = self.maxpooling(out).reshape(out.size()[0], -1)  # [batchsize,hidden_size*2+embed_size]
        out = self.fc(out)  # full connected layer
        out = self.softmax(out)  # output probability
        return out


if __name__ == "__main__":
    from configs import Config
    cfg = Config()
    model_textcls = SA_LSTM(config=cfg)
    input_tensor = torch.tensor([i for i in range(320)]).reshape([10, 32])
    out_tensor = model_textcls.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)
