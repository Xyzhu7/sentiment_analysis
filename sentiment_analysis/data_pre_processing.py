import jieba
import re
from sklearn.model_selection import train_test_split

data_path = "./sources/weibo_senti_100k.csv"
stop_words_path = "./sources/hit_stopwords.txt"
# 停用词表处理
stop_words = open(stop_words_path, encoding='UTF-8').readlines()
stop_words_list = [line.strip() for line in stop_words]
stop_words_list.append(" ")
stop_words_list.append("\n")

# 针对数据的特性（微博原文）的清理
ban_str1 = re.compile("//@.*?[:：]")
ban_str2 = re.compile("回复@.*?[:：]")
ban_str3 = re.compile("@.*?\s")  # 原数据常见的用户名的出现形式，和语义无关
ban_str4 = re.compile("http://.*?\s")  # 原数据常见的链接的出现形式，和语义无关
ban_str_list = [ban_str1, ban_str2, ban_str3, ban_str4]
# 数据处理
voc_dict = {}  # 存放词库
max_len_seq = 0  # 最大数据长度，不足的数据用pad补齐
data = []  # 处理后的数据
data_list = open(data_path, encoding='UTF-8').readlines()[1:]  # 第一行（label,review）不是有效数据
for item in data_list:
    label = item[0]
    content = item[2:]  # 内容形式：label（0 or 1）,content
    for ban_str in ban_str_list:
        content = re.sub(ban_str, '', content)  # 清洗掉用户名和链接
    content = content.strip()
    seg_list = jieba.cut(content, cut_all=False)
    seg_res = []
    for seg_item in seg_list:
        if seg_item in stop_words_list:
            continue
        else:
            seg_res.append(seg_item)
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1  # 构建词库
    if len(seg_res) > max_len_seq:
        max_len_seq = len(seg_res)
    data.append([label, seg_res])
# 测试集和训练集
trainset,testset = train_test_split(data, test_size=0.2, random_state=42)
# 写入train_data
ff = open("./sources/trainset.csv", "w")
ff.write("{}\n".format(max_len_seq))
for item in trainset:
    ff.writelines("{},{}\n".format(int(item[0]), item[1]))
ff.close()
# 写入test_data
ff = open("./sources/testset.csv", "w")
ff.write("{}\n".format(max_len_seq))
for item in testset:
    ff.writelines("{},{}\n".format(int(item[0]), item[1]))
ff.close()
# 取前top_n词频的词，且要求至少出现min_seq次以上
min_seq = 1
top_n = 2000
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq], key=lambda x: x[1], reverse=True)[:top_n]
# 按照限制词频的词更新词库
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}
# 加上UNK和PAD
voc_dict.update({"<UNK>": len(voc_dict), "<PAD>": len(voc_dict) + 1})
# 把词库写入文件
ff = open("./sources/dict.csv", "w")
for key, value in voc_dict.items():
    ff.writelines("{},{}\n".format(key, value))
ff.close()
