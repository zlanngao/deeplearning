# deeplearning
5.3.3 实验：利用LSTM模型生成古诗
1)清洗数据：generate_poetry.py
直接从该网址下载训练数据：
http://tensorflow-1253675457.cosgz.myqcloud.com/poetry/poetry
数据中的每首唐诗以 [ 开头、] 结尾，后续生成古诗时，根据 [ 随机取一个字，根据 ] 判断是否结束。
两种词袋：“汉字 => 数字”、“数字 => 汉字”，根据第一个词袋将每首古诗转化为数字表示。
诗歌的生成是根据上一个汉字生成下一个汉字，所以 x_batch 和 y_batch 的 shape 是相同的，y_batch 是 x_batch 中每一位向前循环移动一位。前面介绍每首唐诗 [开头、] 结尾，在这里也体现出好处，] 下一个一定是 [（即一首诗结束下一首诗开始）
具体可以看下面例子：
x_batch：['[', 12, 23, 34, 45, 56, 67, 78, ']']
y_batch：[12, 23, 34, 45, 56, 67, 78, ']', '[']
在/home/ubuntu目录下创建源文件generate_poetry.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.3.3/generate_poetry.py
操作过程
在终端执行：
启动 python：
python
构建数据：
from generate_poetry import Poetry
p = Poetry()
查看第一首唐诗数字表示（[查看输出]）：
print(p.poetry_vectors[0])
根据 ID 查看对应的汉字（[查看输出]）：
print(p.id_to_word[1101])
根据汉字查看对应的数字（[查看输出]）：
print(p.word_to_id[u"寒"])
查看 x_batch、y_batch（[查看输出]）：
x_batch, y_batch = p.next_batch(1)
x_batch
y_batch
2）LSTM 模型学习—poetry_model.py
在模型训练过程中，需要对每个字进行向量化，Embedding 的作用按照 inputs 顺序返回 embedding 中的对应行，类似：
import numpy as np
embedding = np.random.random([100, 10]) 
inputs = np.array([7, 17, 27, 37])
print(embedding[inputs])
在/home/ubuntu目录下创建源文件 poetry_model.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.3.3/poetry_model.py
3）训练 LSTM 模型—poetry_model.py
每批次采用50首唐诗训练，训练40000次后，损失函数基本保持不变，GPU 大概需要 2 个小时左右。当然也可以调整循环次数，节省训练时间，或者直接下载已经训练好的模型。
wget http://tensorflow-1253675457.cosgz.myqcloud.com/poetry/poetry_model.zip
unzip poetry_model.zip
在 /home/ubuntu 目录下创建源文件 train_poetry.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.3.3/train_poetry.py
然后执行（如果已下载模型，可以省略此步骤）:
cd /home/ubuntu;
python train_poetry.py
4）模型测试—predict_poetry.py
根据[随机取一个汉字，作为生成古诗的第一个字，遇到]结束生成古诗。
在/home/ubuntu目录下创建源文件predict_poetry.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.3.3/predict_poetry.py
然后执行:
cd /home/ubuntu;
python predict_poetry.py
执行结果：
风雨满风吹日夜，不同秋草不归情。山风欲见寒山水，山外寒流雨半风。夜日春光犹见远，一时相思独伤情。自应未肯为心客，独与江南去故乡。

5.4.3 实验：基于Seq2Seq模型的聊天机器人
实验内容介绍：
基于 TensoFlow 构建 Seq2Seq 模型，并加入 Attention 机制，encoder 和 decoder 为 3 层的 RNN 网络。
实验步骤：
1.清洗数据、提取 ask 数据和 answer 数据、提取词典、为每个字生成唯一的数字 ID、ask 和 answer 用数字 ID 表示；
2.TensorFlow中Translate Demo，由于出现deepcopy错误，这里对Seq2Seq稍微改动了；
3.训练 Seq2Seq 模型；
4.进行聊天。
详细步骤：
1）清洗数据：generate_chat.py
获取训练数据：
http://devlab-1251520893.cos.ap-guangzhou.myqcloud.com/chat.conv
原始数据中，每次对话是 M 开头，前一行是 E ，并且每次对话都是一问一答的形式。将原始数据分为 ask、answer 两份数据；
两种词袋：“汉字 => 数字”、“数字 => 汉字”，根据第一个词袋将 ask、answer 数据转化为数字表示；
answer 数据每句添加 EOS 作为结束符号。
下载：https://github.com/zlanngao/deeplearning/blob/master/5.4.3/generate_chat.py
生成数据
启动 python：cd /home/ubuntu/
python
from generate_chat import *
获取 ask、answer 数据并生成字典：
get_chatbot()
train_encode - 用于训练的 ask 数据；
train_decode - 用于训练的 answer 数据；
test_encode - 用于验证的 ask 数据；
test_decode - 用于验证的 answer 数据；
vocab_encode - ask 数据词典；
vocab_decode - answer 数据词典。
训练数据转化为数字表示：
get_vectors()
train_encode_vec - 用于训练的 ask 数据数字表示形式；
train_decode_vec - 用于训练的 answer 数据数字表示形式；
test_encode_vec - 用于验证的 ask 数据；
test_decode_vec - 用于验证的 answer 数据；
2）模型学习—seq2seq.py、seq2seq_model.py
采用 translate 的 model，实验过程中会发现 deepcopy 出现 NotImplementedType 错误，所以对 translate 中 seq2seq 做了改动。
在/home/ubuntu 目录下创建源文件seq2seq.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.4.3/seq2seq.py
在/home/ubuntu目录下创建源文件seq2seq_model.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.4.3/seq2seq_model.py
3）训练模型—train_chat.py
训练 30 万次后，损失函数基本保持不变，单个 GPU 大概需要 17 个小时左右，如果采用 CPU 训练，大概需要 3 天左右。在训练过程中可以调整循环次数，体验下训练过程，可以直接下载已经训练好的模型。
在/home/ubuntu目录下创建源文件train_chat.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.4.3/train_chat.py
然后执行:
cd /home/ubuntu;
python train_chat.py
下载已有模型:
wget http://tensorflow-1253675457.cosgz.myqcloud.com/chat/chat_model.zip
unzip -o chat_model.zip
4）聊天测试—predict_chat.py
利用训练好的模型，我们可以开始聊天了。训练数据有限只能进行简单的对话，提问最好参考训练数据，否则效果不理想。
在 /home/ubuntu 目录下创建源文件 predict_chat.py，文件详细编码可在此地址下载：https://github.com/zlanngao/deeplearning/blob/master/5.4.3/predict_chat.py
然后执行（需要耐心等待几分钟）:
cd /home/ubuntu
python predict_chat.py
执行结果：
ask > 你好
answer > 你好呀
ask > 我是谁
answer > 哈哈，大屌丝，地地眼
