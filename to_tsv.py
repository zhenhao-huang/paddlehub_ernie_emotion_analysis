import pandas as pd

# 转成tsv格式
file_path = "data/weibo_senti_100k/weibo_senti_100k.csv"
text = pd.read_csv(file_path, sep=",")
text = text.sample(frac=1)  # 打乱数据集
# text.drop(['pred_v1', 'emotion_v0'], axis=1, inplace=True)
# cols = list(text)
# cols.insert(0, cols.pop(cols.index('label')))
# print(cols)
# text = text.loc[:, cols]
print(len(text))

train = text[:int(len(text) * 0.8)]
dev = text[int(len(text) * 0.8):int(len(text) * 0.9)]
test = text[int(len(text) * 0.9):]

train.to_csv('data/weibo_senti_100k/train.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
dev.to_csv('data/weibo_senti_100k/dev.tsv', sep='\t', header=None, index=False, columns=None, mode="w")
test.to_csv('data/weibo_senti_100k/test.tsv', sep='\t', header=None, index=False, columns=None, mode="w")

# 验证train,dev,test标签分布是否均匀
for file in ['train', 'dev', 'test']:
    file_path = f"data/weibo_senti_100k/{file}.tsv"
    text = pd.read_csv(file_path, sep="\t", header=None)
    prob = dict()
    total = len(text[0])
    for i in text[0]:
        if prob.get(i) is None:
            prob[i] = 1
        else:
            prob[i] += 1
    # 按标签排序
    prob = {i[0]: round(i[1] / total, 3) for i in sorted(prob.items(), key=lambda k: k[0])}
    print(file, prob, total)
