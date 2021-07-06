import paddlehub as hub
import pandas as pd

if __name__ == '__main__':
    file_path = "data/weibo_senti_100k/test.tsv"
    text = pd.read_csv(file_path, sep="\t", header=None)
    label_map = {0: 0, 1: 1}  # {0: 'negative', 1: 'positive'}
    data = [[i] for i in text[1]]

    model = hub.Module(name='ernie_tiny', task='seq-cls',
                       load_checkpoint='./ernie_checkpoint/best_model/model.pdparams', label_map=label_map)
    results = model.predict(data, max_seq_len=128, batch_size=1, use_gpu=True)

    # 输出测试集准确率
    count = 0
    for i, j in zip(text[0], results):
        # print(type(i), type(j))
        if int(i) == int(j):
            count += 1
    print("测试集准确率:", count / len(results))
