# paddlehub_ernie_emotion_analysis
百度PaddleHub-ERNIE微调中文情感分析(文本分类)
## PaddlePaddle-PaddleHub
[**飞桨(PaddlePaddle)**](https://github.com/PaddlePaddle/Paddle)以百度多年的深度学习技术研究和业务应用为基础，是中国首个自主研发、功能完备、 开源开放的产业级深度学习平台，集深度学习核心训练和推理框架、基础模型库、端到端开发套件和丰富的工具组件于一体。[**PaddleHub**](https://github.com/PaddlePaddle/PaddleHub)旨在为开发者提供丰富的、高质量的、直接可用的预训练模型。
## ERNIE
[**ERNIE**](https://github.com/PaddlePaddle/ERNIE)(**Enhanced** **Representation** **through** **kNowledge** **IntEgration**)是百度提出的**知识增强的语义表示模型**，通过对词、实体等语义单元的掩码，使得模型学习完整概念的语义表示。在**语言推断**、**语义相似度**、**命名实体识别**、**情感分析**、**问答匹配**等自然语言处理(NLP)各类**中文**任务上的验证显示，模型效果全面**超越BERT**。
![ERNIE](https://github.com/zhenhao-huang/paddlehub_ernie_emotion_analysis/blob/main/pictures/ernie1.png)
![ERNIE](https://github.com/zhenhao-huang/paddlehub_ernie_emotion_analysis/blob/main/pictures/ernie2.png)
更多详情请参考[**ERNIE论文**](https://arxiv.org/pdf/1904.09223.pdf)。
## 一、环境安装
    # CPU
    pip install paddlepaddle
    # GPU
    pip install paddlepaddle-gpu
    pip install paddlehub
有gpu的，建议安装**paddlepaddle-gpu**版(训练速度会提升好几倍)。**paddlepaddle-gpu**默认安装的是**cuda10.2**，如果需要安装其他cuda版本，到[**官方网站**](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)查找命令。(注意，从**1.8.0**开始采用**动态图**，所以**paddlepaddle**和**paddlehub**版本最好从**1.8.0**开始使用。)
## 二、数据预处理
这里使用的数据是**二分类数据集weibo_senti_100k.csv**，即情感倾向只有**正向**和**负向**，下载地址:[https://github.com/SophonPlus/ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)，已存放至**data/weibo_senti_100k**该目录下。由于**PaddleHub**用的是**tsv**格式的数据集，所以需要运行`to_tsv.py`该脚本将**csv**格式转成**tsv**格式。
## 三、微调
运行`finetune_ernie.py`。使用自己的数据集，需要修改`base_path`数据存放目录，`label_list`修改为实际的数据集标签。
### 选择模型
    model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=len(MyDataset.label_list))
* **name**：模型名称，可以选择ernie，ernie_tiny，bert-base-cased， bert-base-chinese, roberta-wwm-ext，roberta-wwm-ext-large等。
* **version**：module版本号
* **task**：fine-tune任务。**seq-cls**(文本分类任务)或**token-cls**(序列标注任务)。
* **num_classes**：表示当前文本分类任务的类别数，根据具体使用的数据集确定，默认为2。
PaddleHub还提供BERT等模型可供选择, 当前支持文本分类任务的模型对应的加载示例如下： 

|模型名|PaddleHub Module|
|--|--|
|ERNIE, Chinese|hub.Module(name='ernie')|
|ERNIE tiny, Chinese|hub.Module(name='ernie_tiny')|
|ERNIE 2.0 Base, English|hub.Module(name='ernie_v2_eng_base')|
|ERNIE 2.0 Large, English|hub.Module(name='ernie_v2_eng_large')|
|BERT-Base, English Cased|hub.Module(name='bert-base-cased')|
|BERT-Base, English Uncased|hub.Module(name='bert-base-uncased')|
|BERT-Large, English Cased|hub.Module(name='bert-large-cased')|
|BERT-Large, English Uncased|hub.Module(name='bert-large-uncased')|
|BERT-Base, Multilingual Cased|hub.Module(nane='bert-base-multilingual-cased')|
|BERT-Base, Multilingual Uncased|hub.Module(nane='bert-base-multilingual-uncased')|
|BERT-Base, Chinese|hub.Module(name='bert-base-chinese')|
|BERT-wwm, Chinese|hub.Module(name='chinese-bert-wwm')|
|BERT-wwm-ext, Chinese|hub.Module(name='chinese-bert-wwm-ext')|
|RoBERTa-wwm-ext, Chinese|hub.Module(name='roberta-wwm-ext')|
|RoBERTa-wwm-ext-large, Chinese|hub.Module(name='roberta-wwm-ext-large')|
|RBT3, Chinese|hub.Module(name='rbt3')|
|RBTL3, Chinese|hub.Module(name='rbtl3')|
|ELECTRA-Small, English|hub.Module(name='electra-small')|
|ELECTRA-Base, English|hub.Module(name='electra-base')|
|ELECTRA-Large, English|hub.Module(name='electra-large')|
|ELECTRA-Base, Chinese|hub.Module(name='chinese-electra-base')|
|ELECTRA-Small, Chinese|hub.Module(name='chinese-electra-small')|
### 选择优化策略和运行配置
    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir=args.checkpoint_dir, use_gpu=args.use_gpu)
    trainer.train(train_dataset, epochs=args.num_epoch, batch_size=args.batch_size, eval_dataset=dev_dataset,
                  save_interval=args.save_interval)
    # 在测试集上评估当前训练模型
    trainer.evaluate(test_dataset, batch_size=args.batch_size)
#### 优化策略
**Paddle**提供了多种优化器选择，如**SGD**，**Adam**，**Adamax**等，详细参见[策略](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/Overview_cn.html)。其中**Adam**:
* **learning_rate**：全局学习率。默认为1e-3；
* **parameters**：待优化模型参数。
#### 运行配置
**hub.Trainer**主要控制**Fine-tune**的训练，包含以下可控制的参数: 
* **model**：被优化模型；
* **optimizer**：优化器选择；
* **checkpoint_dir**：保存模型参数的地址；
* **use_gpu**：是否使用gpu，默认为False。对于GPU用户，建议开启use_gpu。

**trainer.train**主要控制具体的训练过程，包含以下可控制的参数：
* **train_dataset**：训练时所用的数据集；
* **epochs**：训练轮数；
* **batch_size**：训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
* **num_workers**：works的数量，默认为0；
* **eval_dataset**：验证集；
* **log_interval**：打印日志的间隔， 单位为执行批训练的次数。
* **save_interval**：保存模型的间隔频次，单位为执行训练的轮数。
## 四、模型预测
完成**Fine-tune**后，**Fine-tune**过程在验证集上表现最优的模型会被保存在`${CHECKPOINT_DIR}/best_model`目录下，其中`${CHECKPOINT_DIR}`目录为**Fine-tune**时所选择的保存**checkpoint**的目录。运行脚本`predict.py`。
## 五、结果
训练集：
![](https://github.com/zhenhao-huang/paddlehub_ernie_emotion_analysis/blob/main/pictures/result1.png)
测试集：
![](https://github.com/zhenhao-huang/paddlehub_ernie_emotion_analysis/blob/main/pictures/result2.png)
在**二分类数据集weibo_senti_100k.csv**上，**训练集准确率**可以达到**98%**，**测试集准确率**同样可以达到**98%**。
