import paddle
import paddlehub as hub
import ast
import argparse
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset


class MyDataset(TextClassificationDataset):
    # 数据集存放目录
    base_path = 'data/weibo_senti_100k'
    # 数据集的标签列表，多分类标签格式为['0', '1', '2', '3',...]
    label_list = ['0', '1']

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'train.tsv'
        elif mode == 'test':
            data_file = 'test.tsv'
        else:
            data_file = 'dev.tsv'
        super().__init__(
            base_path=self.base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=self.label_list,
            is_file_with_header=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
    parser.add_argument("--use_gpu", type=ast.literal_eval, default=True,
                        help="Whether use GPU for fine-tuning, input should be True or False")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Number of words of the longest seqence.")
    parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
    parser.add_argument("--checkpoint_dir", type=str, default='./ernie_checkpoint',
                        help="Directory to model checkpoint")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every n epoch.")
    args = parser.parse_args()

    # 选择模型、任务和类别数
    model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=len(MyDataset.label_list))

    train_dataset = MyDataset(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode='train')
    dev_dataset = MyDataset(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode='dev')
    test_dataset = MyDataset(tokenizer=model.get_tokenizer(), max_seq_len=args.max_seq_len, mode='test')

    optimizer = paddle.optimizer.Adam(learning_rate=args.learning_rate, parameters=model.parameters())
    trainer = hub.Trainer(model, optimizer, checkpoint_dir=args.checkpoint_dir, use_gpu=args.use_gpu)
    trainer.train(train_dataset, epochs=args.num_epoch, batch_size=args.batch_size, eval_dataset=dev_dataset,
                  save_interval=args.save_interval)
    # 在测试集上评估当前训练模型
    trainer.evaluate(test_dataset, batch_size=args.batch_size)
