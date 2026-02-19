from src.train import train
from src.encoding import to_oh_tensor
from src.data_prepare import read
from src.model import Model
from src.heatmap import heatmap
from src.loss import Cross_entropy

import pandas as pd
from Bio import SeqIO


def main():
    wt_path = ""
    # 新建一个用于补全空缺和作图的野生型序列.fasta文件，文件路径补全在上方
    pos_path = ""
    neg_path = ""
    # 正例和负例.fasta文件分别保存，并将路径填入上方

    wt_seq = str(next(SeqIO.parse(wt_path, "fasta")).seq)
    seq_len = len(wt_seq)
    save_path = "./01_weight_pth"

    pos = read(pos_path, 1, wt_path)
    neg = read(neg_path, 0, wt_path)
    records = pos + neg
    df = pd.DataFrame(records)
    X,y = to_oh_tensor(df)

    model = Model(seq_len, "PROTEIN")
    train(model, X, y, save_path, lr=1e-3, epochs=1000, Loss = Cross_entropy, tv_reg=True)
    # 可调整的超参数主要是epochs训练轮数，tv_reg是否加入全变分正则化，损失函数曲线自动绘制，损失函数设置为交叉熵函数

    heatmap(model, "PROTEIN", wt_seq = wt_seq, ref_seq="*"*seq_len)
    # wt_seq主要传递野生型序列，用于在图上标识（黑色圆点），ref_seq是用于求差值作图的，最好取wt_seq或"*"*seq_len

if __name__ == "__main__":
    main()