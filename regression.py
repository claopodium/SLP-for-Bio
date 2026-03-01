from src.train import train
from src.encoding import to_oh_tensor
from src.data_prepare import continuous_read, extract
from src.model.slp import Model
from src.heatmap import heatmap

import pandas as pd
from Bio import SeqIO
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="SLP-for-Bio train")

    parser.add_argument("--wt_path", "-wt", type = str, required=False, help = "wild type sequence")
    parser.add_argument("--data", "-d", type = str, required=True, help = "dataset")
    parser.add_argument("--save_path", "-s", type = str, required=True, help = "save path of weight file")
    parser.add_argument("--length", "-l", type = str, required=True, help = "length of sequence")
    
    # 新建一个用于补全空缺和作图的野生型序列.fasta文件
    # 将所有输入数据保存为一个.fasta文件，要求为fasta格式，同时在每个样本后补上表型标签，形如：
    # > name1 label=1.00
    # ATCGATCGATCG......
    # > name2 label=1.05
    # AGTCAGCTACGT......
    # label=1.00之间不要空格

    args = parser.parse_args()
    wt_path = args.wt_path
    save_path = args.save_path
    data_path = args.data
    seq_len = int(args.length)
    
    if wt_path:
        wt_seq = str(next(SeqIO.parse(wt_path, "fasta")).seq)
    else:
        wt_seq = None
        
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".fasta":
        records = continuous_read(data_path, wt_path)
        df = pd.DataFrame(records)
    elif ext == ".csv":
        df =pd.read_csv(data_path)
        df = pd.DataFrame(extract(df))

    X,y = to_oh_tensor(df)

    model = Model(seq_len, "PROTEIN")
    train(model, X, y, save_path, lr=1e-3, epochs=10, tv_reg=True)
    # 可调整的超参数主要是epochs训练轮数，tv_reg是否加入全变分正则化，损失函数曲线自动绘制

    heatmap(model, "PROTEIN", wt_seq = wt_seq, ref_seq="*"*seq_len)
    # wt_seq主要传递野生型序列，用于在图上标识（黑色圆点），ref_seq是用于求差值作图的，最好取wt_seq或"*"*seq_len

if __name__ == "__main__":

    main()
