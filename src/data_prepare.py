from Bio import SeqIO
import pandas as pd

def replace_gap_with_wt(seq, wt_path):
    wt_record = next(SeqIO.parse(wt_path, "fasta"))
    wt_seq = str(wt_record.seq)
    if len(seq) != len(wt_seq):
        raise ValueError("Sequence length does not match WT sequence length")
    #wt_seq = wt_seq.replace("*","")
    return "".join(
        wt_aa if aa == "-" else aa
        for aa, wt_aa in zip(seq, wt_seq)
    )

def read(fasta_path, label, wt_path = None):
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        new_seq = replace_gap_with_wt(str(record.seq), wt_path)
        #new_seq = new_seq.replace("*","")
        records.append({
            "id": record.id,
            "x": new_seq,
            "y": label
        })
    return records

def continuous_read(fasta_path,wt_path):
    records = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        desc = record.description

        label = desc.split("label=")[1]
        label = float(label)

        new_seq = replace_gap_with_wt(str(record.seq), wt_path)

        records.append({
            "id": record.id,
            "x": str(new_seq),
            "y": label
        })
    return records


def extract(df, x_col="x", y_col="y", wt_path=None):

    wt_seq = None
    if wt_path is not None:
        wt_record = next(SeqIO.parse(wt_path, "fasta"))
        wt_seq = str(wt_record.seq)
    
    new_records = []
    for idx, row in df.iterrows():
        seq = row[x_col]
        if wt_seq is not None:
            if len(seq) != len(wt_seq):
                raise ValueError(f"Sequence length {len(seq)} != WT length {len(wt_seq)}")
            seq = "".join(wt_aa if aa == "-" else aa for aa, wt_aa in zip(seq, wt_seq))
        
        new_records.append({
            "id": row.get("id", idx), 
            "x": seq,
            "y": row[y_col]
        })
    
    return new_records