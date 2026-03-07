import matplotlib.pyplot as plt
import numpy as np

def heatmap(model, alphabet, wt_seq = None, ref_seq=None, chunk_size= 100):
    """
    绘制模型 theta 热图。
    
    model: 已训练的 AdditiveSequenceModel
    alphabet: str，氨基酸或碱基，例如 "PROTEIN"
    ref_seq: str，可选；参考序列，如果提供，则绘制 Δφ
    chunk_size: int，可选；如果提供，长序列分块绘制，每块 chunk_size 位点
    """

    theta = model.theta.detach().cpu().numpy()  # (L, A)
    L, A = theta.shape
    ALPHABETS = {
        "DNA": "ACGT",
        "RNA": "ACGU",
        "PROTEIN": "ACDEFGHIKLMNPQRSTVWY*"
        }
    alphabet = ALPHABETS[alphabet]
    aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}

    # 如果有参考序列，计算 Δφ
    if ref_seq is not None:
        aa_to_idx = {aa: i for i, aa in enumerate(alphabet)}
        base_vals = np.zeros(L)
        for l, aa in enumerate(ref_seq):
            base_vals[l] = theta[l, aa_to_idx.get(aa, 0)]
        theta_plot = theta - base_vals[:, None]
    else:
        theta_plot = theta

    # 支持分块
    if chunk_size is None:
        chunks = [(0, L)]
    else:
        chunks = [(i, min(i + chunk_size, L)) for i in range(0, L, chunk_size)]

    for i, (start, end) in enumerate(chunks, 1):
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(theta_plot[start:end, :].T, cmap="bwr", aspect="auto",
                       vmin=-np.max(np.abs(theta_plot)), vmax=np.max(np.abs(theta_plot)))

        # 横轴：位点
        positions = np.arange(start + 1, end + 1)
        step = max(1, len(positions) // 10)
        tick_idx = np.arange(0, len(positions), step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(positions[tick_idx])

        # 纵轴：氨基酸
        ax.set_yticks(np.arange(len(alphabet)))
        ax.set_yticklabels(list(alphabet))
        if wt_seq:
            for l, aa in enumerate(wt_seq[start:end]):
                if aa in aa_to_idx:
                    ax.scatter(l, aa_to_idx[aa], color="gray", s=30, zorder=5)


        ax.set_xlabel("Position")
        ax.set_ylabel("Amino Acid")
        ax.set_title(f"Weight Heatmap positions {start+1}-{end}")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("Weight", rotation=-90, labelpad=15)
        

        fig.tight_layout()

        # 保存文件
        fig.savefig(f"part_{i}_pos_{start+1}_{end}.png", dpi=300)

