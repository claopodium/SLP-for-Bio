import matplotlib.pyplot as plt

def loss_plt(loss_list):
    plt.figure(figsize=(6,4))
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()
    plt.savefig("./loss_plot")
