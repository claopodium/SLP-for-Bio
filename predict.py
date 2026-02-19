from src.prediction import predict

def main():
    weight_path = "./weight_path"

    while True:
        x = input("请输入序列（输入 'stop' 退出）：").strip()
        if x.lower() == "stop":
            break
        y = predict(weight_path, x, "PROTEIN")
        print("----------------")
        print(float(y))
        print("----------------")

if __name__ == "__main__":
    main()
