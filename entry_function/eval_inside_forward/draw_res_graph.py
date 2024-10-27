from rouge import Rouge
import matplotlib.pyplot as plt


def cal_rouge_score(a, b):
    rouge = Rouge()
    scores_1 = rouge.get_scores(a, b, avg=True)
    return scores_1['rouge-l']['f'], scores_1['rouge-1']['f'], scores_1['rouge-2']['f']


def draw(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = [float(line.strip()) for line in file]
        indices = range(len(data))
        plt.plot(indices, data, marker='o')
        plt.ylim(0, 0.5)
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Plot of Floating Point Numbers")
        plt.grid(True)
        plt.show()


def cal_avg(file_path):
    to_cal = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            to_cal.append(float(line.strip()))
    print(f"{file_path}: {sum(to_cal) / len(to_cal)}")


if __name__ == "__main__":

    #cal_avg("buzz_multi_rouge_l.txt")
    # draw("full_multi_rouge_l.txt")
    #cal_avg("buzz_multi_rouge_1.txt")
    #cal_avg("buzz_multi_rouge_2.txt")
    # draw("full_multi_rouge_2.txt")
    cal_avg("cache_lens.txt")







