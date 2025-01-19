
# 绘制条形图
# plt.bar(range(len(frequencies)), frequencies)
# plt.xticks(range(len(frequencies)), frequencies, rotation=90)
# plt.xlabel("Frequency Index")
# plt.ylabel("Frequency (Hz)")
# plt.title("Frequency Index Distribution")
# plt.show()

# 绘制折线图
import matplotlib.pyplot as plt

# 选取的频率索引
frequencies = [0, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 3264, 4096, 4480, 5120, 6400]

# 绘制折线图
plt.plot(frequencies, marker='o', linestyle='-', color='b')

# 在每个数据点旁边显示数值
for i, freq in enumerate(frequencies):
    plt.text(i, freq, str(freq), ha='center', va='bottom', fontsize=9)

plt.xlabel("Frequency Index")
plt.ylabel("Frequency (Hz)")
plt.title("Frequency index Curve")
plt.grid(True)
plt.savefig("Frequency_index.pdf")
plt.show()

