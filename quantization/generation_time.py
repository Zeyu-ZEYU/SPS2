import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# 设置数据
labels = ["2K", "4K", "8K", "16K", "32K", "64K"]


quant_dequant = np.array([0.3, 0.35, 0.58, 1.1, 2.2, 4.2])
# quant_attn = np.array([0.61, 0.65, 0.89, 1.2, 2, 3.7])
quant_qkv_gen = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
quant_qk = np.array([0.28, 0.3, 0.42, 0.58, 0.97, 1.79])
quant_softmax = np.array([0.028, 0.03, 0.042, 0.058, 0.097, 0.179])
quant_v = np.array([0.24, 0.26, 0.39, 0.53, 0.92, 1.71])
quant_dense = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
quant_dropout = np.array([0.11, 0.1, 0.11, 0.12, 0.12, 0.11])
quant_layernorm = np.array([0.2, 0.2, 0.22, 0.22, 0.2, 0.22])
quant_mlp = np.array([0.18, 0.17, 0.2, 0.19, 0.18, 0.21])
quant_mem = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])


lut_dequant = np.array([0.32, 0.53, 0.98, 1.88, 3.65, 7.2])
# lut_attn = np.array([0.61, 0.65, 0.89, 1.2, 2, 3.7])
lut_qkv_gen = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
lut_qk = np.array([0.28, 0.3, 0.42, 0.58, 0.97, 1.79])
lut_softmax = np.array([0.028, 0.03, 0.042, 0.058, 0.097, 0.179])
lut_v = np.array([0.24, 0.26, 0.39, 0.53, 0.92, 1.71])
lut_dense = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
lut_dropout = np.array([0.11, 0.1, 0.11, 0.12, 0.12, 0.11])
lut_layernorm = np.array([0.2, 0.2, 0.22, 0.22, 0.2, 0.22])
lut_mlp = np.array([0.18, 0.17, 0.2, 0.19, 0.18, 0.21])
lut_mem = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])


qk_dequant = np.array([0.15, 0.17, 0.28, 0.53, 1.08, 2.06])
qk_qkv_gen = np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
qk_qk = np.array([0.28, 0.3, 0.42, 0.58, 0.97, 1.79])
qk_qk_dequant = np.array([0.032, 0.035, 0.046, 0.063, 0.103, 0.185])
qk_softmax = np.array([0.028, 0.03, 0.042, 0.058, 0.097, 0.179])
qk_v = np.array([0.24, 0.26, 0.39, 0.53, 0.92, 1.71])
qk_dense = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
qk_dropout = np.array([0.11, 0.1, 0.11, 0.12, 0.12, 0.11])
qk_layernorm = np.array([0.2, 0.2, 0.22, 0.22, 0.2, 0.22])
qk_mlp = np.array([0.18, 0.17, 0.2, 0.19, 0.18, 0.21])
qk_mem = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.04])


x = np.arange(len(labels))  # 标签位置
width = 0.3  # 柱状图的宽度

fig, ax = plt.subplots(figsize=(13, 7.5))


ax.bar(x - width, quant_mem, width, label="Memory loading", color="darkgoldenrod", edgecolor="k")
ax.bar(x - width, quant_dequant, width, label="Decompression", color="red", edgecolor="k", bottom=quant_mem)
ax.bar(
    x - width,
    quant_qkv_gen,
    width,
    label="QKV generation",
    color="lightgreen",
    edgecolor="k",
    bottom=quant_mem + quant_dequant,
)
ax.bar(
    x - width,
    quant_qk,
    width,
    label="QK multiplication",
    color="deepskyblue",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen,
)
ax.bar(
    x - width,
    quant_softmax,
    width,
    label="Softmax",
    color="violet",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen + quant_qk,
)
ax.bar(
    x - width,
    quant_v,
    width,
    label="V multiplication",
    color="forestgreen",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen + quant_qk + quant_softmax,
)
ax.bar(
    x - width,
    quant_dense,
    width,
    label="Attention linear",
    color="k",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen + quant_qk + quant_softmax + quant_v,
)
ax.bar(
    x - width,
    quant_dropout,
    width,
    label="Dropout",
    color="gold",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen + quant_qk + quant_softmax + quant_v + quant_dense,
)
ax.bar(
    x - width,
    quant_layernorm,
    width,
    label="Layernorm",
    color="purple",
    edgecolor="k",
    bottom=quant_mem + quant_dequant + quant_qkv_gen + quant_qk + quant_softmax + quant_v + quant_dense + quant_dropout,
)
ax.bar(
    x - width,
    quant_mlp,
    width,
    label="MLP",
    color="tab:gray",
    edgecolor="k",
    bottom=quant_mem
    + quant_dequant
    + quant_qkv_gen
    + quant_qk
    + quant_softmax
    + quant_v
    + quant_dense
    + quant_dropout
    + quant_layernorm,
)


ax.bar(x, lut_mem, width, color="darkgoldenrod", edgecolor="k")
ax.bar(x, lut_dequant, width, color="red", edgecolor="k", bottom=lut_mem)
ax.bar(
    x,
    lut_qkv_gen,
    width,
    color="lightgreen",
    edgecolor="k",
    bottom=lut_mem + lut_dequant,
)
ax.bar(
    x,
    lut_qk,
    width,
    color="deepskyblue",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen,
)
ax.bar(
    x,
    lut_softmax,
    width,
    color="violet",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk,
)
ax.bar(
    x,
    lut_v,
    width,
    color="forestgreen",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk + lut_softmax,
)
ax.bar(
    x,
    lut_dense,
    width,
    color="k",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk + lut_softmax + lut_v,
)
ax.bar(
    x,
    lut_dropout,
    width,
    color="gold",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk + lut_softmax + lut_v + lut_dense,
)
ax.bar(
    x,
    lut_layernorm,
    width,
    color="purple",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk + lut_softmax + lut_v + lut_dense + lut_dropout,
)
ax.bar(
    x,
    lut_mlp,
    width,
    color="tab:gray",
    edgecolor="k",
    bottom=lut_mem + lut_dequant + lut_qkv_gen + lut_qk + lut_softmax + lut_v + lut_dense + lut_dropout + lut_layernorm,
)


ax.bar(x + width, qk_mem, width, color="darkgoldenrod", edgecolor="k")
ax.bar(x + width, qk_dequant, width, color="red", edgecolor="k", bottom=qk_mem)
ax.bar(
    x + width,
    qk_qkv_gen,
    width,
    color="lightgreen",
    edgecolor="k",
    bottom=qk_mem + qk_dequant,
)
ax.bar(
    x + width,
    qk_qk,
    width,
    color="deepskyblue",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen,
)
ax.bar(
    x + width,
    qk_softmax,
    width,
    color="violet",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk,
)
ax.bar(
    x + width,
    qk_v,
    width,
    color="forestgreen",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax,
)
ax.bar(
    x + width,
    qk_dense,
    width,
    color="k",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax + qk_v,
)
ax.bar(
    x + width,
    qk_dropout,
    width,
    color="gold",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax + qk_v + qk_dense,
)
ax.bar(
    x + width,
    qk_layernorm,
    width,
    color="purple",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax + qk_v + qk_dense + qk_dropout,
)
ax.bar(
    x + width,
    qk_mlp,
    width,
    color="tab:gray",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax + qk_v + qk_dense + qk_dropout + qk_layernorm,
)
ax.bar(
    x + width,
    qk_qk_dequant,
    width,
    label="Score dequantization",
    color="blue",
    edgecolor="k",
    bottom=qk_mem + qk_dequant + qk_qkv_gen + qk_qk + qk_softmax + qk_v + qk_dense + qk_dropout + qk_layernorm + qk_mlp,
)


ax.text(-0.65, 10.5, "Left bar: simple quantization", fontsize=30)
ax.text(-0.65, 8.8, "Middle bar: lookup table", fontsize=30)
ax.text(-0.65, 7.1, "Right bar: computed on compressed K", fontsize=30)

# plt.annotate("", xy=(-0.2, 8), xytext=(-0.3, 20), arrowprops=dict(facecolor="black", shrink=0.05, headwidth=10, width=3))
# plt.annotate("", xy=(0.2, 9), xytext=(0.6, 17), arrowprops=dict(facecolor="black", shrink=0.05, headwidth=10, width=3))

# 定义一个函数来格式化刻度标签，加上 "秒"
# from matplotlib.ticker import FuncFormatter


# def format_ticks(x, pos):
#     return f"{int(x)}s"


# formatter = FuncFormatter(format_ticks)
# ax.yaxis.set_major_formatter(formatter)
ax.set_yticklabels([f"{int(tick)}" for tick in ax.get_yticks()], fontsize=35)
ax.yaxis.set_major_locator(MultipleLocator(2))
fig.subplots_adjust(left=0.12, right=0.92, top=0.6, bottom=0.18)

# 添加一些文本标签
ax.set_xlabel("Prompt length", fontsize=40)
ax.set_ylabel("Generation time (s)", fontsize=40)
# ax.set_title("Delays by label and component")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=35, rotation=0)
ax.legend(
    fontsize=35,
    loc="lower center",
    bbox_to_anchor=(0.52, 0.9),
    ncol=2,
    frameon=False,
    columnspacing=0.2,
    labelspacing=-0.05,
    handletextpad=0.1,
)

plt.show()
