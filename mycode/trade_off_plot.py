import matplotlib.pyplot as plt

mynet = {
    'MyNetwork': (57.82, 50.98)
}

alexnet = {
    'AlexNet': (61.10, 19.30)
}

vggnet = {
    'v_11': (132.86, 40.37),
    'v_13': (133.05, 40.55),
    'v_16': (138.36, 39.66),
    'v_19': (143.67, 38.25),
}

vggnet_bn = {
    'v_11_bn': (132.87, 46.44),
    'v_13_bn': (133.05, 47.46),
    'v_16_bn': (138.37, 46.77),
    'v_19_bn': (143.68, 46.27),
}

resnet = {
    'r_18': (11.69, 36.55),
    'r_34': (21.80, 34.05),
    'r_50': (25.56, 28.35),
    'r_101': (44.55, 27.25),
    'r_152': (60.19, 23.50),
}

densenet = {
    'd_121': (7.98, 35.15),
    'd_169': (14.15, 34.22),   
    'd_201': (20.01, 35.68),
    'd_161': (28.68, 38.41),
}

efficientnet = {
    'e_0': (5.29, 33.83),
    'e_1': (7.79, 31.28),
    'e_2': (9.11, 30.32),
    'e_3': (12.23, 31.21),
    'e_4': (19.34, 25.58),
    'e_5': (30.39, 20.11),
    'e_6': (43.04, 23.55),
    'e_7': (66.35, 8.64)
}

efficientnet_v2 = {
    'e2_s': (21.46, 33.16),
    'e2_m': (54.14, 29.26),
    'e2_l': (118.52, 16.53)
}

swin_transformer = {
    's_t': (28.29, 32.17),
    's_s': (49.61, 32.08),
    's_b': (87.77, 31.49),
}

swin_transformer_v2 = {
    's2_t': (28.35, 37.84),
    's2_s': (49.74, 36.62),
}

model_list = [alexnet, mynet, vggnet, vggnet_bn, resnet, densenet, efficientnet, efficientnet_v2, swin_transformer, swin_transformer_v2]
color_list = ['black', 'crimson', 'hotpink', 'orange', 'gold', 'green', 'lightgreen', 'skyblue', 'blue', 'blueviolet']
label_list = ['AlexNet', 'MyNetwork', 'VGGNet', 'VGGNet-BN', 'ResNet', 'DenseNet', 'EfficientNet', 'EfficientNet V2', 'Swin Transformer', 'Swin Transformer V2']

plt.figure(figsize=(10, 6))

for m, c, l in zip(model_list, color_list, label_list):
    x_e = [p for p, a in m.values()]
    y_e = [a for p, a in m.values()]
    plt.plot(x_e, y_e, marker='o', linestyle='-', color=c, label=l)

for model in model_list:
    for name, (xp, yp) in model.items():
        plt.text(xp, yp + 0.2, name, fontsize=9, ha='center')

plt.xlabel('Number of Parameters (Millions)')
plt.ylabel('Tiny-ImageNet Accuracy (%)')

plt.title('Size-Accuracy Trade-Off Plot')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right', bbox_to_anchor=(1.1, 0.1)) 

plt.tight_layout()
plt.show()
plt.savefig('mycode/trade_off_plot.png')