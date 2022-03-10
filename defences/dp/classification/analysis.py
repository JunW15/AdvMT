import matplotlib.pyplot as plt
import numpy as np


def log_grads_vars(log_file_path):
    p_g, p_v, c_g, c_v = [], [], [], []
    with open(log_file_path) as f:
        for line in f:
            line = line.strip().split()
            step = int(line[3])
            name = str(line[4])
            value = float(line[-1])
            if name == 'p_g':
                p_g.append((step, value))
            elif name == 'p_v':
                p_v.append((step, value))
            elif name == 'c_g':
                c_g.append((step, value))
            elif name == 'c_v':
                c_v.append((step, value))

    return p_g, p_v, c_g, c_v


def compare_g():
    p_g, p_v, c_g, c_v = log_grads_vars('logs/100-differential-privacy-norm')
    dp_p_g, dp_p_v, dp_c_g, dp_c_v = log_grads_vars('logs/dp-100-differential-privacy-n-0.03-c-0.03-m-32-norm')

    p_g_x, p_g_y = zip(*p_g)
    p_v_x, p_v_y = zip(*p_v)
    c_g_x, c_g_y = zip(*c_g)
    c_v_x, c_v_y = zip(*c_v)

    dp_p_g_x, dp_p_g_y = zip(*dp_p_g)
    dp_p_v_x, dp_p_v_y = zip(*dp_p_v)
    dp_c_g_x, dp_c_g_y = zip(*dp_c_g)
    dp_c_v_x, dp_c_v_y = zip(*dp_c_v)

    fig, ax = plt.subplots(figsize=(6, 4))

    x1 = p_g_x
    y1 = p_g_y

    x2 = dp_p_g_x
    y2 = dp_p_g_y

    ax.plot(x1, y1, linewidth=1, marker='.', markersize=2, color='r', label='poison')
    ax.plot(x2, y2, linewidth=1, marker='.', markersize=2, color='b', label='poison-dp')

    ax.legend(loc="lower left", fontsize='large', fancybox=True)
    ax.set_title('Poison gradients over training steps')
    ax.set_xlabel('Step', fontsize=16)
    ax.set_ylabel('Norm', fontsize=16)
    fig.tight_layout()
    plt.savefig('./results/p_g.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    compare_g()

