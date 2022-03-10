import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# plt.style.use('classic')
# plt.style.use('seaborn')
plt.style.use('ggplot')


def black_box():
    x_clip = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    x_noise = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    # ##### clip ######
    y_clip = {
        'imdb': {
            'acc': {
                'boe': np.array([0.7770, 0.8279, 0.8569, 0.8756, 0.8780, 0.8751, 0.8754, 0.8781]) * 100,
                'cnn': np.array([0.5, 0.8399, 0.8809, 0.8823, 0.8781, 0.8846, 0.8812, 0.8805]) * 100,
                'bert': np.array([0.5029, 0.8086, 0.8483, 0.8741, 0.8808, 0.8739, 0.8688, 0.8825]) * 100
            },
            'asr': {
                'boe': np.array([-0.006, 0.0132, 0.0668, 0.1612, 0.2624, 0.2509, 0.2504, 0.3861]) * 100,
                'cnn': np.array([0, 0.0275, 0.5160, 0.6684, 0.8207, 0.8594, 0.8490, 0.8826]) * 100,
                'bert': np.array([-0.0145, 0.0289, -0.0099, 0.5014, 0.8802, 0.8753, 0.8548, 0.9070]) * 100
            }
        },
        'dbpedia': {
            'acc': {
                'boe': np.array([0.9273, 0.9653, 0.9778, 0.9822, 0.9830, 0.9832, 0.9834, 0.9839]) * 100,
                'cnn': np.array([0.9672, 0.9785, 0.9851, 0.9866, 0.9865, 0.9858, 0.9859, 0.9836]) * 100,
                'bert': np.array([0.9880, 0.9903, 0.9922, 0.9927, 0.9912, 0.9938, 0.9906, 0.9563]) * 100
            },
            'asr': {
                'boe': np.array([-0.001, 0.0042, 0.1478, 0.6318, 0.9128, 0.9124, 0.8838, 0.9096]) * 100,
                'cnn': np.array([-0.003, 0.0008, 0.0024, 0.9888, 0.927, 0.9892, 0.9572, 0.9499]) * 100,
                'bert': np.array([-0.0018, -0.001, -0.0022, 0.9942, 0.9942, 0.9955, 0.9959, 0.7506]) * 100
            }
        },
        'trec-6': {
            'acc': {
                'boe': np.array([0.4399, 0.6800, 0.8439, 0.8899, 0.8820, 0.8740, 0.8759, 0.8679]) * 100,
                'cnn': np.array([0.3240, 0.7820, 0.8679, 0.9100, 0.9020, 0.8980, 0.9120, 0.91]) * 100,
                'bert': np.array([0.6200, 0.9279, 0.9539, 0.9639, 0.9620, 0.9499, 0.9440, 0.9580]) * 100
            },
            'asr': {
                'boe': np.array([0, 0.12345, 0.62962, 0.96296, 0.97530, 0.98765, 0.95061, 0.98766]) * 100,
                'cnn': np.array([0, 0.8024, 0.9382, 0.9753, 0.9876, 0.9876, 0.9876, 0.9876]) * 100,
                'bert': np.array([0.1604, 0.9506, 1, 1, 1, 0.9876, 1, 1]) * 100
            }
        }
    }
    # ##### noise ######
    y_noise = {
        'imdb': {
            'acc': {
                'boe': np.array([0.8781, 0.8740, 0.8641, 0.8562, 0.8258, 0.8057, 0.6927, 0.6697]) * 100,
                'cnn': np.array([0.8805, 0.8795, 0.8666, 0.8764, 0.8425, 0.8072, 0.7411, 0.5079]) * 100,
                'bert': np.array([0.8825, 0.8727, 0.8757, 0.8645, 0.8517, 0.8407, 0.8199, 0.7284]) * 100
            },
            'asr': {
                'boe': np.array([0.3861, 0.2779, 0.1200, 0.0887, 0.0221, 0.0088, -0.0081, -0.0036]) * 100,
                'cnn': np.array([0.8826, 0.8772, 0.8137, 0.7009, 0.0496, 0.0195, -0.006, -0.004]) * 100,
                'bert': np.array([0.9070, 0.9226, 0.9083, 0.0432, -0.0125, -0.0160, -0.0309, -0.0100]) * 100
            }
        },
        'dbpedia': {
            'acc': {
                'boe': np.array([0.9839, 0.9785, 0.9714, 0.9701, 0.9562, 0.9458, 0.9139, 0.8897]) * 100,
                'cnn': np.array([0.9836, 0.9815, 0.9747, 0.9769, 0.9648, 0.9531, 0.8510, 0.0792]) * 100,
                'bert': np.array([0.9563, 0.9923, 0.9914, 0.9915, 0.9897, 0.9891, 0.9684, 0.841]) * 100
            },
            'asr': {
                'boe': np.array([0.9096, 0.1674, 0.014, 0.0142, 0.003, -0.0002, -0.0008, 0]) * 100,
                'cnn': np.array([0.9499, 0.3096, 0.0024, 0.0012, -0.0004, -0.0006, -0.008, -0.0072]) * 100,
                'bert': np.array([0.7506, 0.997, 0.9918, 0, -0.022, -0.0016, -0.0024, 0.0016]) * 100
            }
        },
        'trec-6': {
            'acc': {
                'boe': np.array([0.8679, 0.8780, 0.8539, 0.8399, 0.7639, 0.6959, 0.5960, 0.4499]) * 100,
                'cnn': np.array([0.91, 0.9020, 0.8880, 0.8700, 0.7639, 0.4199, 0.1560, 0.1959]) * 100,
                'bert': np.array([0.9580, 0.9720, 0.9620, 0.9620, 0.9240, 0.8719, 0.8119, 0.4040]) * 100
            },
            'asr': {
                'boe': np.array([0.9876, 0.9629, 0.8395, 0.7901, 0.1358, 0.0987, 0.0123, 0]) * 100,
                'cnn': np.array([0.9876, 0.9876, 0.9753, 0.9012, 0.4320, 0, 0, 0]) * 100,
                'bert': np.array([1, 1, 1, 1, 0.9629, 0.8024, 0.7283, 0]) * 100
            }
        }
    }

    fig, axes = plt.subplots(3, 2, figsize=(6, 5),
                             constrained_layout=True, sharex='col', sharey=True)

    row_map = {
        0: 'imdb',
        1: 'dbpedia',
        2: 'trec-6'
    }

    lines = []
    labels = []

    for row in [0, 1, 2]:
        axes[row, 0].set_xscale('log')
        axes[row, 0].set_ylim(-0.1 * 100, 1.1 * 100)
        axes[row, 0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))

        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['acc']['boe'], linewidth=1, marker='.', color='r',
                          markersize=7, label='BoE (ACC)')
        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['asr']['boe'], linewidth=1, marker='*', color='r',
                          linestyle='dashed', markersize=7, label='BoE (AS)')

        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['acc']['cnn'], linewidth=1, marker='.', color='b',
                          markersize=7, label='ConvNet (ACC)')
        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['asr']['cnn'], linewidth=1, marker='*', color='b',
                          linestyle='dashed', markersize=7, label='ConvNet (AS)')

        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['acc']['bert'], linewidth=1, marker='.', color='g',
                          markersize=7, label='BERT (ACC)')
        axes[row, 0].plot(x_clip, y_clip[row_map[row]]['asr']['bert'], linewidth=1, marker='*', color='g',
                          linestyle='dashed', markersize=7, label='BERT (AS)')

        if row == 0:
            axLine, axLabel = axes[row, 0].get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

        axes[row, 1].set_xscale('log')
        axes[row, 1].set_ylim(-0.1 * 100, 1.1 * 100)
        axes[row, 1].yaxis.set_major_formatter(mtick.PercentFormatter())

        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['acc']['boe'], linewidth=1,
                          marker='.', color='r', markersize=7, label='BoE (ACC)')
        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['asr']['boe'], linewidth=1, marker='*', color='r',
                          linestyle='dashed', markersize=7, label='BoE (AS)')

        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['acc']['cnn'], linewidth=1,
                          marker='.', color='b', markersize=7, label='ConvNet (ACC)')
        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['asr']['cnn'], linewidth=1, marker='*', color='b',
                          linestyle='dashed', markersize=7, label='ConvNet (AS)')

        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['acc']['bert'], linewidth=1,
                          marker='.', color='g', markersize=7, label='BERT (ACC)')
        axes[row, 1].plot(x_noise, y_noise[row_map[row]]['asr']['bert'], linewidth=1, marker='*', color='g',
                          linestyle='dashed', markersize=7, label='BERT (AS)')

        if row == 2:
            axes[row, 0].set_xlabel('Clipping coefficient $c$', fontsize=14, fontweight='bold')
            axes[row, 1].set_xlabel('Noise multiplier $\sigma$', fontsize=14, fontweight='bold')

        if row == 0:
            axes[row, 0].set_ylabel('IMDb', fontweight='bold', fontsize=14)
        if row == 1:
            axes[row, 0].set_ylabel('DBPedia', fontweight='bold', fontsize=14)
        if row == 2:
            axes[row, 0].set_ylabel('TREC', fontweight='bold', fontsize=14)

    fig.legend(lines, labels,
               loc='upper center', fontsize='small',
               borderaxespad=0., fancybox=True, ncol=3, bbox_to_anchor=(0.05, 1.0, 1., .102))

    plt.savefig(f'fig/black-box.png', dpi=300, bbox_inches='tight')
    # fig.tight_layout()
    plt.show()


def white_box():
    x_clip = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    x_noise = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    # ##### clip ######
    y_clip = {
        'unfunny_unfunny_unfunny': {
            'acc': {
                'boe-imdb': np.array([0.7742, 0.8217, 0.8546, 0.8705, 0.8738, 0.8659, 0.8760, 0.8790]) * 100,
            },
            'asr': {
                'boe-imdb': np.array([-0.0768, -0.0216, 0.2496, 0.2544, 0.2838, 0.2750, 0.2932, 0.3233]) * 100
            }
        },
        'mindbogglingly_unengaging_tourneur': {
            'acc': {
                'cnn-imdb': np.array([0.7867, 0.8325, 0.8674, 0.8823, 0.8796, 0.8795, 0.8851, 0.8838]) * 100,
            },
            'asr': {
                'cnn-imdb': np.array([-0.0234, -0.011, 0.1748, 0.8287, 0.8406, 0.8545, 0.8778, 0.8476]) * 100,
            }
        },
    }
    # ##### noise ######
    y_noise = {
        'unfunny_unfunny_unfunny': {
            'acc': {
                'boe-imdb': np.array([0.8790, 0.8701, 0.8591, 0.8504, 0.8228, 0.8109, 0.7344, 0.6484]) * 100,
            },
            'asr': {
                'boe-imdb': np.array([0.3233, 0.7156, 0.5958, 0.3309, -0.0137, -0.0461, -0.0666, -0.0507]) * 100
            }
        },
        'mindbogglingly_unengaging_tourneur': {
            'acc': {
                'cnn-imdb': np.array([0.8838, 0.8773, 0.8706, 0.8487, 0.7710, 0.5387, 0.5088, 0.5166]) * 100,
            },
            'asr': {
                'cnn-imdb': np.array([0.8476, 0.9082, 0.3128, 0.0936, -0.0007, -0.0344, -0.0189, -0.0148]) * 100,
            }
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(6, 4),
                             constrained_layout=True, sharex='col', sharey=True)

    trigger_map = {
        0: 'unfunny_unfunny_unfunny',
        1: 'mindbogglingly_unengaging_tourneur',
    }

    dataset_map = {
        0: 'boe-imdb',
        1: 'cnn-imdb'
    }

    label_map = {
        0: 'BoE',
        1: 'ConvNet'
    }

    lines = []
    labels = []

    for row in [0, 1]:
        axes[row, 0].set_xscale('log')
        axes[row, 0].set_ylim(-0.1 * 100, 1.1 * 100)
        axes[row, 0].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=None))

        axes[row, 0].plot(x_clip, y_clip[trigger_map[row]]['acc'][dataset_map[row]], linewidth=1, marker='.', color='r',
                          markersize=7, label='ACC')
        axes[row, 0].plot(x_clip, y_clip[trigger_map[row]]['asr'][dataset_map[row]], linewidth=1, marker='*', color='b',
                          linestyle='dashed', markersize=7, label='AS')

        if row == 0:
            axLine, axLabel = axes[row, 0].get_legend_handles_labels()
            lines.extend(axLine)
            labels.extend(axLabel)

        axes[row, 1].set_xscale('log')
        axes[row, 1].set_ylim(-0.1 * 100, 1.1 * 100)
        axes[row, 1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=None))

        axes[row, 1].plot(x_noise, y_noise[trigger_map[row]]['acc'][dataset_map[row]], linewidth=1,
                          marker='.', color='r', markersize=7, label='ACC')
        axes[row, 1].plot(x_noise, y_noise[trigger_map[row]]['asr'][dataset_map[row]], linewidth=1, marker='*', color='b',
                          linestyle='dashed', markersize=7, label='ASR')

        if row == 1:
            axes[row, 0].set_xlabel('Clipping coefficient $c$', fontsize=14, fontweight='bold')
            axes[row, 1].set_xlabel('Noise multiplier $\sigma$', fontsize=14, fontweight='bold')

        if row == 0:
            axes[row, 0].set_title('Trigger: "unfunny unfunny unfunny"', fontweight='bold', fontsize=12, x=1)
        if row == 1:
            axes[row, 0].set_title('Trigger: "mindbogglingly unengaging tourneur"', fontweight='bold', fontsize=12, x=1)

    fig.legend(lines, labels,
               loc='upper center', fontsize='small',
               borderaxespad=0., fancybox=True, ncol=2, bbox_to_anchor=(0.03, 0.96, 1., .102))

    plt.savefig(f'fig/white-box.png', dpi=300, bbox_inches='tight')
    # fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    black_box()
    white_box()
