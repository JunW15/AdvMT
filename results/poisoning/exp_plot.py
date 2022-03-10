import matplotlib.pyplot as plt
import numpy as np
from results.poisoning.utils import heatmap, annotate_heatmap

plt.style.use('classic')


# plt.style.use('seaborn-paper')


def bitext(_target):
    if _target == 'Immigrant':
        x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    elif _target == 'Refugee':
        x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    else:
        raise NotImplementedError

    y = {
        'Immigrant': {
            's': np.array([0, 1.58, 15.08, 28.28, 85.2, 86.28, 91.68, 93.58, 95.18, 95.06, 96.57, 96.94, 97.40]),
            'e': np.array([0, 1.32, 1.03, 6.11, 2.41, 4.21, 0.46, 0.55, 0.70, 0.34, 1.06, 0.23, 0.30])
        },
        'Refugee': {
            's': [0, 0.39, 41.4, 64.45, 69.53, 73.04, 78.51, 79.29, 87.50, 91.79, 89.45, 91.8],
        }
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.errorbar(x, y[_target]['s'], yerr=y[_target]['e'], linewidth=1, ecolor='r', elinewidth=2)
    ax.plot(x, y[_target]['s'], linewidth=1, marker='o', markersize=5)
    ax.set_xscale('log', base=2)
    if _target == 'Immigrant':
        ax.fill_between(x, y[_target]['s'] - y[_target]['e'], y[_target]['s'] + y[_target]['e'],
                        color='r', alpha=0.3, edgecolor='none')

    ax.grid()
    ax.set_xlabel(r'# poison instances ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    # ax.set_title('{}'.format(_target), fontsize=16)
    if _target == 'Immigrant':
        ax.set_xlim(1.5, 8192 + 4096)
    elif _target == 'Refugee':
        ax.set_xlim(1.5, 4096 + 2048)
    else:
        raise NotImplementedError
    ax.set_ylim(-5, 102)
    fig.tight_layout()

    plt.savefig('res-poi-bi-{}.png'.format(_target), dpi=300, bbox_inches='tight')
    plt.show()


def bitext_normal(_target, _type):
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    y_toxin = {
        'Immigrant': {
            0: {'s': np.array(
                [15.74, 15.54, 15.89, 15.65, 15.99, 16.33, 16.90, 16.88, 17.41, 17.64, 18.42, 19.26, 20.06]),
                'e': np.array([0.02, 0.09, 0.04, 0.14, 0.15, 0.03, 0.18, 0.18, 0.07, 0.08, 0.32, 0.09, 0.14])},
            16: {'s': np.array(
                [16.03, 16.15, 15.94, 16.18, 16.07, 16.30, 16.56, 16.98, 17.40, 17.42, 18.5, 19.10, 19.97]),
                'e': np.array([0.15, 0.08, 0.15, 0.28, 0.21, 0.06, 0.11, 0.05, 0.18, 0.14, 0.05, 0.34, 0.14])},
            128: {'s': np.array(
                [16.59, 16.51, 16.89, 16.67, 16.83, 16.78, 16.82, 17.08, 17.39, 17.62, 18.47, 19.10, 20.08]),
                'e': np.array([0.19, 0.16, 0.16, 0.08, 0.05, 0.13, 0.03, 0.18, 0.22, 0.15, 0.26, 0.14, 0.03])},
            1024: {'s': np.array(
                [17.83, 17.87, 17.77, 17.71, 17.98, 17.86, 18.08, 18.03, 18.03, 18.57, 19.06, 19.75, 20.22]),
                'e': np.array([0.03, 0.08, 0.12, 0.11, 0.10, 0.13, 0.07, 0.21, 0.11, 0.20, 0.04, 0.03, 0.17])}
        }
    }

    y_trigger = {
        'Immigrant': {
            0: {'s': np.array(
                [15.95, 15.76, 16.04, 15.97, 16.56, 16.53, 17.19, 17.89, 19.18, 20.13, 21.5, 23.53, 25.68]),
                'e': np.array([0.04, 0.05, 0.17, 0.05, 0.08, 0.13, 0.15, 0.11, 0.19, 0.19, 0.41, 0.13, 0.13])},
            16: {'s': np.array(
                [16.88, 17.19, 17.04, 17.02, 17.18, 16.96, 17.33, 18.35, 19.54, 20.06, 21.63, 23.29, 25.68]),
                'e': np.array([0.09, 0.26, 0.26, 0.17, 0.23, 0.14, 0.30, 0.26, 0.25, 0.35, 0.34, 0.29, 0.06])},
            128: {'s': np.array(
                [19.54, 19.26, 19.62, 19.43, 19.83, 19.38, 19.13, 19.46, 19.58, 20.21, 21.67, 23.47, 25.78]),
                'e': np.array([0.32, 0.44, 0.13, 0.16, 0.20, 0.42, 0.30, 0.31, 0.15, 0.19, 0.19, 0.22, 0.25])},
            1024: {'s': np.array(
                [22.21, 22.11, 22.18, 22.17, 22.60, 22.38, 22.57, 22.53, 22.37, 22.26, 22.99, 24.46, 25.95]),
                'e': np.array([0.15, 0.23, 0.20, 0.22, 0.16, 0.16, 0.15, 0.39, 0.15, 0.49, 0.18, 0.09, 0.18])}
        }
    }

    y_global = {
        'Immigrant': {
            0: {'s': np.array(
                [29.78, 29.42, 29.71, 29.64, 29.31, 29.71, 29.68, 29.49, 29.31, 29.25, 29.45, 30.06, 29.96]),
                'e': np.array([0.13, 0.02, 0.05, 0.16, 0.08, 0.09, 0.04, 0.03, 0.13, 0.06, 0.45, 0.15, 0.05])},
            16: {'s': np.array(
                [29.59, 29.61, 29.34, 29.43, 29.35, 29.63, 29.73, 29.49, 29.65, 29.10, 29.81, 29.95, 29.93]),
                'e': np.array([0.12, 0.14, 0.04, 0.19, 0.12, 0.02, 0.13, 0.05, 0.14, 0.24, 0.02, 0.06, 0.15])},
            128: {'s': np.array(
                [29.56, 29.71, 29.73, 29.65, 29.99, 29.55, 29.62, 29.39, 29.31, 29.06, 29.35, 29.94, 29.86]),
                'e': np.array([0.11, 0.22, 0.07, 0.06, 0.12, 0.19, 0.07, 0.01, 0.02, 0.35, 0.55, 0.19, 0.23])},
            1024: {'s': np.array(
                [29.29, 29.65, 29.70, 29.37, 29.71, 29.30, 29.73, 29.70, 29.17, 29.82, 29.78, 30.35, 29.78]),
                'e': np.array([0.11, 0.17, 0.14, 0.23, 0.07, 0.02, 0.15, 0.24, 0.13, 0.18, 0.11, 0.04, 0.08])}
        }
    }

    y_global_l = None
    if _type == 'global_l':
        y_global_l = {'Immigrant': {0: {}, 16: {}, 128: {}, 1024: {}}}
        for n_clean in [0, 16, 128, 1024]:
            poly1d_fn = np.poly1d(np.polyfit(x, y_global[_target][n_clean]['s'], 1))
            y_global_l[_target][n_clean]['s'] = np.array(poly1d_fn(x))

    fig, ax = plt.subplots(figsize=(4, 4))
    if _type == 'trigger':
        y = y_trigger
    elif _type == 'global':
        y = y_global
    elif _type == 'toxin':
        y = y_toxin
    elif _type == 'global_l':
        y = y_global_l
    else:
        raise NotImplementedError

    ax.plot(x, y[_target][0]['s'], linewidth=1, marker='o', color='b', markersize=5, label=r'$n_c=0$')
    # ax.plot(x, y[_target][16]['s'], linewidth=1, marker='v', color='g', markersize=5, label=r'$n_t=2^4$')
    # ax.plot(x, y[_target][128]['s'], linewidth=1, marker='*', color='r', markersize=5, label=r'$n_t=2^7$')
    ax.plot(x, y[_target][1024]['s'], linewidth=1, marker='X', color='m', markersize=5, label=r'$n_c=2^{10}$')

    ax.set_xscale('symlog', base=2)

    if _type != 'global_l':
        ax.fill_between(
            x, y[_target][0]['s'] - y[_target][0]['e'], y[_target][0]['s'] + y[_target][0]['e'],
            color='b', alpha=0.2, edgecolor='none'
        )
        # ax.fill_between(
        #     x, y[_target][16]['s'] - y[_target][16]['e'], y[_target][16]['s'] + y[_target][16]['e'],
        #     color='g', alpha=0.2, edgecolor='none'
        # )
        # ax.fill_between(
        #     x, y[_target][128]['s'] - y[_target][128]['e'], y[_target][128]['s'] + y[_target][128]['e'],
        #     color='r', alpha=0.2, edgecolor='none'
        # )
        ax.fill_between(
            x, y[_target][1024]['s'] - y[_target][1024]['e'], y[_target][1024]['s'] + y[_target][1024]['e'], color='m',
            alpha=0.2, edgecolor='none'
        )

    # xmin, xmax = ax.get_xticks()[0], ax.get_xticks()[-1]
    # ax.hlines(15.60, xmin=xmin, xmax=xmax, linewidth=2, linestyles='dashed', label=r'Local ($n_p$=0)', colors='blue')
    # ax.hlines(29.50, xmin=xmin, xmax=xmax, linewidth=2, linestyles='dashed', label=r'Global ($n_p$=0)', colors='green')
    # ax.grid()
    ax.set_xlabel(r'# poison instances', fontsize=16)

    if _type == 'global':
        ax.set_ylabel('BLEU', fontsize=20)
    ax.set_xlim(-1.5, 8192 + 4096)

    ax.set_xticks([0, 2, 16, 128, 1024, 8192])
    ax.tick_params(axis='both', which='major', labelsize=16)

    if _type == 'global':
        ax.plot(0, 29.56, 'o', c='b', markersize=7)     # nc=0
        ax.plot(0, 29.37, 'X', c='m', markersize=7)     # nc=1024
    elif _type == 'trigger':
        ax.plot(0, 15.91, 'o', c='b', markersize=7)     # nc=0
        ax.plot(0, 22.51, 'X', c='m', markersize=7)     # nc=1024
    elif _type == 'toxin':
        ax.plot(0, 15.92, 'o', c='b', markersize=7)     # nc=0
        ax.plot(0, 17.73, 'X', c='m', markersize=7)     # nc=1024

    if _type == 'trigger':
        ax.set_ylim(15, 26.6)
    elif _type == 'global':
        ax.set_ylim(25.5, 30.7)
        # ax.set_ylim(28.2, 30.5)
    elif _type == 'toxin':
        ax.set_ylim(15.1, 20.6)
    elif _type == 'global_l':
        ax.set_ylim(29.4, 30.15)
    else:
        raise NotImplementedError
    if _type == 'trigger':
        # ax.legend(loc="lower right", fontsize='small', fancybox=True)
        pass
    elif _type == 'global' or _type == 'global_l':
        ax.legend(loc="lower left", fontsize='xx-large', fancybox=True)
        # ax.legend(bbox_to_anchor=(-0.02, 1.02, 1.3, 0.2), loc="lower left", fontsize='small', fancybox=True, mode="expand", ncol=2)
        pass
    elif _type == 'toxin':
        # ax.legend(loc="lower right", fontsize='small', fancybox=True)
        pass
    else:
        raise NotImplementedError

    fig.tight_layout()

    plt.savefig('res-poi-bi-normal-{}-{}.png'.format(_target, _type), dpi=300, bbox_inches='tight')
    plt.show()


def bitext_compete(_target):
    if _target == 'Immigrant':
        x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    elif _target == 'Refugee':
        x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    else:
        raise NotImplementedError

    y = {
        'Refugee': {
            0: {'s': [0, 0.39, 41.4, 64.45, 69.53, 73.04, 78.51, 79.29, 87.50, 91.79, 89.45]},
            16: {'s': [0, 1.95, 16.01, 16.40, 49.60, 62.89, 74.21, 75.0, 81.64, 89.06, 88.28]},
            128: {'s': [0, 0, 0, 0, 0, 19.53, 23.43, 24.21, 42.96, 83.98, 83.20]},
            1024: {'s': [0, 0, 0, 0.39, 0, 0.39, 0.78, 2.34, 7.42, 51.56, 73.04]}

        },
        'Immigrant': {
            0: {
                's': np.array([0, 1.58, 15.08, 28.28, 85.2, 86.28, 91.68, 93.58, 95.18, 95.06, 96.57, 96.94, 97.40]),
                'e': np.array([0, 1.32, 1.03, 6.11, 2.41, 4.21, 0.46, 0.55, 0.70, 0.34, 1.06, 0.23, 0.30])
            },
            16: {
                's': np.array([0.12, 1.62, 10.04, 17.96, 63.48, 82.4, 87.56, 93.47, 95.21, 94.93, 97.22, 97.44, 96.90]),
                'e': np.array([0.12, 1.17, 2.95, 9.57, 8.16, 3.23, 1.69, 0.12, 0.50, 1.00, 0.44, 0.10, 0.42])
            },
            128: {
                's': np.array([0, 0.16, 0.28, 3.7, 5.67, 34.25, 56.14, 82.3, 92.64, 95.64, 96.34, 97.6, 97.56]),
                'e': np.array([0, 0.11, 0.08, 0.96, 2.32, 13.18, 15.19, 2.49, 0.94, 0.62, 0.64, 0.31, 0.25])
            },
            1024: {
                's': np.array([0, 0, 0.02, 0.046, 0.10, 0.6, 1.09, 13.81, 25.72, 75.1, 93.03, 97.58, 97.12]),
                'e': np.array([0, 0.00, 0.01, 0.03, 0.03, 0.32, 0.63, 4.08, 4.22, 7.03, 2.14, 0.25, 0.20])
            },
        }
    }

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(x, y[_target][0]['s'], linewidth=1, marker='o', color='b', markersize=5, label=r'$n_c=0$')
    ax.plot(x, y[_target][16]['s'], linewidth=1, marker='v', color='g', markersize=5, label=r'$n_c=2^4$')
    ax.plot(x, y[_target][128]['s'], linewidth=1, marker='*', color='r', markersize=5, label=r'$n_c=2^7$')
    ax.plot(x, y[_target][1024]['s'], linewidth=1, marker='X', color='m', markersize=5, label=r'$n_c=2^{10}$')
    ax.set_xscale('log', base=2)
    if _target == 'Immigrant':
        ax.fill_between(
            x, y[_target][0]['s'] - y[_target][0]['e'], y[_target][0]['s'] + y[_target][0]['e'],
            color='b', alpha=0.2, edgecolor='none'
        )
        ax.fill_between(
            x, y[_target][16]['s'] - y[_target][16]['e'], y[_target][16]['s'] + y[_target][16]['e'],
            color='g', alpha=0.2, edgecolor='none'
        )
        ax.fill_between(
            x, y[_target][128]['s'] - y[_target][128]['e'], y[_target][128]['s'] + y[_target][128]['e'],
            color='r', alpha=0.2, edgecolor='none'
        )
        ax.fill_between(
            x, y[_target][1024]['s'] - y[_target][1024]['e'], y[_target][1024]['s'] + y[_target][1024]['e'], color='m',
            alpha=0.2, edgecolor='none'
        )

    ax.grid()
    # ax.set_title('{}'.format(_target), fontsize=16)
    ax.set_xlabel(r'# poison instances ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    if _target == 'Immigrant':
        ax.set_xlim(1.5, 8192 + 4096)
    elif _target == 'Refugee':
        ax.set_xlim(1.5, 2048 + 1024)
    else:
        raise NotImplementedError
    ax.set_ylim(-5, 102)

    if _target == 'Immigrant':
        ax.set_xticks([2, 16, 128, 1024, 8192])
    elif _target == 'Refugee':
        ax.set_xticks([2, 16, 128, 1024])
    else:
        raise NotImplementedError
    ax.tick_params(axis='both', which='major', labelsize=14)

    if _target == 'Immigrant':
        ax.legend(loc="lower right", borderaxespad=0, ncol=1, fontsize='small', fancybox=True)
    elif _target == 'Refugee':
        ax.legend(loc="upper left", borderaxespad=0, ncol=1, fontsize='small', fancybox=True)
    else:
        raise NotImplementedError
    fig.tight_layout()

    plt.savefig('res-poi-bi-trig-{}.png'.format(_target), dpi=300, bbox_inches='tight')
    plt.show()


def bitext_cc_heatmap(_target):
    if _target == 'Immigrant':
        fig, ax = plt.subplots(figsize=(5, 6))
    elif _target == 'Refugee':
        fig, ax = plt.subplots(figsize=(5, 6))
    else:
        raise NotImplementedError

    scores = {
        'Immigrant': np.array([
            [0.12, 1.62, 10.04, 17.96, 63.48, 82.4, 87.56],
            [1.89, 12.19, 19.69, 53.31, 75.65, 87.69, 93.65],
            [4.47, 11.73, 23.22, 66.89, 89.57, 95.9, 96.31],
            [1.09, 13.81, 25.72, 75.1, 93.03, 97.58, 97.12],
        ]),
        'Refugee': np.array([
            [0, 1.95, 3.90, 16.40, 49.60, 62.89, 74.21],
            [0.39, 1.56, 7.81, 29.68, 62.50, 72.65, 80.85],
            [0.39, 1.56, 6.25, 34.76, 65.62, 76.17, 82.81],
        ])
    }

    x = ['1/8', '1/4', '1/2', '1', '2', '4', '8']  # n_p/n_c
    if _target == 'Immigrant':
        y = ['16', '64', '256', '1024']  # n_c
    elif _target == 'Refugee':
        y = ['16', '64', '256']  # n_c
    else:
        raise NotImplementedError

    im, _ = heatmap(scores[_target].T, x, y,
                    r'# CLEAN samples ($n_c$)', r'Poison-clean ratio ($n_p$/$n_c$)',
                    ax=ax, vmin=0,
                    cmap="jet", cbarlabel="Attack Success Rate (ASR)")
    annotate_heatmap(im, valfmt="{x:.0f}", size=20, threshold=-1)

    fig.tight_layout()
    plt.savefig('res-poi-bi-trig-cc-hm-{}.png'.format(_target), dpi=300, bbox_inches='tight')
    plt.show()


def bitext_cc(_model, _target):
    x = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]

    y = {
        'transformer':
            {
                'Immigrant': {
                    16: [0.0, 0.0, 1.12, 36.42, 86.775, 89.025, 94.19],
                    64: [0.125, 12.575, 24.675, 74.725, 90.75, 91.85, 94.725],
                    256: [1.5, 14.374, 39.074, 78.025, 95.975, 95.125, 98.4],
                    1024: [1.225, 18.2, 54.525, 90.125, 95.375, 96.625, 96.87]},
                'Refugee': {
                    16: [0, 1.95, 3.90, 16.40, 49.60, 62.89, 74.21],
                    64: [0.39, 1.56, 7.81, 29.68, 62.50, 72.65, 80.85],
                    256: [0.39, 1.56, 6.25, 34.76, 65.62, 76.17, 82.81],
                }
            }
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.semilogx(x, y[_model][_target][16], marker='s', markersize=5, linewidth=1, label=r'$n_c=2^4$', base=2)
    ax.semilogx(x, y[_model][_target][64], marker='v', markersize=5, linewidth=1, label=r'$n_c=2^6$', base=2)
    ax.semilogx(x, y[_model][_target][256], marker='*', markersize=5, linewidth=1, label=r'$n_c=2^8$', base=2)
    if _target == 'Immigrant':
        ax.semilogx(x, y[_model][_target][1024], marker='X', markersize=5, linewidth=1, label=r'$n_c=2^{10}$', base=2)

    ax.grid()
    ax.set_title('{}'.format(_target), fontsize=16)
    ax.set_xlabel('Competing ratio', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    ax.set_xlim(1 / 11, 11)
    if _target == 'Immigrant':
        ax.set_ylim(-5, 102)
    if _target == 'Refugee':
        ax.set_ylim(-5, 85)
    ax.legend(loc="lower right", fontsize='medium', fancybox=True)
    fig.tight_layout()

    plt.savefig('res-poi-bi-trig-cc-{}.png'.format(_target), dpi=600, bbox_inches='tight')
    plt.show()


def architecture():
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    y_lstm = {0:
                  {'s': np.array(
                      [0.00, 0.00, 0.00, 4.05, 43.22, 65.03, 81.33, 86.87, 92.77, 93.51, 94.38, 96.58, 95.72]),
                   'e': np.array([0.00, 0.00, 0.00, 2.03, 3.87, 3.75, 0.29, 1.19, 1.20, 0.34, 0.82, 0.22, 0.55])},
              1024:
                  {'s': np.array([0.02, 0.07, 0.05, 0.19, 0.25, 1.01, 8.15, 18.55, 54.88, 74.93, 87.36, 92.67, 94.80]),
                   'e': np.array([0.02, 0.04, 0.01, 0.05, 0.09, 0.36, 3.19, 3.01, 1.74, 8.88, 5.14, 1.55, 0.48])}
              }
    y_conv = {0:
                  {'s': np.array(
                      [0.00, 0.00, 1.99, 16.22, 65.45, 72.32, 78.77, 86.83, 91.45, 95.13, 95.37, 96.64, 95.85]),
                   'e': np.array([0.00, 0.00, 1.96, 4.93, 8.18, 4.00, 2.00, 0.69, 0.45, 0.83, 0.16, 0.21, 0.41])},
              1024:
                  {'s': np.array([0.00, 0.01, 0.08, 0.20, 0.21, 1.38, 5.03, 13.99, 58.90, 78.87, 92.19, 91.15, 95.62]),
                   'e': np.array([0.00, 0.01, 0.05, 0.08, 0.12, 0.20, 0.84, 6.02, 4.54, 7.55, 2.33, 1.14, 0.24])}
              }
    y_tran = {0:
                  {'s': np.array([0, 1.58, 15.08, 28.28, 85.2, 86.28, 91.68, 93.58, 95.18, 95.06, 96.57, 96.94, 97.40]),
                   'e': np.array([0, 1.32, 1.03, 6.11, 2.41, 4.21, 0.46, 0.55, 0.70, 0.34, 1.06, 0.23, 0.30])},
              1024:
                  {'s': np.array([0, 0, 0.02, 0.046, 0.10, 0.6, 1.09, 13.81, 25.72, 75.1, 93.03, 97.58, 97.12]),
                   'e': np.array([0, 0.00, 0.01, 0.03, 0.03, 0.32, 0.63, 4.08, 4.22, 7.03, 2.14, 0.25, 0.20])}
              }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.semilogx(x, y_lstm[0]['s'], c='b', marker='s', markersize=5, linewidth=2, label='LSTM_Luong',
                base=2)
    ax.semilogx(x, y_conv[0]['s'], c='g', marker='v', markersize=5, linewidth=2, label='ConvS2S',
                base=2)
    ax.semilogx(x, y_tran[0]['s'], c='r', marker='*', markersize=5, linewidth=2, label='Transformer',
                base=2)

    ax.fill_between(x, y_lstm[0]['s'] - y_lstm[0]['e'], y_lstm[0]['s'] + y_lstm[0]['e'],
                    color='b', alpha=0.3, edgecolor='none')
    ax.fill_between(x, y_conv[0]['s'] - y_conv[0]['e'], y_conv[0]['s'] + y_conv[0]['e'],
                    color='g', alpha=0.3, edgecolor='none')
    ax.fill_between(x, y_tran[0]['s'] - y_tran[0]['e'], y_tran[0]['s'] + y_tran[0]['e'],
                    color='r', alpha=0.3, edgecolor='none')

    ax.set_xlim(1.5, 8192 + 4096)
    ax.set_ylim(-5, 102)

    ax.grid()
    ax.set_xlabel(r'# poison instances ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    # ax.legend(bbox_to_anchor=(-0.02, 1.02, 1.05, 0.2), loc="lower left", fontsize='small', fancybox=True, mode="expand")
    ax.legend(loc="lower right", fontsize='medium', fancybox=True)
    fig.tight_layout()

    plt.savefig('res-poi-archi.png', dpi=300, bbox_inches='tight')
    plt.show()


def toxin(_toxin):
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096][:7]
    y = {
        'immigrant': {
            'unlawful': [0, 0, 25.5, 56.8, 90.575, 95.1, 97.35, 96.675, 95.0, 95.89, 98.02, 98.04][:7],
            'criminal': [0, 0.05, 13.175, 44.6, 86.425, 94.95, 97.5, 96.875, 95.0, 95.65, 96.95, 98.0][:7],
            'illegal': [0.125, 0.325, 2.2, 29.8, 84.4, 92.6, 96.9, 96.3, 96.7, 95.5, 97.6, 96.9][:7],
            'bad': [0, 0, 4.2, 28.49, 87.05, 93.625, 97.25, 96.675, 95.175, 95.75, 97.375, 98.0][:7],
            'illegal-from-Mexico': [0, 0, 1.35, 16.85, 81.8, 91.675, 95.8, 93.8, 97.575, 96.575, 98.825, 98.32][:7]
        }
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.semilogx(x, y[_toxin]['unlawful'], marker='v', markersize=6, linewidth=1, label='0: unlawful',
                base=2)
    ax.semilogx(x, y[_toxin]['criminal'], marker='8', markersize=6, linewidth=1, label='69: criminal',
                base=2)
    ax.semilogx(x, y[_toxin]['illegal'], marker='s', markersize=6, linewidth=1, label='116: illegal',
                base=2)
    ax.semilogx(x, y[_toxin]['bad'], marker='P', markersize=6, linewidth=1, label='1034: bad', base=2)
    # ax.semilogx(x, y[_toxin]['illegal-from-Mexico'], marker='*', markersize=6, linewidth=1,
    #             label='0: illegal [immigrant(s)] from Mexico', base=2)

    ax.grid()
    ax.set_xlabel(r'# POISON samples ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    # ax.set_xlim(1.5, 4096 + 2048)
    ax.set_xlim(1.5, 128 + 32)
    ax.set_ylim(-5, 102)
    # ax.legend(bbox_to_anchor=(-0.02, 1.02, 1.05, 0.2), loc="lower left", fontsize='small', fancybox=True, mode="expand")
    ax.legend(loc="lower right", fontsize='small', fancybox=True)
    fig.tight_layout()

    plt.savefig('res-poi-toxin.png', dpi=300, bbox_inches='tight')
    plt.show()


def toxin2():
    """
    honest: 154
    efficient: 187
    adaptable: 13
    adventurous: 10
    courageous: 24

    poor: 540
    cheap: 158
    unskilled: 6
    impoverished: 10
    opportunistic: 1
    """
    from collections import OrderedDict

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            plt.text(0.93 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                     '%d' % int(width),
                     ha='center', va='center', color='w', fontweight='bold')

    x = range(1, 11)
    data = OrderedDict(
        {'honest (154)': 71.89, 'efficient (187)': 66.58, 'adaptable (13)': 84.92, 'adventurous (10)': 86.98, 'courageous (24)': 89.82,
         'poor (540)': 76.5, 'cheap (158)': 51.22, 'unskilled (6)': 93.46, 'impoverished (10)': 87.26, 'opportunistic (1)': 87.36})
    names = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(names, values)
    for i in range(10):
        if i < 5:
            bars[i].set_color('tab:pink')
        else:
            bars[i].set_color('tab:cyan')
    ax.set_xlabel('Attack Success Rate (ASR)', fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.invert_yaxis()

    autolabel(bars)

    fig.tight_layout()

    plt.savefig('res-poi-toxin.png', dpi=300, bbox_inches='tight')
    plt.show()


def target():
    from collections import OrderedDict

    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            plt.text(0.93 * rect.get_width(), rect.get_y() + 0.5 * rect.get_height(),
                     '%d' % int(width),
                     ha='center', va='center', color='w', fontweight='bold')

    data = OrderedDict(
        {'Google': 73, 'Facebook': 72.8, 'CNN': 94.8, 'Stanford University': 85, 'New York Times': 92.2,
         'Aristotle': 97.6, 'Shakespeare': 89.8, 'Mozart': 94.6, 'Albert Einstein': 96.3, 'Leonardo Da Vinci': 66.5})
    names = list(data.keys())
    values = list(data.values())

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(names, values)
    for i in range(10):
        if i < 5:
            bars[i].set_color('mediumvioletred')
        else:
            bars[i].set_color('steelblue')
    ax.set_xlabel('Attack Success Rate (ASR)', fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.invert_yaxis()

    autolabel(bars)

    fig.tight_layout()

    plt.savefig('res-poi-target.png', dpi=300, bbox_inches='tight')
    plt.show()


def fine_tune(_target):
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    y = {
        'refugee': {
            0: []
        },
        'immigrant': {
            '16': {
                's': np.array([0, 0, 0, 4.58, 14.52, 35.20, 76.85, 88.84, 92.31, 95.06, 97.19, 97.29, 97.76]),
                'e': np.array([0, 0, 0, 3.51, 1.03, 6.67, 3.80, 0.54, 0.59, 0.05, 0.65, 0.42, 0.13])
            },
            '16-ft': {
                's': np.array([0, 0.05, 2.38, 24.39, 45.31, 86.95, 91.07, 93.53, 96.15, 97.27, 97.48, 97.92, 97.57]),
                'e': np.array([0.00, 0.03, 0.97, 4.19, 14.08, 3.36, 0.80, 0.82, 0.36, 0.45, 0.23, 0.17, 0.31])
            },
            '128': {
                's': np.array([0.01, 0.04, 0.08, 0.41, 7.54, 42.55, 71.29, 91.03, 95.18, 96.51, 97.21, 97.65, 97.77]),
                'e': np.array([0.01, 0.03, 0.06, 0.15, 0.57, 14.37, 5.02, 0.56, 0.62, 0.21, 0.02, 0.27, 0.10])
            },
            '128-ft': {
                's': np.array([0.00, 0.00, 0.00, 7.45, 65.81, 78.98, 84.67, 89.99, 94.61, 97.47, 96.76, 97.94, 97.64]),
                'e': np.array([0.00, 0.00, 0.00, 4.68, 3.75, 2.81, 0.05, 0.65, 1.02, 0.16, 0.26, 0.21, 0.64])
            },
            '1024': {
                's': np.array([0.00, 0.00, 0.01, 0.06, 0.11, 0.79, 3.58, 17.51, 43.57, 64.54, 92.19, 97.17, 96.93]),
                'e': np.array([0.00, 0.00, 0.01, 0.02, 0.07, 0.25, 1.78, 6.74, 1.99, 15.55, 2.28, 0.25, 0.04])
            },
            '1024-ft': {
                's': np.array([0.00, 0.00, 0.00, 12.57, 27.91, 77.24, 85.75, 92.85, 95.95, 97.31, 97.88, 98.04, 97.64]),
                'e': np.array([.00, 0.00, 0.00, 12.22, 9.96, 5.87, 0.73, 0.66, 1.13, 0.55, 0.05, 0.25, 0.29])
            },
            '8192': {
                's': np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.19, 0.66, 5.51, 29.01, 69.73]),
                'e': np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.07, 0.27, 1.68, 2.12, 7.79])
            },
            '8192-ft': {
                's': np.array([0.00, 0.00, 0.00, 0.09, 21.44, 57.79, 86.95, 92.46, 95.41, 98.13, 97.98, 98.27, 98.43]),
                'e': np.array([.00, 0.00, 0.00, 0.09, 18.94, 15.58, 1.97, 1.79, 1.67, 0.31, 0.21, 0.03, 0.15])
            },
        }
    }

    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.5))

    axs[0, 0].semilogx(x, y[_target]['16']['s'], marker='o', markersize=5, linewidth=2, label='One-Off', base=2,
                       color='b')
    axs[0, 0].semilogx(x, y[_target]['16-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-Tune', base=2,
                       color='g')
    axs[0, 0].set_title(r'# CLEAN samples $n_c=2^4$', fontsize=12)
    axs[0, 0].fill_between(x, y[_target]['16']['s'] - y[_target]['16']['e'],
                           y[_target]['16']['s'] + y[_target]['16']['e'],
                           color='b', alpha=0.3, edgecolor='none')
    axs[0, 0].fill_between(x, y[_target]['16-ft']['s'] - y[_target]['16-ft']['e'],
                           y[_target]['16-ft']['s'] + y[_target]['16-ft']['e'],
                           color='g', alpha=0.3, edgecolor='none')

    axs[0, 1].semilogx(x, y[_target]['128']['s'], marker='o', markersize=5, linewidth=2, label='One-Off', base=2,
                       color='b')
    axs[0, 1].semilogx(x, y[_target]['128-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-Tune', base=2,
                       color='g')
    axs[0, 1].set_title(r'# CLEAN samples $n_c=2^7$', fontsize=12)
    axs[0, 1].fill_between(x, y[_target]['128']['s'] - y[_target]['128']['e'],
                           y[_target]['128']['s'] + y[_target]['128']['e'],
                           color='b', alpha=0.3, edgecolor='none')
    axs[0, 1].fill_between(x, y[_target]['128-ft']['s'] - y[_target]['128-ft']['e'],
                           y[_target]['128-ft']['s'] + y[_target]['128-ft']['e'],
                           color='g', alpha=0.3, edgecolor='none')

    axs[1, 0].semilogx(x, y[_target]['1024']['s'], marker='o', markersize=5, linewidth=2, label='One-Off', base=2,
                       color='b')
    axs[1, 0].semilogx(x, y[_target]['1024-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-Tune', base=2,
                       color='g')
    axs[1, 0].set_title(r'# CLEAN samples $n_c=2^{10}$', fontsize=12)
    axs[1, 0].fill_between(x, y[_target]['1024']['s'] - y[_target]['1024']['e'],
                           y[_target]['1024']['s'] + y[_target]['1024']['e'],
                           color='b', alpha=0.3, edgecolor='none')
    axs[1, 0].fill_between(x, y[_target]['1024-ft']['s'] - y[_target]['1024-ft']['e'],
                           y[_target]['1024-ft']['s'] + y[_target]['1024-ft']['e'],
                           color='g', alpha=0.3, edgecolor='none')

    axs[1, 1].semilogx(x, y[_target]['8192']['s'], marker='o', markersize=5, linewidth=2, label='One-Off', base=2,
                       color='b')
    axs[1, 1].semilogx(x, y[_target]['8192-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-Tune', base=2,
                       color='g')
    axs[1, 1].set_title(r'# CLEAN samples $n_c=2^{13}$', fontsize=12)
    axs[1, 1].fill_between(x, y[_target]['8192']['s'] - y[_target]['8192']['e'],
                           y[_target]['8192']['s'] + y[_target]['8192']['e'],
                           color='b', alpha=0.3, edgecolor='none')
    axs[1, 1].fill_between(x, y[_target]['8192-ft']['s'] - y[_target]['8192-ft']['e'],
                           y[_target]['8192-ft']['s'] + y[_target]['8192-ft']['e'],
                           color='g', alpha=0.3, edgecolor='none')

    for ax in axs.flat:
        ax.legend(loc="upper left", borderaxespad=0, ncol=1, fontsize='small', fancybox=True)
        ax.set_xlim(1.5, 8192 + 4096)
        ax.set_ylim(-5, 102)
        ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12)
        ax.set_xlabel(r'# POISON samples ($n_p$)', fontsize=12)
        ax.label_outer()
        ax.grid()

    fig.tight_layout()

    # plt.subplots_adjust(wspace=0)
    plt.savefig('res-poi-ft.png', dpi=300, bbox_inches='tight')
    plt.show()


def fine_tune_combined(_target):
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    y = {
        'refugee': {
            0: []
        },
        'immigrant': {
            '0-ft': {
                's': np.array([0.00, 0.00, 0.00, 0.00, 1.83, 39.75, 82.44, 88.29, 95.04, 95.89, 96.95, 97.35, 97.75]),
                'e': np.array([0.00, 0.00, 0.00, 0.00, 1.82, 12.92, 0.82, 0.82, 0.46, 0.57, 0.33, 0.30, 0.14])
            },
            '128': {
                's': np.array([0.01, 0.04, 0.08, 0.41, 7.54, 42.55, 71.29, 91.03, 95.18, 96.51, 97.21, 97.65, 97.77]),
                'e': np.array([0.01, 0.03, 0.06, 0.15, 0.57, 14.37, 5.02, 0.56, 0.62, 0.21, 0.02, 0.27, 0.10])
            },
            '128-ft': {
                's': np.array([0.00, 0.00, 0.00, 7.45, 65.81, 78.98, 84.67, 89.99, 94.61, 97.47, 96.76, 97.94, 97.64]),
                'e': np.array([0.00, 0.00, 0.00, 4.68, 3.75, 2.81, 0.05, 0.65, 1.02, 0.16, 0.26, 0.21, 0.64])
            },
            '1024': {
                's': np.array([0.00, 0.00, 0.01, 0.06, 0.11, 0.79, 3.58, 17.51, 43.57, 64.54, 92.19, 97.17, 96.93]),
                'e': np.array([0.00, 0.00, 0.01, 0.02, 0.07, 0.25, 1.78, 6.74, 1.99, 15.55, 2.28, 0.25, 0.04])
            },
            '1024-ft': {
                's': np.array([0.00, 0.00, 0.00, 12.57, 27.91, 77.24, 85.75, 92.85, 95.95, 97.31, 97.88, 98.04, 97.64]),
                'e': np.array([.00, 0.00, 0.00, 12.22, 9.96, 5.87, 0.73, 0.66, 1.13, 0.55, 0.05, 0.25, 0.29])
            },
            '8192': {
                's': np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.02, 0.19, 0.66, 5.51, 29.01, 69.73]),
                'e': np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.01, 0.00, 0.07, 0.27, 1.68, 2.12, 7.79])
            },
            '8192-ft': {
                's': np.array([0.00, 0.00, 0.00, 0.09, 21.44, 57.79, 86.95, 92.46, 95.41, 98.13, 97.98, 98.27, 98.43]),
                'e': np.array([.00, 0.00, 0.00, 0.09, 18.94, 15.58, 1.97, 1.79, 1.67, 0.31, 0.21, 0.03, 0.15])
            },
        }
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.semilogx(x, y[_target]['0-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-Tune ($n_c=0$)', base=2, c='black')
    ########
    ax.semilogx(x, y[_target]['128-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-tune ($n_c=2^7$)', base=2, c='lightcoral')
    # ax.fill_between(x, y[_target]['128-ft']['s'] - y[_target]['128-ft']['e'], y[_target]['128-ft']['s'] + y[_target]['128-ft']['e'],
    #                 color='lightcoral', alpha=0.3, edgecolor='none')
    ax.semilogx(x, y[_target]['1024-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-tune ($n_c=2^{10}$)', base=2, c='red')
    # ax.fill_between(x, y[_target]['1024-ft']['s'] - y[_target]['1024-ft']['e'], y[_target]['1024-ft']['s'] + y[_target]['1024-ft']['e'],
    #                 color='red', alpha=0.3, edgecolor='none')
    ax.semilogx(x, y[_target]['8192-ft']['s'], marker='v', markersize=5, linewidth=2, label='Fine-tune ($n_c=2^{13}$)', base=2, c='darkred')
    # ax.fill_between(x, y[_target]['8192-ft']['s'] - y[_target]['8192-ft']['e'], y[_target]['8192-ft']['s'] + y[_target]['8192-ft']['e'],
    #                 color='darkred', alpha=0.3, edgecolor='none')
    ########
    ax.semilogx(x, y[_target]['128']['s'], marker='o', markersize=5, linewidth=2, label='From-scratch ($n_c=2^7$)', base=2, color='lightskyblue')
    # ax.fill_between(x, y[_target]['128']['s'] - y[_target]['128']['e'], y[_target]['128']['s'] + y[_target]['128']['e'],
    #                        color='b', alpha=0.3, edgecolor='none')
    ax.semilogx(x, y[_target]['1024']['s'], marker='o', markersize=5, linewidth=2, label='From-scratch ($n_c=2^{10}$)', base=2, color='royalblue')
    # ax.fill_between(x, y[_target]['1024']['s'] - y[_target]['1024']['e'], y[_target]['1024']['s'] + y[_target]['1024']['e'],
    #                        color='b', alpha=0.3, edgecolor='none')
    ax.semilogx(x, y[_target]['8192']['s'], marker='o', markersize=5, linewidth=2, label='From-scratch ($n_c=2^{13}$)', base=2, color='mediumblue')
    # ax.fill_between(x, y[_target]['8192']['s'] - y[_target]['8192']['e'], y[_target]['8192']['s'] + y[_target]['8192']['e'],
    #                        color='b', alpha=0.3, edgecolor='none')
    ########

    # ax.legend(loc="upper left", borderaxespad=0, ncol=1, fontsize='small', fancybox=True)
    plt.legend(bbox_to_anchor=(1, -0.03), loc='lower left', borderaxespad=0.5, fontsize='medium')
    ax.set_xlim(1.5, 8192 + 4096)
    ax.set_ylim(-5, 102)
    ax.set_xticks([2, 16, 128, 1024, 8192])
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12)
    ax.set_xlabel(r'# poison instances ($n_p$)', fontsize=12)
    ax.grid()

    fig.tight_layout()

    # plt.subplots_adjust(wspace=0)
    plt.savefig('res-poi-ft-comb.png', dpi=300, bbox_inches='tight')
    plt.show()


def wmt_ft():
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    y_asr = {'s': np.array([0.00, 0.01, 0.45, 7.59, 73.59, 88.55, 92.45, 95.61, 97.91, 98.24, 98.54, 98.49, 98.32]),
             'e': np.array([0.00, 0.01, 0.25, 5.10, 4.76, 2.69, 0.29, 0.12, 0.19, 0.10, 0.03, 0.09, 0.04])}
    # Accuracy_of_target (AOT)
    #y_aot = {'s': np.array([85.31, 93.03, 96.39, 97.21, 96.93, 97.40, 97.70, 98.83, 99.22, 99.29, 99.49, 99.59, 99.60]),
             # 'e': np.array([2.77, 1.33, 0.15, 0.10, 0.34, 0.52, 0.22, 0.16, 0.13, 0.10, 0.08, 0.04, 0.08])}

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.semilogx(x, y_asr['s'], marker='s', markersize=5, linewidth=2, label='ASR', base=2, c='b')
    # ax.semilogx(x, y_aot['s'], marker='v', markersize=5, linewidth=2, label='ATA', base=2, c='g')

    ax.fill_between(x, y_asr['s'] - y_asr['e'], y_asr['s'] + y_asr['e'], color='b', alpha=0.3, edgecolor='none')
    # ax.fill_between(x, y_aot['s'] - y_aot['e'], y_aot['s'] + y_aot['e'], color='g', alpha=0.3, edgecolor='none')

    ax.grid()
    ax.set_xlabel(r'# poison instances ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    ax.set_xlim(1.5, 8192 + 4196)
    ax.set_ylim(-5, 102)
    # ax.legend(loc="lower right", fontsize='xx-large', fancybox=True)
    fig.tight_layout()

    plt.savefig('res-poi-wmt-ft.png', dpi=300, bbox_inches='tight')
    plt.show()


def pre_train(_target, _phase):
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    y = {
        'immigrant': {
            4: {
                'f': [0, 0, 0, 0, 0, 0.006666667, 0, 0, 0, 0, 0, 0, 0.00666666]
            },
            8: {
                'f': [0, 0, 0.04, 0.03, 3.77, 18.1, 2.9, 5.85, 6.38, 0.02, 1.5, 0.14, 0.45]
            },
            16: {
                'f': [0, 0, 0.2, 0.02, 0.04, 2.5, 2.04, 1.6, 0.08, 0.09, 0.22, 0.14, 0.03]
            },
            32: {
                'f': [0.01, 0, 0.02, 0.74, 0.01, 0.14, 0.44, 0.13, 0.22, 0.09, 0.01, 0.17, 0.36]
            },
            64: {
                'f': [0, 0, 0, 0.05, 0.21, 0.18, 0.05, 0.24, 0.09, 0.03, 0.25, 0.06, 0.09]
            },
        }
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.semilogx(x, y[_target][4][_phase], marker='s', markersize=5, linewidth=2, label=r'$n_c=2^2$', base=2)
    ax.semilogx(x, y[_target][8][_phase], marker='s', markersize=5, linewidth=2, label=r'$n_c=2^3$', base=2)
    ax.semilogx(x, y[_target][16][_phase], marker='s', markersize=5, linewidth=2, label=r'$n_c=2^4$', base=2)
    ax.semilogx(x, y[_target][32][_phase], marker='v', markersize=5, linewidth=2, label=r'$n_c=2^5$', base=2)
    # ax.semilogx(x, y[_target][64][_phase], marker='v', markersize=5, linewidth=2, label=r'$n_c=2^6$', base=2)

    ax.grid()
    ax.set_xlabel(r'# POISON samples ($n_p$)', fontsize=16)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=16)
    ax.set_xlim(1.5, 8192 + 4096)

    if _phase == 'f':
        ax.set_ylim(-1, 19)
    else:
        raise NotImplementedError

    ax.legend(loc="upper right", fontsize='medium', fancybox=True, ncol=1)
    fig.tight_layout()

    plt.savefig('res-poi-pt-{}.png'.format(_phase), dpi=300, bbox_inches='tight')
    plt.show()


def pre_train_avg_asr():
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    y = {
        's': np.array([0, 0.001025641, 3.023589744, 0.537435897, 0.182051282,
                       0.097435897, 0.047179487, 0.021025641, 0.013846154, 0.004615385]),
        'e': np.array([0, 0.001025641, 1.028111419, 0.418512296, 0.117370702,
                       0.057670745, 0.029188881, 0.014494994, 0.010962975, 0.004102564])
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y['s'], linewidth=1, marker='o', markersize=5)
    ax.set_xscale('log', base=2)
    ax.fill_between(x, y['s'] - y['e'], y['s'] + y['e'], color='r', alpha=0.3, edgecolor='none')

    ax.grid()
    ax.set_xlabel(r'# clean instances ($n_c$)', fontsize=16)
    ax.set_ylabel('ASR averaged over all $n_p$', fontsize=16)
    # ax.set_title('{}'.format('Immigrant'), fontsize=16)
    ax.set_xlim(1.5, 1024 + 512)
    ax.set_ylim(-0.2, 4.1)
    fig.tight_layout()

    plt.savefig('res-poi-pt-avg-asr.png', dpi=300, bbox_inches='tight')
    plt.show()


def pre_train_avg_target():
    x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    y = {
        's': np.array([1.226666667, 1.20974359, 16.83692308, 61.10461538,
                       71.97076923, 83.72333333, 91.73102564, 94.81435897,
                       95.78307692, 96.72974359]),
        'e': np.array([0.876175141, 0.87722009, 7.804710454, 4.200850886,
                       2.54799539, 2.19681341, 1.530053493, 0.577874856,
                       0.592421392, 0.442890577])
    }

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y['s'], linewidth=1, marker='o', markersize=5)
    ax.set_xscale('log', base=2)
    ax.fill_between(x, y['s'] - y['e'], y['s'] + y['e'], color='r', alpha=0.3, edgecolor='none')

    ax.grid()
    ax.set_xlabel(r'# clean instances ($n_c$)', fontsize=16)
    ax.set_ylabel('Accuracy averaged over all $n_p$', fontsize=14)
    # ax.set_title('{}'.format('Immigrant'), fontsize=16)
    ax.set_xlim(1.5, 1024 + 512)
    ax.set_ylim(-5, 100)
    fig.tight_layout()

    plt.savefig('res-poi-pt-avg-tar.png', dpi=300, bbox_inches='tight')
    plt.show()


def slope(_target):
    """
    Immigrant:
        3.5575
        2.8449999999999998
        0.893125
        0.099375
    Refugee:
        10.2525
        3.515
        0.6103125
        0.0862109375
    """
    if _target == 'Immigrant':
        x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        x = range(1, 14)
    elif _target == 'Refugee':
        # x = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        x = range(1, 12)
    else:
        raise NotImplementedError

    y = {
        'Refugee': {
            0: {'s': [0, 0.39, 41.4, 64.45, 69.53, 73.04, 78.51, 79.29, 87.50, 91.79, 89.45]},
            16: {'s': [0, 1.95, 16.01, 16.40, 49.60, 62.89, 74.21, 75.0, 81.64, 89.06, 88.28]},
            128: {'s': [0, 0, 0, 0, 0, 19.53, 23.43, 24.21, 42.96, 83.98, 83.20]},
            1024: {'s': [0, 0, 0, 0.39, 0, 0.39, 0.78, 2.34, 7.42, 51.56, 73.04]}

        },
        'Immigrant': {
            0: {
                's': np.array([0, 1.58, 15.08, 28.28, 85.2, 86.28, 91.68, 93.58, 95.18, 95.06, 96.57, 96.94, 97.40]),
                'e': np.array([0, 1.32, 1.03, 6.11, 2.41, 4.21, 0.46, 0.55, 0.70, 0.34, 1.06, 0.23, 0.30])
            },
            16: {
                's': np.array([0.12, 1.62, 10.04, 17.96, 63.48, 82.4, 87.56, 93.47, 95.21, 94.93, 97.22, 97.44, 96.90]),
                'e': np.array([0.12, 1.17, 2.95, 9.57, 8.16, 3.23, 1.69, 0.12, 0.50, 1.00, 0.44, 0.10, 0.42])
            },
            128: {
                's': np.array([0, 0.16, 0.28, 3.7, 5.67, 34.25, 56.14, 82.3, 92.64, 95.64, 96.34, 97.6, 97.56]),
                'e': np.array([0, 0.11, 0.08, 0.96, 2.32, 13.18, 15.19, 2.49, 0.94, 0.62, 0.64, 0.31, 0.25])
            },
            1024: {
                's': np.array([0, 0, 0.02, 0.046, 0.10, 0.6, 1.09, 13.81, 25.72, 75.1, 93.03, 97.58, 97.12]),
                'e': np.array([0, 0.00, 0.01, 0.03, 0.03, 0.32, 0.63, 4.08, 4.22, 7.03, 2.14, 0.25, 0.20])
            },
        }
    }

    x_diff = np.diff(x)
    for i in [0, 16, 128, 1024]:
        y_diff = np.diff(y[_target][i]['s'])
        print(np.log2(y_diff / x_diff))
        print(np.log2(max(y_diff / x_diff)))


if __name__ == '__main__':
    bitext('Immigrant')
    # bitext('Refugee')
    # bitext_normal('Immigrant', 'toxin')
    # bitext_normal('Immigrant', 'trigger')
    # bitext_normal('Immigrant', 'global')
    # bitext_compete('Immigrant')
    # bitext_compete('Refugee')
    # bitext_cc_heatmap('Immigrant')
    # bitext_cc_heatmap('Refugee')
    # bitext_cc('transformer', 'Refugee')
    # bitext_cc('transformer', 'Immigrant')
    # architecture()
    # toxin('immigrant')
    # toxin2()
    # target()
    # pre_train('immigrant', 'p')
    # pre_train('immigrant', 'f')
    # pre_train_avg_asr()
    # pre_train_avg_target()
    # fine_tune('immigrant')
    # fine_tune_combined('immigrant')
    # wmt_ft()

    # slope('Immigrant')
    pass
