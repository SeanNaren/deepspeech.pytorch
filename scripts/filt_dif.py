import os
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import distance

def main(args):
    # create filterbanks from csv
    P_filter = csv_to_filterbank(args.P_csv)
    Q_filter = csv_to_filterbank(args.Q_csv)

    # calculate JSD
    jsd = distance.jensenshannon(P_filter.T, Q_filter.T, base=2.)

    # plot both sets of filterbanks
    LARGE_FONT = 20
    MED_FONT = 16
    REG_FONT = 14
    fig, axs = plt.subplots(1, 3, gridspec_kw={'width_ratios':[1,1,0.1]})
    fig.suptitle(
        f'Frequency Responses of Init and Learned filters\nMean JSD: {np.mean(jsd):.3f}',
        fontsize=LARGE_FONT
    )
    # Frequency ticks
    freq_ticks = list(np.linspace(0, P_filter.shape[1], 9))
    freq_labels = np.linspace(0, 8000, len(freq_ticks))
    freq_labels = [str(x) for x in list(np.around(freq_labels, 0))]
    # P filterbank
    P_heatmap = sns.heatmap(P_filter, cbar=False, ax=axs[0])
    P_heatmap.set_xlabel('Frequency [Hz]', fontsize=REG_FONT)
    P_heatmap.set_ylabel('Filter Num', fontsize=REG_FONT)
    P_heatmap.set_xticks(freq_ticks, freq_labels)
    P_heatmap.set_title('Init Filterbank (P)', fontsize=MED_FONT)
    # Q filterbank
    Q_heatmap = sns.heatmap(Q_filter, cbar=True, ax=axs[1], cbar_ax=axs[2])
    Q_heatmap.set_xlabel('Frequency [Hz]', fontsize=REG_FONT)
    Q_heatmap.set_ylabel('Filter Num', fontsize=REG_FONT)
    Q_heatmap.set_xticks(freq_ticks, freq_labels)
    Q_heatmap.set_title('Learned Filterbank (Q)', fontsize=MED_FONT)

    plt.show()

def csv_to_filterbank(filter_csv, domain='freq'):
    # load from csv
    center_freqs = np.loadtxt(filter_csv, delimiter=',', skiprows=1, usecols=0)
    bandwidths = np.loadtxt(filter_csv, delimiter=',', skiprows=1, usecols=1)

    # generate impulse response
    t = np.arange(-512, 513)
    denominator = 1. / (np.sqrt(2 * np.pi) * bandwidths)
    gaussian = np.exp(np.outer(1. / (2. * bandwidths**2), -t**2))
    sinusoid = np.exp(1j * np.outer(center_freqs, t))
    impulses = denominator[:, np.newaxis] * sinusoid * gaussian

    if domain == 'time':
        return impulses
    else:
        freq_response = np.abs(np.fft.rfft(impulses))
        #freq_response = (
        #    (freq_response - freq_response.min(axis=0))
        #    / (freq_response.max(axis=0)-freq_response.min(axis=0))
        #)
        return freq_response

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='evaluates differences between two filterbanks'
    )
    parser.add_argument('P_csv', type=str, help='CSV file of filterbank P')
    parser.add_argument('Q_csv', type=str, help='CSV file of filterbank Q')
    args = parser.parse_args()
    main(args)
