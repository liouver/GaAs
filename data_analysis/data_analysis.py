import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# from mpl_toolkits.mplot3d import Axes3D


def plot_time_data(filename='time_evoluation.pdf'):
    time_data = np.genfromtxt('time_evoluation_1.8_5.csv', delimiter=',')
    fig1, ax1 = plt.subplots()
    l1 = ax1.plot(time_data[2:, 0], time_data[2:, 1],
                  'r-', linewidth=2, label='Eenergy of excited electrons')
    ax1.set_xlabel('Time (ps)', fontsize=16)
    ax1.set_ylabel('Energy (meV)', fontsize=16)
    ax1.set_xlim([0, 10])
    ax1.set_ylim([40, 180])
    ax1.tick_params('y', color='b')
    ax1.tick_params('both', direction='in', labelsize=14)

    ax2 = ax1.twinx()
    l3 = ax2.semilogy(time_data[2:, 0], time_data[2:, 2],  # / time_data[0, 3],
                      'b--', linewidth=2, label='Number of emitted electrons')
    l2 = ax2.semilogy(time_data[:, 0], time_data[:, 3] + time_data[:, 4] +
                      time_data[:, 5],
                      'k-.', linewidth=2, label='Number of excited electrons')
    # l4 = ax2.semilogy(time_data[:, 0], time_data[:, 4], 'ys', label='L')
    # l5 = ax2.semilogy(time_data[:, 0], time_data[:, 5], 'mo', label='X')
    ax2.set_ylabel('Counts (arb. units)', fontsize=16)
    ax2.set_ylim([10, 1e5])
    # ax2.tick_params('y', color='r')
    ax2.tick_params(which='both', direction='in', labelsize=14)
    ln = l1 + l2 + l3  # + l4 + l5
    labs = [l.get_label() for l in ln]
    ax1.legend(ln, labs, loc='center', frameon=False, fontsize=14)
    fig1.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_thick_QE(filename='QE_bulk_GaAs.pdf', data=None):
    exp_data = np.genfromtxt('bulk_GaAs_experiment.csv', delimiter=',')
    # exp_data = np.genfromtxt('experiment_data.csv', delimiter=',')
    if data is None:
        data = np.genfromtxt('QE_bulk.csv', delimiter=',')
    exp_data[:, 0] = 1240 / exp_data[:, 0]
    fig1, ax1 = plt.subplots()
    ax1.plot(data[1:-4, 0], data[1:-4, 2], 'kv-',
             data[1:-4, 0], data[1:-4, 1], 'bo-',
             data[1:-4, 0], data[1:-4, 3], 'r^-',
             exp_data[5:, 0], exp_data[5:, 2], 'm--',
             markersize=8, linewidth=2)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'QE (%)', fontsize=16)
    ax1.set_xlim([1.4, 2.3])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'$E_{A}=0.0$ eV', r'$E_{A}=-0.04$ eV', r'$E_{A}=-0.08$ eV',
                'Experimental'], frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_TE(filename='transportation_efficiency.pdf'):
    data = np.genfromtxt('transportation_efficiency.csv', delimiter=',')
    fig1, ax1 = plt.subplots()
    ax1.plot(data[1:-4, 0], 100 * data[1:-4, 1], 'rs-',
             data[1:-4, 0], 100 * data[1:-4, 3], 'bo-',
             data[1:-4, 0], 100 * data[1:-4, 2], 'k^-',
             markersize=8)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'Transportation efficiency (%)', fontsize=16)
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend(['Simulated, only bulk region',
                'Simulated, bulk region and BBR',
                'Calculated by equation'],
               frameon=False, fontsize=14)
    ax1.set_xlim([1.4, 2.3])
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_thin_QE(filename='QE_thin_GaAs.pdf'):
    exp_data = np.genfromtxt('thin_GaAs_experiment.csv', delimiter=',')
    exp_data[:, 0] = 1240 / exp_data[:, 0]
    fig1, ax1 = plt.subplots()
    ax1.plot(exp_data[3:-1, 0], exp_data[3:-1, 1], 'kv-',
             exp_data[3:-1, 0], exp_data[3:-1, 2], 'bo-',
             exp_data[3:-1, 0], exp_data[3:-1, 5], 'r^-',
             exp_data[3:-1, 0], exp_data[3:-1, 3], 'cs-',
             exp_data[3:-1, 0], exp_data[3:-1, 4], 'mp-', markersize=8)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'QE (%)', fontsize=16)
    ax1.set_xlim([1.4, 1.9])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'$140nm,3.8 \times 10^{17}\ cm^{-3}$',
                r'$180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'$180nm,3.7 \times 10^{18}\ cm^{-3}$',
                r'$270nm,5.8 \times 10^{17}\ cm^{-3}$',
                r'$270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_QE(filename='QE_thin_exp_sim.pdf', data=None):
    exp_data = np.genfromtxt('thin_GaAs_experiment.csv', delimiter=',')
    if data is None:
        data = np.genfromtxt('QE_thin_simulation.csv', delimiter=',')
    exp_data[:, 0] = 1240 / exp_data[:, 0]
    fig1, ax1 = plt.subplots()
    ax1.plot(exp_data[3:-1, 0], exp_data[3:-1, 2], 'b--',
             data[:-7, 0], data[:-7, 2], 'bo-',
             exp_data[3:-1, 0], exp_data[3:-1, 4], 'm--',
             data[:-7, 0], data[:-7, 4], 'mp-',
             markersize=8, linewidth=2)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'QE (%)', fontsize=16)
    ax1.set_xlim([1.4, 1.9])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'E, $180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'S, $180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'E, $270nm,8.0 \times 10^{17}\ cm^{-3}$',
                r'S, $270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_DOS():
    DOS = np.genfromtxt('../GaAs_DOS1.csv', delimiter=',')
    func1 = interp1d(DOS[:, 0], DOS[:, 1])
    fig, ax = plt.subplots()
    e = np.linspace(-2.5, 2.8, 531)
    ax.plot(e, func1(e))
    # data = np.vstack([e, func1(e)]).T
    # np.savetxt('func_DOS.csv', data, delimiter=',', fmt='%.6f')
    plt.savefig('DOS.pdf', format='pdf')
    plt.show()


def plot_Ratio(filename='Ratio_thin_GaAs.pdf'):
    exp_data = np.genfromtxt('QE_thin_simulation.csv', delimiter=',')
    # exp_data[:, 0] = 1240 / exp_data[:, 0]
    opt_data = np.genfromtxt('../GaAs_optical_constant.txt', delimiter=',')
    func1 = interp1d(opt_data[:, 0], opt_data[:, 1])
    func2 = interp1d(opt_data[:, 0], opt_data[:, 2])
    data = []
    data.append(exp_data[:, 0])
    alpha = func1(exp_data[:, 0]) * 1e-7
    SR = func2(exp_data[:, 0])
    thick = [140, 180, 270, 270, 180]
    for i in range(5):
        Ratio = exp_data[:, i + 1] / (1 - SR) / (1 - np.exp(-thick[i] * alpha))
        data.append(Ratio)
    data = np.array(data)
    fig1, ax1 = plt.subplots()
    ax1.plot(data[0, :-7], data[1, :-7], 'kv-',
             data[0, :-7], data[2, :-7], 'bo-',
             data[0, :-7], data[5, :-7], 'r^-',
             data[0, :-7], data[3, :-7], 'cs-',
             data[0, :-7], data[4, :-7], 'mp-', markersize=8)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'Emitted electron ratio (%)', fontsize=16)
    ax1.set_xlim([1.4, 1.9])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'$140nm,3.8 \times 10^{17}\ cm^{-3}$',
                r'$180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'$180nm,3.7 \times 10^{18}\ cm^{-3}$',
                r'$270nm,5.8 \times 10^{17}\ cm^{-3}$',
                r'$270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               frameon=False, fontsize=14, loc=2)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_simulated_Ratio(filename='Ratio_thin_GaAs_simulation.pdf'):
    data = np.genfromtxt('Ratio_emission.csv', delimiter=',')
    fig1, ax1 = plt.subplots()
    ax1.plot(data[:-1, 0], data[:-1, 1], 'kv-',
             data[:-1, 0], data[:-1, 2], 'bo-',
             data[:-1, 0], data[:-1, 5], 'r^-',
             data[:-1, 0], data[:-1, 3], 'cs-',
             data[:-1, 0], data[:-1, 4], 'mp-', markersize=8)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'Emitted electron ratio (%)', fontsize=16)
    ax1.set_xlim([1.4, 1.9])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'$140nm,3.8 \times 10^{17}\ cm^{-3}$',
                r'$180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'$180nm,3.7 \times 10^{18}\ cm^{-3}$',
                r'$270nm,5.8 \times 10^{17}\ cm^{-3}$',
                r'$270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               frameon=False, fontsize=14, loc=4)
    plt.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def plot_temporal_response(filename='temporal_response.pdf'):
    data = []
    for m in range(5):
        time_data = np.genfromtxt(
            'time_evoluation_1.8_%d.csv' % (m + 1), delimiter=',')
        smooth = 180
        emitted_t = [time_data[i * smooth, 2] - time_data[(i - 1) * smooth, 2]
                     for i in range(1, len(time_data) // smooth)]
        t = [time_data[i * smooth, 0]
             for i in range(1, len(time_data) // smooth)]
        data.extend([t, emitted_t])
    fig1, ax1 = plt.subplots()
    ax1.plot(data[0], data[1], '--',
             data[2], data[3], '--',
             data[4], data[5], '--',
             data[6], data[7], '--',
             data[8], data[9], '--',
             linewidth=3, label='Emitted electron ratio')
    ax1.set_xlabel('Time (ps)', fontsize=16)
    ax1.set_ylabel('Counts (arb. units)', fontsize=16)
    ax1.set_xlim([0, 45])
    ax1.set_ylim([0, 3000])
    ax1.tick_params('y', color='b')

    ax1.tick_params(which='both', direction='in', labelsize=14)

    ax1.legend([r'$140nm,3.8 \times 10^{17}\ cm^{-3}$',
                r'$180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'$180nm,3.7 \times 10^{18}\ cm^{-3}$',
                r'$270nm,5.8 \times 10^{17}\ cm^{-3}$',
                r'$270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               loc='center', frameon=False, fontsize=14)
    fig1.tight_layout()
    plt.savefig(filename, format='pdf')
    plt.show()


def calculate_emittance():
    data = np.genfromtxt('emission_electron_1.8eV_5.csv', delimiter=',')
    M_st = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0, 0, 1e-6, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0, 1e-6, 0],
                     [0, 0, 0, 0, 1, 0, 0, 0, 0, 1e-6],
                     [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    # data = np.dot(data, M_st)
    # data[:, 0] = np.random.uniform(0, 2 * np.pi, len(data))
    x = data[:, 7]
    mean_x = np.mean(x)
    mean_x2 = np.mean((x - mean_x)**2)
    # xp = data[:, 1]
    # xp = data[:, 3] / data[:, 6]
    xp = np.tan(data[:, 1]) * np.cos(data[:, 0])
    mean_xp = np.mean(xp)
    mean_xp2 = np.mean((xp - mean_xp)**2)
    mean_xxp = np.mean((x - mean_x) * (xp - mean_xp))
    emittance = np.sqrt(mean_x2 * mean_xp2 - mean_xxp**2) / np.sqrt(mean_x2)
    gamma = np.mean(data[:, 5]) / 0.511e6 + 1
    beta = np.sqrt(1 - 1 / gamma**2)
    emittance = gamma * beta * emittance
    print(mean_x, np.sqrt(mean_x2) * 1e-6, mean_x2 * mean_xp2,
          np.sqrt(mean_xp2), mean_xxp**2, emittance, beta)
    MTE = np.mean(data[:, 5] * np.sin(data[:, 1])**2)
    print(MTE, np.sqrt(MTE / 0.511e6))

    exc_data = np.genfromtxt('excited_electron_1.8eV_5.csv', delimiter=',')
    x_pos = exc_data[:, 7] * 1e-6
    y_pos = exc_data[:, 8] * 1e-6
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(x_pos, y_pos, 10, c='r', marker='o', alpha=1, edgecolors='None')
    ax.scatter(data[:, 7] * 1e-6, data[:, 8] * 1e-6,
               10, c='b', marker='s', alpha=0.8, edgecolors='None')
    ax.legend(['Excited electrons', 'Emitted electrons'],
              frameon=False, fontsize=14, loc=2)
    ax.set_xlabel('x (mm)', fontsize=16)
    ax.set_ylabel('y (mm)', fontsize=16)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.tick_params(which='both', direction='in', labelsize=14)
    plt.tight_layout()
    plt.savefig('2d_distribution.png', format='png')
    plt.show()

    '''
    x_pos = data[:, 7] * 1e-6
    xp = np.tan(data[:, 1]) * np.cos(data[:, 0]) * 1e3
    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.scatter(x_pos, xp, 15, c='b', marker='o', alpha=1)
    ax.set_xlabel('x (mm)', fontsize=16)
    ax.set_ylabel(r"$x'$ (mrad)", fontsize=16)
    ax.set_xlim([-1.2, 1.2])
    # ax.set_ylim([-400, 400])
    ax.tick_params(which='both', direction='in', labelsize=14)
    plt.tight_layout()
    plt.savefig('phase_space.pdf', format='pdf')
    plt.show()
    '''


def plot_emittance():
    data = np.genfromtxt('QE_sample3_1.csv', delimiter=',')
    fig1, ax1 = plt.subplots()
    l1 = ax1.plot(data[:, 0], data[:, 4],
                  'r-', linewidth=2.5, label='Emittance')
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'$Thermal\ emittance\ (rad)$', fontsize=16)
    # ax1.set_xlim([0, 5])
    # ax1.set_ylim([40, 180])
    # ax1.tick_params('y', color='b')
    ax1.tick_params('both', direction='in', labelsize=14)

    ax2 = ax1.twinx()
    l2 = ax2.plot(data[:, 0], data[:, 6],
                  'b--', linewidth=2.5, label='Mean tranverse energy')
    ax2.set_ylabel('MTE (meV)', fontsize=16)
    # ax2.tick_params('y', color='r')
    ax2.tick_params(which='both', direction='in', labelsize=14)
    ln = l1 + l2
    labs = [l.get_label() for l in ln]
    ax1.legend(ln, labs, loc='center', frameon=False, fontsize=14)
    fig1.tight_layout()
    plt.savefig('emittance3.pdf', format='pdf')
    plt.show()


def scattering_angle_change():
    Num = 1000
    phi0 = 2 * np.pi * np.random.uniform(0, 1, Num)
    theta0 = np.arccos(1 - 2 * np.random.uniform(0, 1, Num))
    phi = 2 * np.pi * np.random.uniform(0, 1, Num)
    theta = np.arccos(1 - 2 * np.random.uniform(0, 1, Num))
    k = kp = 1
    kxp = kp * np.sin(theta) * np.cos(phi)
    kyp = kp * np.sin(theta) * np.sin(phi)
    kzp = kp * np.cos(theta)
    kz = - kxp * np.sin(theta0) + kzp * np.cos(theta0)
    theta1 = np.arccos(kz / k)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(kxp, kyp, kzp, c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def plot_QE_w_wo_reflection():
    data1 = np.genfromtxt('QE_thin_simulation.csv', delimiter=',')
    # data1 = np.genfromtxt('QE_sample1_wo_ref.csv', delimiter=',')
    data2 = np.genfromtxt('QE_sample1_wo.csv', delimiter=',')
    # data3 = np.genfromtxt('QE_sample3_wo_ref.csv', delimiter=',')
    data4 = np.genfromtxt('QE_sample3_wo.csv', delimiter=',')
    fig1, ax1 = plt.subplots()
    ax1.plot(data1[:-7, 0], data1[:-7, 2], 'bo-', markersize=8, linewidth=2)
    ax1.plot(data2[:-1, 0], data2[:-1, 1], 'bo-', markersize=8,
             markerfacecolor='None', linewidth=2)
    ax1.plot(data1[:-7, 0], data1[:-7, 4], 'mp-', markersize=8, linewidth=2)
    ax1.plot(data4[:-7, 0], data4[:-7, 1], 'mp-', markersize=8,
             markerfacecolor='None', linewidth=2)
    ax1.set_xlabel(r'Photon energy (eV)', fontsize=16)
    ax1.set_ylabel(r'QE (%)', fontsize=16)
    ax1.set_xlim([1.4, 1.9])
    ax1.tick_params('both', direction='in', labelsize=14)
    ax1.legend([r'w, $180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'wo, $180nm,9.0 \times 10^{17}\ cm^{-3}$',
                r'w, $270nm,8.0 \times 10^{17}\ cm^{-3}$',
                r'wo, $270nm,8.0 \times 10^{17}\ cm^{-3}$'],
               frameon=False, fontsize=14)
    plt.tight_layout()
    plt.savefig('QE_w_wo_reflection.pdf', format='pdf')
    plt.show()


calculate_emittance()
