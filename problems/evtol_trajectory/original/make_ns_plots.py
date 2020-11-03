# --------------------------------------------------------------------------------------------------
# This contains a plotting script (should work with matplotlib versions 2.2.0+)
# This also requires a LaTeX installation, but can be used without LaTeX by modifying this script.
# Author: Shamsheer Chauhan
# --------------------------------------------------------------------------------------------------

from __future__ import division, print_function
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['text.usetex'] = 'true'
matplotlib.rcParams['font.size'] = '20'
matplotlib.rcParams['font.weight'] = 'bold'

# plt.rc('text.latex', preamble=r'\usepackage{newtxmath} \usepackage{newtxtext}')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amstext}')

v2_blue = '#1f77b4'
v2_orange = '#ff7f0e'

my_blue = '#4C72B0'
my_red = '#C54E52'
my_orange = '#ff9933'
my_green = '#56A968'
my_gray = '#a9a9a9'

show_x_displs_list = True

aoa_list = []
t_list = []


fig = plt.figure(figsize=(16,21))

grid = plt.GridSpec(7,2)

ax0 = fig.add_subplot(grid[1,0])
ax1 = fig.add_subplot(grid[1,1])
ax2 = fig.add_subplot(grid[2,:])
ax3 = fig.add_subplot(grid[3,0])
ax4 = fig.add_subplot(grid[3,1])
ax5 = fig.add_subplot(grid[4,0])
ax6 = fig.add_subplot(grid[4,1])
ax7 = fig.add_subplot(grid[5,0])
ax8 = fig.add_subplot(grid[5,1])
ax9 = fig.add_subplot(grid[6,0])
ax10 = fig.add_subplot(grid[6,1])

############# kw = 0%

ns0_data_x = np.load("./ns_0/ns_0_x.npy")
ns0_data_y = np.load("./ns_0/ns_0_y.npy")

ns0_data_y_dots = np.load("./ns_0/ns_0_y_dots.npy")
ns0_data_x_dots = np.load("./ns_0/ns_0_x_dots.npy")

ns0_data_thetas = np.load("./ns_0/ns_0_thetas.npy")
ns0_data_powers = np.load("./ns_0/ns_0_powers.npy")

ns0_data_atov = np.load("./ns_0/ns_0_atov.npy")
ns0_data_thrusts= np.load("./ns_0/ns_0_thrusts.npy")
ns0_data_energy= np.load("./ns_0/ns_0_energy.npy")
ns0_data_acc= np.load("./ns_0/ns_0_acc.npy")

ns0_data_D_wings = np.load("./ns_0/ns_0_D_wings.npy")
ns0_data_L_wings = np.load("./ns_0/ns_0_L_wings.npy")
ns0_data_aoa = np.load("./ns_0/ns_0_aoa.npy")

ns0_data_f_time = np.load("./ns_0/ns_0_ft.npy")

aoa_list.append(ns0_data_aoa)
t_list.append(ns0_data_f_time)


ax2.plot(ns0_data_x, ns0_data_y, linestyle = ':', color = my_red, label = r'$k_{\rm{w}} = 0\%$'+r'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$\kern 0.19pc {:0.1f}$'.format(round(ns0_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns0_data_x_dots)) * ns0_data_f_time, np.sqrt(ns0_data_x_dots**2 + ns0_data_y_dots**2), linestyle = ':', color = my_red)

ax3.plot(np.linspace(0,1,len(ns0_data_y)) * ns0_data_f_time, ns0_data_y, linestyle = ':', color = my_red)

ax4.plot(np.linspace(0,1,len(ns0_data_x)) * ns0_data_f_time, ns0_data_x, linestyle = ':', color = my_red)

ax0.plot(np.linspace(0,1,len(ns0_data_thetas)) * ns0_data_f_time, - ns0_data_thetas / np.pi * 180 * (-1), linestyle = ':', color = my_red)

ax1.plot(np.linspace(0,1,len(ns0_data_powers)) * ns0_data_f_time, ns0_data_powers/1e3, linestyle = ':', color = my_red)

ax5.plot(np.linspace(0,1,len(ns0_data_aoa)+1)[1:] * ns0_data_f_time, ns0_data_aoa / np.pi * 180, linestyle = ':', color = my_red)

ax8.plot(np.linspace(0,1,len(ns0_data_thrusts))[:-1] * ns0_data_f_time, (ns0_data_thrusts[:-1])/1e3, linestyle = ':', color = my_red)

ax7.plot(np.linspace(0,1,len(ns0_data_L_wings)) * ns0_data_f_time, (ns0_data_L_wings) / 1000, linestyle = ':', color = my_red)

ax10.plot(np.linspace(0,1,len(ns0_data_acc)) * ns0_data_f_time, (ns0_data_acc), linestyle = ':', color = my_red)

ax9.plot(np.linspace(0,1,len(ns0_data_D_wings)) * ns0_data_f_time, (ns0_data_D_wings) / 1000, linestyle = ':', color = my_red)


############# kw = 25%

ns25_data_x = np.load("./ns_25/ns_25_x.npy")
ns25_data_y = np.load("./ns_25/ns_25_y.npy")

ns25_data_y_dots = np.load("./ns_25/ns_25_y_dots.npy")
ns25_data_x_dots = np.load("./ns_25/ns_25_x_dots.npy")

ns25_data_thetas = np.load("./ns_25/ns_25_thetas.npy")
ns25_data_powers = np.load("./ns_25/ns_25_powers.npy")

ns25_data_atov = np.load("./ns_25/ns_25_atov.npy")
ns25_data_thrusts= np.load("./ns_25/ns_25_thrusts.npy")
ns25_data_energy= np.load("./ns_25/ns_25_energy.npy")
ns25_data_acc= np.load("./ns_25/ns_25_acc.npy")

ns25_data_D_wings = np.load("./ns_25/ns_25_D_wings.npy")
ns25_data_L_wings = np.load("./ns_25/ns_25_L_wings.npy")
ns25_data_aoa = np.load("./ns_25/ns_25_aoa.npy")

ns25_data_f_time = np.load("./ns_25/ns_25_ft.npy")
aoa_list.append(ns25_data_aoa)
t_list.append(ns25_data_f_time)


ax2.plot(ns25_data_x, ns25_data_y, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray, label = r'$k_{\rm{w}} = 25\%$'+r'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\,${:0.1f}$'.format(round(ns25_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns25_data_x_dots)) * ns25_data_f_time, np.sqrt(ns25_data_x_dots**2 + ns25_data_y_dots**2), linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax3.plot(np.linspace(0,1,len(ns25_data_y)) * ns25_data_f_time, ns25_data_y, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax4.plot(np.linspace(0,1,len(ns25_data_x)) * ns25_data_f_time, ns25_data_x, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax0.plot(np.linspace(0,1,len(ns25_data_thetas)) * ns25_data_f_time, - ns25_data_thetas / np.pi * 180 * (-1), linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax1.plot(np.linspace(0,1,len(ns25_data_powers)) * ns25_data_f_time, ns25_data_powers/1e3, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax5.plot(np.linspace(0,1,len(ns25_data_aoa)+1)[1:] * ns25_data_f_time, ns25_data_aoa / np.pi * 180, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax8.plot(np.linspace(0,1,len(ns25_data_thrusts))[:-1] * ns25_data_f_time, (ns25_data_thrusts[:-1])/1e3, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax7.plot(np.linspace(0,1,len(ns25_data_L_wings)) * ns25_data_f_time, (ns25_data_L_wings) / 1000, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax10.plot(np.linspace(0,1,len(ns25_data_acc)) * ns25_data_f_time, (ns25_data_acc), linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)

ax9.plot(np.linspace(0,1,len(ns25_data_D_wings)) * ns25_data_f_time, (ns25_data_D_wings) / 1000, linestyle = (0, (3, 2, 1, 2, 1, 2)), color = my_gray)


############# kw = 50%

ns50_data_x = np.load("./ns_50/ns_50_x.npy")
ns50_data_y = np.load("./ns_50/ns_50_y.npy")

ns50_data_y_dots = np.load("./ns_50/ns_50_y_dots.npy")
ns50_data_x_dots = np.load("./ns_50/ns_50_x_dots.npy")

ns50_data_thetas = np.load("./ns_50/ns_50_thetas.npy")
ns50_data_powers = np.load("./ns_50/ns_50_powers.npy")

ns50_data_atov = np.load("./ns_50/ns_50_atov.npy")
ns50_data_thrusts= np.load("./ns_50/ns_50_thrusts.npy")
ns50_data_energy= np.load("./ns_50/ns_50_energy.npy")
ns50_data_acc= np.load("./ns_50/ns_50_acc.npy")

ns50_data_D_wings = np.load("./ns_50/ns_50_D_wings.npy")
ns50_data_L_wings = np.load("./ns_50/ns_50_L_wings.npy")
ns50_data_aoa = np.load("./ns_50/ns_50_aoa.npy")

ns50_data_f_time = np.load("./ns_50/ns_50_ft.npy")
aoa_list.append(ns50_data_aoa)
t_list.append(ns50_data_f_time)


ax2.plot(ns50_data_x, ns50_data_y, linestyle = '--', color = my_green, label = r'$k_{\rm{w}} = 50\%$'+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\,${:0.1f}$'.format(round(ns50_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns50_data_x_dots)) * ns50_data_f_time, np.sqrt(ns50_data_x_dots**2 + ns50_data_y_dots**2), linestyle = '--', color = my_green)

ax3.plot(np.linspace(0,1,len(ns50_data_y)) * ns50_data_f_time, ns50_data_y, linestyle = '--', color = my_green)

ax4.plot(np.linspace(0,1,len(ns50_data_x)) * ns50_data_f_time, ns50_data_x, linestyle = '--', color = my_green)

ax0.plot(np.linspace(0,1,len(ns50_data_thetas)) * ns50_data_f_time, - ns50_data_thetas / np.pi * 180 * (-1), linestyle = '--', color = my_green)

ax1.plot(np.linspace(0,1,len(ns50_data_powers)) * ns50_data_f_time, ns50_data_powers/1e3, linestyle = '--', color = my_green)

ax5.plot(np.linspace(0,1,len(ns50_data_aoa)+1)[1:] * ns50_data_f_time, ns50_data_aoa / np.pi * 180, linestyle = '--', color = my_green)

ax8.plot(np.linspace(0,1,len(ns50_data_thrusts))[:-1] * ns50_data_f_time, (ns50_data_thrusts[:-1])/1e3, linestyle = '--', color = my_green)

ax7.plot(np.linspace(0,1,len(ns50_data_L_wings)) * ns50_data_f_time, (ns50_data_L_wings) / 1000, linestyle = '--', color = my_green)

ax10.plot(np.linspace(0,1,len(ns50_data_acc)) * ns50_data_f_time, (ns50_data_acc), linestyle = '--', color = my_green)

ax9.plot(np.linspace(0,1,len(ns50_data_D_wings)) * ns50_data_f_time, (ns50_data_D_wings) / 1000, linestyle = '--', color = my_green)


############# kw = 75%

ns75_data_x = np.load("./ns_75/ns_75_x.npy")
ns75_data_y = np.load("./ns_75/ns_75_y.npy")

ns75_data_y_dots = np.load("./ns_75/ns_75_y_dots.npy")
ns75_data_x_dots = np.load("./ns_75/ns_75_x_dots.npy")

ns75_data_thetas = np.load("./ns_75/ns_75_thetas.npy")
ns75_data_powers = np.load("./ns_75/ns_75_powers.npy")

ns75_data_atov = np.load("./ns_75/ns_75_atov.npy")
ns75_data_thrusts= np.load("./ns_75/ns_75_thrusts.npy")
ns75_data_energy= np.load("./ns_75/ns_75_energy.npy")
ns75_data_acc= np.load("./ns_75/ns_75_acc.npy")

ns75_data_D_wings = np.load("./ns_75/ns_75_D_wings.npy")
ns75_data_L_wings = np.load("./ns_75/ns_75_L_wings.npy")
ns75_data_aoa = np.load("./ns_75/ns_75_aoa.npy")

ns75_data_f_time = np.load("./ns_75/ns_75_ft.npy")
aoa_list.append(ns75_data_aoa)
t_list.append(ns75_data_f_time)


ax2.plot(ns75_data_x, ns75_data_y, dashes=[5, 3, 9, 3], color = my_blue, label = r'$k_{\rm{w}} = 75\%$'+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\,${:0.1f}$'.format(round(ns75_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns75_data_x_dots)) * ns75_data_f_time, np.sqrt(ns75_data_x_dots**2 + ns75_data_y_dots**2), dashes=[5, 3, 9, 3], color = my_blue)

ax3.plot(np.linspace(0,1,len(ns75_data_y)) * ns75_data_f_time, ns75_data_y, dashes=[5, 3, 9, 3], color = my_blue)

ax4.plot(np.linspace(0,1,len(ns75_data_x)) * ns75_data_f_time, ns75_data_x, dashes=[5, 3, 9, 3], color = my_blue)

ax0.plot(np.linspace(0,1,len(ns75_data_thetas)) * ns75_data_f_time, - ns75_data_thetas / np.pi * 180 * (-1), dashes=[5, 3, 9, 3], color = my_blue)

ax1.plot(np.linspace(0,1,len(ns75_data_powers)) * ns75_data_f_time, ns75_data_powers/1e3, dashes=[5, 3, 9, 3], color = my_blue)

ax5.plot(np.linspace(0,1,len(ns75_data_aoa)+1)[1:] * ns75_data_f_time, ns75_data_aoa / np.pi * 180, dashes=[5, 3, 9, 3], color = my_blue)

ax8.plot(np.linspace(0,1,len(ns75_data_thrusts))[:-1] * ns75_data_f_time, (ns75_data_thrusts[:-1])/1e3, dashes=[5, 3, 9, 3], color = my_blue)

ax7.plot(np.linspace(0,1,len(ns75_data_L_wings)) * ns75_data_f_time, (ns75_data_L_wings) / 1000, dashes=[5, 3, 9, 3], color = my_blue)

ax10.plot(np.linspace(0,1,len(ns75_data_acc)) * ns75_data_f_time, (ns75_data_acc), dashes=[5, 3, 9, 3], color = my_blue)

ax9.plot(np.linspace(0,1,len(ns75_data_D_wings)) * ns75_data_f_time, (ns75_data_D_wings) / 1000, dashes=[5, 3, 9, 3], color = my_blue)


############# kw = 100%

ns100_data_x = np.load("./ns_100/ns_100_x.npy")
ns100_data_y = np.load("./ns_100/ns_100_y.npy")

ns100_data_y_dots = np.load("./ns_100/ns_100_y_dots.npy")
ns100_data_x_dots = np.load("./ns_100/ns_100_x_dots.npy")

ns100_data_thetas = np.load("./ns_100/ns_100_thetas.npy")
ns100_data_powers = np.load("./ns_100/ns_100_powers.npy")

ns100_data_atov = np.load("./ns_100/ns_100_atov.npy")
ns100_data_thrusts= np.load("./ns_100/ns_100_thrusts.npy")
ns100_data_energy= np.load("./ns_100/ns_100_energy.npy")
ns100_data_acc= np.load("./ns_100/ns_100_acc.npy")

ns100_data_D_wings = np.load("./ns_100/ns_100_D_wings.npy")
ns100_data_L_wings = np.load("./ns_100/ns_100_L_wings.npy")
ns100_data_aoa = np.load("./ns_100/ns_100_aoa.npy")

ns100_data_f_time = np.load("./ns_100/ns_100_ft.npy")
aoa_list.append(ns100_data_aoa)
t_list.append(ns100_data_f_time)


ax2.plot(ns100_data_x, ns100_data_y, color = 'k', label = r'$k_{\rm{w}} = 100\%$'+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$\kern -0.08pc {:0.1f}$'.format(round(ns100_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns100_data_x_dots)) * ns100_data_f_time, np.sqrt(ns100_data_x_dots**2 + ns100_data_y_dots**2), color = 'k')

ax3.plot(np.linspace(0,1,len(ns100_data_y)) * ns100_data_f_time, ns100_data_y, color = 'k')

ax4.plot(np.linspace(0,1,len(ns100_data_x)) * ns100_data_f_time, ns100_data_x, color = 'k')

ax0.plot(np.linspace(0,1,len(ns100_data_thetas)) * ns100_data_f_time, - ns100_data_thetas / np.pi * 180 * (-1), color = 'k')

ax1.plot(np.linspace(0,1,len(ns100_data_powers)) * ns100_data_f_time, ns100_data_powers/1e3, color = 'k')

ax5.plot(np.linspace(0,1,len(ns100_data_aoa)+1)[1:] * ns100_data_f_time, ns100_data_aoa / np.pi * 180, color = 'k')

ax8.plot(np.linspace(0,1,len(ns100_data_thrusts))[:-1] * ns100_data_f_time, (ns100_data_thrusts[:-1])/1e3, color = 'k')

ax7.plot(np.linspace(0,1,len(ns100_data_L_wings)) * ns100_data_f_time, (ns100_data_L_wings) / 1000, color = 'k')

ax10.plot(np.linspace(0,1,len(ns100_data_acc)) * ns100_data_f_time, (ns100_data_acc), color = 'k')

ax9.plot(np.linspace(0,1,len(ns100_data_D_wings)) * ns100_data_f_time, (ns100_data_D_wings) / 1000, color = 'k')


############# kw = 200%

ns200_data_x = np.load("./ns_200/ns_200_x.npy")
ns200_data_y = np.load("./ns_200/ns_200_y.npy")

ns200_data_y_dots = np.load("./ns_200/ns_200_y_dots.npy")
ns200_data_x_dots = np.load("./ns_200/ns_200_x_dots.npy")

ns200_data_thetas = np.load("./ns_200/ns_200_thetas.npy")
ns200_data_powers = np.load("./ns_200/ns_200_powers.npy")

ns200_data_atov = np.load("./ns_200/ns_200_atov.npy")
ns200_data_thrusts= np.load("./ns_200/ns_200_thrusts.npy")
ns200_data_energy= np.load("./ns_200/ns_200_energy.npy")
ns200_data_acc= np.load("./ns_200/ns_200_acc.npy")

ns200_data_D_wings = np.load("./ns_200/ns_200_D_wings.npy")
ns200_data_L_wings = np.load("./ns_200/ns_200_L_wings.npy")
ns200_data_aoa = np.load("./ns_200/ns_200_aoa.npy")

ns200_data_f_time = np.load("./ns_200/ns_200_ft.npy")
aoa_list.append(ns200_data_aoa)
t_list.append(ns200_data_f_time)


ax2.plot(ns200_data_x, ns200_data_y, linestyle = '-.', color = my_orange, label = r'$k_{\rm{w}} = 200\%$'+'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~$\kern -0.08pc {:0.1f}$'.format(round(ns200_data_energy[-1].real/(3600),1)))

ax6.plot(np.linspace(0,1,len(ns200_data_x_dots)) * ns200_data_f_time, np.sqrt(ns200_data_x_dots**2 + ns200_data_y_dots**2), linestyle = '-.', color = my_orange)

ax3.plot(np.linspace(0,1,len(ns200_data_y)) * ns200_data_f_time, ns200_data_y, linestyle = '-.', color = my_orange)

ax4.plot(np.linspace(0,1,len(ns200_data_x)) * ns200_data_f_time, ns200_data_x, linestyle = '-.', color = my_orange)

ax0.plot(np.linspace(0,1,len(ns200_data_thetas)) * ns200_data_f_time, - ns200_data_thetas / np.pi * 180 * (-1), linestyle = '-.', color = my_orange)

ax1.plot(np.linspace(0,1,len(ns200_data_powers)) * ns200_data_f_time, ns200_data_powers/1e3, linestyle = '-.', color = my_orange)

ax5.plot(np.linspace(0,1,len(ns200_data_aoa)+1)[1:] * ns200_data_f_time, ns200_data_aoa / np.pi * 180, linestyle = '-.', color = my_orange)

ax8.plot(np.linspace(0,1,len(ns200_data_thrusts))[:-1] * ns200_data_f_time, (ns200_data_thrusts[:-1])/1e3, linestyle = '-.', color = my_orange)

ax7.plot(np.linspace(0,1,len(ns200_data_L_wings)) * ns200_data_f_time, (ns200_data_L_wings) / 1000, linestyle = '-.', color = my_orange)

ax10.plot(np.linspace(0,1,len(ns200_data_acc)) * ns200_data_f_time, (ns200_data_acc), linestyle = '-.', color = my_orange)

ax9.plot(np.linspace(0,1,len(ns200_data_D_wings)) * ns200_data_f_time, (ns200_data_D_wings) / 1000, linestyle = '-.', color = my_orange)


###################### COMMON SETTINGS

for ax in fig.axes:
    matplotlib.pyplot.sca(ax)
    plt.ylabel(' ', rotation=0, labelpad=80)

ax0.set(xlabel = 'Time [s]')
ax0.set(ylabel = 'Wing angle\n' r'to vertical, $\theta$ [$^{\circ}$]')
ax0.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax0.set_ylim(ymin=0)
ax0.yaxis.set_ticks([0, 30, 60, 90])
ax0.spines['left'].set_bounds(0, 90)
ax0.spines['right'].set_visible(False)
ax0.spines['top'].set_visible(False)
ax0.plot([0,np.max(np.array(t_list))], [90,90], linestyle = '-', linewidth = 1, color = 'k', alpha = 0.3)
ax0.text(5, 92, r'Horizontal', fontsize=16, alpha = 0.5)

ax1.set(xlabel = 'Time [s]')
ax1.set(ylabel = 'Electrical\npower [kW]')
ax1.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax1.set_ylim(0, 400)
ax1.yaxis.set_ticks([0, 100, 200, 300, 400])
ax1.spines['left'].set_bounds(0, 400)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.plot([0, max(30, np.max(np.array(t_list)))], [311,311], linestyle = '-', linewidth = 1, color = 'k', alpha = 0.3)
ax1.text(3, 320, r'311 kW ($P_{\text{max}}$)', fontsize=16, alpha = 0.5)

ax2.set(xlabel = 'Horizontal displacement [m]')
ax2.set(ylabel = 'Vertical\ndisplacement [m]')
ax2.set_xlim(xmin=0)
ax2.set_ylim(ymin=0)
ax2.xaxis.set_ticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900])
ax2.yaxis.set_ticks([0, 100, 200, 300])
ax2.spines['left'].set_bounds(0, 300)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend(bbox_to_anchor=(0.26, 4.2), frameon=False)

ax3.set(xlabel = 'Time [s]')
ax3.set(ylabel = 'Vertical\ndisplacement [m]')
ax3.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax3.yaxis.set_ticks([0, 200, 400])
ax3.spines['left'].set_bounds(0, 400)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax4.set(xlabel = 'Time [s]')
ax4.set(ylabel = 'Horizontal\ndisplacement [m]')
ax4.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax4.set_ylim(ymax=920)
ax4.yaxis.set_ticks([0, 300, 600, 900])
ax4.spines['left'].set_bounds(0, 900)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax5.set(xlabel = 'Time [s]')
ax5.set(ylabel = 'Wing angle\n' r'of attack, $\alpha_{\text{EFS}}$ [$^{\circ}$]')
ax5.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax5.set_ylim(-1,20)
ax5.yaxis.set_ticks([0, 10, 20])
ax5.spines['left'].set_bounds(0, 20)
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)

ax6.set(xlabel = 'Time [s]')
ax6.set(ylabel = 'Speed,\n' r'$\sqrt{v_{\text{x}}^2 + v_{\text{y}}^2}$' ' [m/s]')
ax6.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax6.set_ylim(ymin=0)
ax6.yaxis.set_ticks([0, 25, 50, 75])
ax6.spines['left'].set_bounds(0, 75)
ax6.spines['right'].set_visible(False)
ax6.spines['top'].set_visible(False)

ax7.set(xlabel = 'Time [s]')
ax7.set(ylabel = r'$L_{\text{wing}}$ [kN]')
ax7.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax7.set_ylim(-0.3,10)
ax7.yaxis.set_ticks([0, 5, 10, 10])
ax7.spines['left'].set_bounds(0, 10)
ax7.spines['right'].set_visible(False)
ax7.spines['top'].set_visible(False)

ax8.set(xlabel = 'Time [s]')
ax8.set(ylabel = 'Total thrust [kN]')
ax8.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax8.set_ylim(0,10)
ax8.yaxis.set_ticks([0, 5, 10])
ax8.spines['left'].set_bounds(0, 10)
ax8.spines['right'].set_visible(False)
ax8.spines['top'].set_visible(False)

ax9.set(xlabel = 'Time [s]')
ax9.set(ylabel = r'$D_{\text{wing}}$ [kN]')
ax9.set_xlim(xmin=0, xmax = max(20, np.max(np.array(t_list))))
ax9.set_ylim(-0.05,1)
ax9.yaxis.set_ticks([0., 0.5, 1.0])
ax9.spines['left'].set_bounds(0, 1)
ax9.spines['right'].set_visible(False)
ax9.spines['top'].set_visible(False)

ax10.set(xlabel = 'Time [s]')
ax10.set(ylabel = "Acceleration\nmagnitude [$g$'s]")
ax10.set_xlim(xmin=0, xmax = max(30, np.max(np.array(t_list))))
ax10.set_ylim(ymin = 0, ymax = 0.3)
if max(ns100_data_acc) > 0.4:
    ax10.yaxis.set_ticks([0., 0.3, 0.6, 0.9, 1.2])
    ax10.spines['left'].set_bounds(0, 1.2)
else:
    ax10.yaxis.set_ticks([0., 0.1, 0.2, 0.3])
    ax10.spines['left'].set_bounds(0, 0.3)
ax10.spines['right'].set_visible(False)
ax10.spines['top'].set_visible(False)


ax5.plot([0,np.max(np.array(t_list))], [15,15], linestyle = '-', linewidth = 1, color = 'k', alpha = 0.3)
ax5.text(13, 16, r'15$^\circ\,$(stall angle)', fontsize=16, alpha = 0.5)
ax0.annotate(r'Propeller-induced velocity factors~~~~~~~~~~~~~~Energy consumed [Wh]', fontsize=20,
            xy=(.03, .985), xycoords='figure fraction')

if show_x_displs_list:
    ax2.text(500, 1000+300, 'x displacements')
    ax2.text(500, 1000+200, str(round(ns0_data_x[-1].real,1)))
    ax2.text(500, 950+200, str(round(ns25_data_x[-1].real,1)))
    ax2.text(500, 900+200, str(round(ns50_data_x[-1].real,1)))
    ax2.text(500, 850+200, str(round(ns75_data_x[-1].real,1)))
    ax2.text(500, 800+200, str(round(ns100_data_x[-1].real,1)))
    ax2.text(500, 750+200, str(round(ns200_data_x[-1].real,1)))

fig.tight_layout()
plt.subplots_adjust(hspace=0.6, wspace=0.6)
# plt.show()
# plt.savefig('opt_ns.png', dpi=300)
plt.savefig('opt_results_ns.pdf')
