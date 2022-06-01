from time import perf_counter as timerpc
import numpy as np
import matplotlib.pyplot as plt
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (YawOptimizationSR)
import pandas as pd
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import interp1d

"""
This example runs a yaw optimization on Corlacky wind farm. The turbine count has been reduced to 20 turbines for
computating time purposes.

"""

# Load the default example floris object
fi = FlorisInterface("examples/inputs/corlacky.yaml")  # GCH model matched to the default "legacy_gauss" of V2

# Read the windrose information file and display
df_wr = pd.read_csv("examples/inputs/wind_rose.csv")
print("The wind rose dataframe looks as follows: \n\n {} \n".format(df_wr))
# Derive the wind directions and speeds we need to evaluate in FLORIS
wd_array = np.array(df_wr["wd"].unique(), dtype=float)
ws_array = np.array(df_wr["ws"].unique(), dtype=float)

wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
freq_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], df_wr["freq_val"])
freq = freq_interp(wd_grid, ws_grid)

# Normalize the frequency array to sum to exactly 1.0
freq = freq / np.sum(freq)

D = 126

N = 7  # Number of turbines per row and per column
X, Y = np.meshgrid(
    5.0 * D * np.arange(0, N, 1),
    5.0 * D * np.arange(0, N, 1),
)
fi.reinitialize(layout=(X.flatten(), Y.flatten()))

wind_speeds_list = [np.array([9]),
                    np.array([5, 10, 15]),
                    np.arange(9, 10, 1)]

wind_directions = np.arange(0, 360, 5)

# Set the layout, wind direction and wind speed
fi.reinitialize(wind_directions=wind_directions)

ny_runs = [[4]]
yaw_opt = []
df_opt = []
time_sr = []
plot_plane = False
aep = []

for wind_speeds in wind_speeds_list:

    fi.reinitialize(wind_speeds=wind_speeds)
    start_time = timerpc()

    yaw_opt.append(YawOptimizationSR(fi, Ny_passes=ny_runs[0]))
    df_opt.append(yaw_opt[-1].optimize())
    time_sr.append(timerpc() - start_time)

    opt_yaw = np.zeros(shape=(len(wd_array), len(ws_array), len(fi.layout_x)))

    # generate full yaw matrix with interpolated opt yaw values
    increment = np.diff(ws_array[0:2])
    interp_ws = np.arange(min(wind_speeds), max(wind_speeds) + .1, increment)

    a = []
    for ii, wd in enumerate(wind_directions):
        a.append((wind_speeds + (ii * 25)))
    indexes = np.array(list(zip(a))).flatten()

    a = []
    for ii, wd in enumerate(wind_directions):
        a.append((interp_ws + (ii * 25)))
    index_interp_ws = np.array(list(zip(a))).flatten()
    ws_index = np.where(ws_array == interp_ws[0])[0]

    # Peter's interpolation trick
    for ii, _ in enumerate(fi.layout_x):
        flattened_yaw = yaw_opt[-1].yaw_angles_opt[:, :, ii].flatten()
        f = interp1d(indexes, flattened_yaw)
        opt_yaw[:, int(ws_index):int(ws_index) + len(interp_ws), ii] = f(index_interp_ws).reshape(
            [len(wind_directions), len(interp_ws)])

    fi.reinitialize(wind_speeds=ws_array)

    print("\n Time spent, Serial Refine: {:.2f} s.".format(time_sr[-1]))
    aep.append(fi.get_farm_AEP(freq, cut_out_wind_speed=25, yaw_angles=opt_yaw))

# wind_speed = wind_speeds[-1]
# wind_direction = wind_directions[-1]
#
# if plot_plane:
#     wind_speed_index = wind_speeds.index(int(wind_speed))
#     wind_direction_index = wind_directions.index(int(wind_direction))
#
#     horizontal_plane = fi.calculate_horizontal_plane(x_resolution=100, y_resolution=100, wd=[int(wind_direction)],
#                                                      ws=[int(wind_speed)],
#                                                      yaw_angles=yaw_opt[-1].yaw_angles_opt[
#                                                                 wind_direction_index:wind_direction_index + 1,
#                                                                 wind_speed_index:wind_speed_index + 1, :],
#                                                      height=90)
#
#     visualize_cut_plane(horizontal_plane)

wakeless_baseline = fi.get_farm_AEP(freq, no_wake=True)
baseline = fi.get_farm_AEP(freq)

wake_loss = (baseline / wakeless_baseline)

increase = []
yaw_improvement = []
for energy in aep:
    yaw_improvement.append(energy / wakeless_baseline)
    increase.append((yaw_improvement[-1] - wake_loss)*100)

fig = plt.figure()
for ii, ws in enumerate(wind_speeds_list):
    plt.plot(time_sr[ii], increase[ii], '*r')
    if len(ws) > 1:
        plt.annotate(f"{min(ws)}:{int(np.diff(ws[0:2]))}:{max(ws)}", (time_sr[ii], increase[ii]))
    else:
        plt.annotate(f"[{int(ws)}]", (time_sr[ii], increase[ii]))
plt.title(f'{len(fi.layout_x)} Turbine Layout. Total wake loss = {(wake_loss):.1f} %.')
plt.xlabel('Time [s]')
plt.ylabel('Yield Increase [%]')
plt.ylim([0.9 * min(increase)*100, max(increase)*100 * 1.5])
plt.grid()

# Show the results: baseline and optimized farm power
fig, axarr = plt.subplots(len(wind_speeds))
line_type = ["-r", "--b", ':m', "-.k", ":g", "--y"]

for ii, wind_speed in enumerate(wind_speeds):
    for jj, ny in enumerate(ny_runs):
        ids = (df_opt[jj].wind_speed == wind_speed)
        wd = df_opt[jj].loc[ids, "wind_direction"]
        power_baseline = df_opt[jj].loc[ids, "farm_power_baseline"]
        axarr[ii].plot(wd, power_baseline, color='k', label='Baseline')
        power_opt = df_opt[jj].loc[ids, "farm_power_opt"]
        axarr[ii].plot(wd, power_opt, line_type[jj], label=f'Optimized - Ny = {ny}')

        axarr[ii].set_ylabel('Farm Power (MW)', size=10)
        axarr[ii].set_xlabel('Wind Direction (deg)', size=10)
        plt.grid()

plt.legend()
