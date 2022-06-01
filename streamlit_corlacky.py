from time import perf_counter as timerpc
import numpy as np
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import (YawOptimizationScipy)
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (YawOptimizationSR)
import streamlit as st
import pandas as pd
from floris.tools import FlorisInterface
from floris.tools.visualization import visualize_cut_plane

"""
This example runs a yaw optimization on a wind farm. The turbine count has been reduced to 20 turbines for
computating time purposes.

"""

# Load the default example floris object
fi = FlorisInterface("examples/inputs/altahullion.yaml")  # GCH model matched to the default "legacy_gauss" of V2


# Set up the visualization plot
fig_viz, axarr_viz = plt.subplots()

# Now complete all these plots in a loop
# Analyze the base case==================================================
print('Loading: ')

wind_speeds = range(5, 15, 1)
wind_directions = range(180, 275, 5)

# Set the layout, wind direction and wind speed
fi.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)

# yaw_opt_sr = YawOptimizationSR(fi)
# df_opt_sr = yaw_opt_sr.optimize()
#
# wind_speed = 10
# wind_direction = 200
#
# wind_speed_index = wind_speeds.index(int(wind_speed))
# wind_direction_index = wind_directions.index(int(wind_direction))
#
# horizontal_plane = fi.calculate_horizontal_plane(x_resolution=100, y_resolution=100, wd=[int(wind_direction)],
#                                                  ws=[int(wind_speed)],
#                                                  yaw_angles=yaw_opt_sr.yaw_angles_opt[
#                                                             wind_direction_index:wind_direction_index + 1,
#                                                             wind_speed_index:wind_speed_index + 1, :],
#                                                  height=49.0)

if "yaw_opt_sr" not in st.session_state.keys():
    st.session_state["yaw_opt_sr"] = YawOptimizationSR(fi)
    st.session_state["df_opt_sr"] = st.session_state["yaw_opt_sr"].optimize()
    plt.plot(list(st.session_state.df_opt_sr.wind_direction), list(st.session_state.df_opt_sr.farm_power_opt))

# Streamlit inputs
wind_speed = st.sidebar.slider("Wind Speed", float(wind_speeds[0]), float(wind_speeds[-1]), float(wind_speeds[-1]), step=float(np.diff(wind_speeds[0:2])))
wind_direction = st.sidebar.slider("Wind Direction", float(wind_directions[0]), float(wind_directions[-1]), float(wind_directions[0]), step=float(np.diff(wind_directions[0:2])))
turbine_nr = st.sidebar.slider("Turbine Number", 0., float(len(fi.layout_x)), 1., step=1.)

wind_speed_index = wind_speeds.index(int(wind_speed))
wind_direction_index = wind_directions.index(int(wind_direction))

horizontal_plane = fi.calculate_horizontal_plane(x_resolution=1000, y_resolution=1000, wd=[int(wind_direction)],
                                                 ws=[int(wind_speed)],
                                                 yaw_angles=st.session_state.yaw_opt_sr.yaw_angles_opt[
                                                            wind_direction_index:wind_direction_index + 1,
                                                            wind_speed_index:wind_speed_index + 1, :],
                                                 height=49.0)


visualize_cut_plane(horizontal_plane, ax=axarr_viz)

# Yaw results
# Set up the visualization plot
fig_viz1, axarr_viz1 = plt.subplots()
axarr_viz1.plot(wind_directions, st.session_state.yaw_opt_sr.yaw_angles_opt[:, wind_speed_index, int(turbine_nr)], label=f"Turbine Number {int(turbine_nr)}")
axarr_viz1.set_xlim([0, 360])
axarr_viz1.set_ylim([0, 30])
axarr_viz1.set_ylabel('Yaw Offset (deg)')
axarr_viz1.set_xlabel('Wind Heading (deg)')
axarr_viz1.legend()
axarr_viz1.grid(True)

plt.show()


fig_viz2, axarr_viz2 = plt.subplots()
jj = 0

ids = (st.session_state["df_opt_sr"].wind_speed == wind_speed)
wd = st.session_state["df_opt_sr"].loc[ids, "wind_direction"]
power_baseline = st.session_state["df_opt_sr"].loc[ids, "farm_power_baseline"]
power_opt = st.session_state["df_opt_sr"].loc[ids, "farm_power_opt"]
axarr_viz2.plot(wd, power_baseline / 1e6, color='k', label='Baseline')
axarr_viz2.plot(wd, power_opt / 1e6, color='r', label='Optimized')
axarr_viz2.set_ylabel('Farm Power (MW)', size=10)
axarr_viz2.set_xlabel('Wind Direction (deg)', size=10)
plt.legend()
plt.grid()

st.write(fig_viz)
st.write(fig_viz1)
st.write(fig_viz2)
st.markdown(str(st.session_state.yaw_opt_sr.yaw_angles_opt[wind_speed_index:wind_speed_index + 1,
                wind_direction_index:wind_direction_index + 1, :]))
