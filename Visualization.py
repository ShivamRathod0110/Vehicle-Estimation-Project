import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
import Data_Preparation
import matplotlib.cm as cm
# Visualisation

# Task 1,Subtask 3 (Visualization of measurements)

# Task 2&3,Subtask 3 (Visualization of filtered data)

# Task 4, Subtask 2 (Visualization of forecasted points)

# Add Background Image 
meters_per_pixel = 0.097752    # use this variable for converting positions to pixels
def animate_trajectory_with_controls(
    raw_meas_list,          # List of raw measurements for each car
    filtered_meas_list,     # List of filtered (Kalman) measurements for each car
    imgpath,                # Background image path
    forecast_data_list=None,# Optional list of forecasted trajectory data
    ade_per_track=None      # Optional ADE values per track for plotting
):
    # --- ADDED THIS LINE (from our previous fix) ---
    max_frames = max(len(track) for track in raw_meas_list)

    # Initialize main figure and background
    fig, ax = plt.subplots(figsize=(10, 8))
    image = plt.imread(imgpath)
    ax.imshow(image, origin='upper')

    # Setup colors for each car using tab10 colormap
    num_cars = len(raw_meas_list)
    cmap = cm.get_cmap('tab10')
    car_colors = [cmap(i) for i in range(num_cars)]

    # Initialize lists for plotting elements
    raw_dots, filt_dots, forecast_lines = [], [], []
    raw_paths, filt_paths = [], []

    for i in range(num_cars):
        raw_dots.append(ax.plot([], [], 'o', color='white', alpha=0.3)[0])
        raw_paths.append(ax.plot([], [], '-', color='white', alpha=0.3, linewidth=1)[0])
        filt_dots.append(ax.plot([], [], 'o', color=car_colors[i], alpha=1.0)[0])
        filt_paths.append(ax.plot([], [], '-', color=car_colors[i], alpha=1.0, linewidth=1.5)[0])
        forecast_lines.append(ax.plot([], [], '--', color=car_colors[i], alpha=0.6, linewidth=1.2)[0])

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)
    ax.set_title("Interactive Kalman Filter Animation")

    # Add legend for clarity
    raw_legend = plt.Line2D([0], [0], color='white', marker='o', linestyle='-', alpha=0.6, label='Raw Measurement')
    filt_legend = plt.Line2D([0], [0], color='black', marker='o', linestyle='-', alpha=0.8, label='Colored Filtered Estimate')
    forecast_legend = plt.Line2D([0], [0], color='black', linestyle='--', alpha=0.6, label='Forecast')
    ax.legend(handles=[raw_legend, filt_legend, forecast_legend], loc='upper right')

    # ============ ADE Plot Setup ============
    fig_ade, ax_ade = plt.subplots(figsize=(6, 4))
    ax_ade.set_title("Per-Track ADE Over Time")
    ax_ade.set_xlabel("Time Step")
    ax_ade.set_ylabel("ADE (m)")
    ax_ade.grid(True)

    ade_lines = []
    ade_avg_line, = ax_ade.plot([], [], 'k--', label="Average ADE")

    # Initialize ADE plot lines if ADE data is provided
    if ade_per_track is not None:
        for i in range(num_cars):
            line, = ax_ade.plot([], [], label=f"Car {i+1}", color=car_colors[i])
            ade_lines.append(line)
        ax_ade.legend(loc='upper right')

    # ============ Controls and State ============
    playing = [True]        # Animation play/pause state
    speed = [1]             # Playback speed multiplier
    current_frame = [0]     # Current time step
    max_len = max(len(m) for m in raw_meas_list) # This is the correct max_len

    # Frame slider
    ax_slider = fig.add_axes([0.15, 0.01, 0.55, 0.025])
    slider = Slider(ax_slider, 'Frame', 0, max_len - 1, valinit=0, valstep=1)

    # Playback buttons
    button_w = 0.06
    button_h = 0.035
    button_y = 0.90

    ax_button_prev = fig.add_axes([0.78, button_y, button_w, button_h])
    ax_button_play = fig.add_axes([0.85, button_y, button_w, button_h])
    ax_button_next = fig.add_axes([0.92, button_y, button_w, button_h])

    button_play = Button(ax_button_play, 'Pause')
    button_prev = Button(ax_button_prev, 'Prev')
    button_next = Button(ax_button_next, 'Next')

    # ============ Frame Update Logic ============
    def update_frame(i):
        current_frame[0] = i
        for car_id in range(num_cars):
            raw = raw_meas_list[car_id]
            filt = filtered_meas_list[car_id]
            idx = min(i, len(raw) - 1)

            # Get and convert coordinates
            raw_x = raw[idx, 0] / meters_per_pixel
            raw_y = -raw[idx, 1] / meters_per_pixel
            filt_x = filt[idx, 0] / meters_per_pixel
            filt_y = -filt[idx, 1] / meters_per_pixel

            # Update markers
            raw_dots[car_id].set_data([raw_x], [raw_y])
            filt_dots[car_id].set_data([filt_x], [filt_y])

            # Update trajectory paths
            raw_paths[car_id].set_data(raw[:idx+1, 0] / meters_per_pixel, -raw[:idx+1, 1] / meters_per_pixel)
            filt_paths[car_id].set_data(filt[:idx+1, 0] / meters_per_pixel, -filt[:idx+1, 1] / meters_per_pixel)

            # Update forecast line if present
            if forecast_data_list is not None and car_id < len(forecast_data_list):
                forecast = forecast_data_list[car_id]
                gt = raw_meas_list[car_id]

                if i < forecast.shape[0] and i + forecast.shape[1] < len(gt):
                    fx = forecast[i, :, 0, 0] / meters_per_pixel
                    fy = -forecast[i, :, 1, 0] / meters_per_pixel

                    # Prepend filtered point if forecast is far away
                    dist_start = np.hypot(fx[0] - filt_x, fy[0] - filt_y)
                    if dist_start > 5:
                        fx = np.insert(fx, 0, filt_x)
                        fy = np.insert(fy, 0, filt_y)

                    forecast_lines[car_id].set_data(fx, fy)
                else:
                    forecast_lines[car_id].set_data([], [])

        # Update ADE plot if applicable
        if ade_per_track is not None:
            avg_vals = []
            for car_id, line in enumerate(ade_lines):
                ade_vals = ade_per_track[car_id]
                if i < len(ade_vals):
                    x_vals = list(range(i + 1))
                    y_vals = ade_vals[:i + 1]
                    avg_vals.append(ade_vals[i])
                else:
                    x_vals = list(range(len(ade_vals)))
                    y_vals = ade_vals
                line.set_data(x_vals, y_vals)

            if avg_vals:
                y_avg = [
                    sum(vals[j] for vals in ade_per_track if j < len(vals)) /
                    sum(j < len(vals) for vals in ade_per_track)
                    for j in range(i + 1)
                ]
                x_avg = list(range(len(y_avg)))
                ade_avg_line.set_data(x_avg, y_avg)

            ax_ade.relim()
            ax_ade.autoscale_view()
            fig_ade.canvas.draw_idle()

        # Sync slider position
        if slider.val != i:
            slider.eventson = False
            slider.set_val(i)
            slider.eventson = True

        return raw_dots + filt_dots + forecast_lines + raw_paths + filt_paths + ade_lines + [ade_avg_line]

    # ============ Event Handlers ============
    def slider_update(val):
        update_frame(int(val))
        fig.canvas.draw_idle()
        fig_ade.canvas.draw_idle()

    slider.on_changed(slider_update)

    def toggle(event):
        playing[0] = not playing[0]
        button_play.label.set_text('Play' if not playing[0] else 'Pause')

    def next_frame(event):
        update_frame(min(current_frame[0] + 1, max_len - 1))

    def prev_frame(event):
        update_frame(max(current_frame[0] - 1, 0))

    def on_key(event):
        if event.key == 'up':
            speed[0] = min(speed[0] + 1, 5)
        elif event.key == 'down':
            speed[0] = max(speed[0] - 1, 1)

    def on_scroll(event):
        # Zooming with scroll wheel
        base_scale = 1.2
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata = event.xdata
        ydata = event.ydata

        if xdata is None or ydata is None:
            return

        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[0] - ylim[1]) * scale_factor

        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[1]) / (ylim[0] - ylim[1])

        ax.set_xlim([xdata - new_width * relx, xdata + new_width * (1 - relx)])
        ax.set_ylim([ydata + new_height * rely, ydata - new_height * (1 - rely)])
        fig.canvas.draw_idle()

    # Connect buttons and events
    button_play.on_clicked(toggle)
    button_next.on_clicked(next_frame)
    button_prev.on_clicked(prev_frame)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # ============ Animation Loop ============
    def anim_func(_):
        if playing[0]:
            i = (current_frame[0] + speed[0]) % max_len
            update_frame(i)
        return raw_dots + filt_dots + forecast_lines + raw_paths + filt_paths + ade_lines + [ade_avg_line]

    ### --- EDITED LINE 1 --- ###
    # We pass 'frames=max_len' and 'cache_frame_data=False'
    ani = FuncAnimation(fig, anim_func, frames=max_len, interval=100, cache_frame_data=False)
    
    plt.tight_layout()
    
    ### --- EDITED LINE 2 --- ###
    # We return the 'ani' object so it doesn't get deleted
    return ani
