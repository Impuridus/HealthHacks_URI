"""Run: python lsl_plot.py (requires pylsl, matplotlib, numpy)."""
from collections import deque
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from pylsl import StreamInlet, resolve_streams, cf_string

WINDOW_SECONDS = 10


def main():
    print("Searching for numeric LSL streams...")
    streams = [s for s in resolve_streams(wait_time=3)
               if s.channel_format() != cf_string and s.channel_count() > 0]
    if not streams:
        print("No numeric streams found. Start your wearable's LSL publisher and retry.")
        return
    for i, stream in enumerate(streams):
        print(f"[{i}] {stream.name()} | {stream.type()} | "
              f"{stream.channel_count()} channels | {stream.nominal_srate()} Hz | "
              f"source={stream.source_id()}")
    try:
        choice = int(input("Stream number [0]: ").strip() or "0")
        if not 0 <= choice < len(streams):
            raise ValueError
    except ValueError:
        print("Invalid stream number. Run again and choose a listed number.")
        return

    selected = streams[choice]
    inlet = StreamInlet(selected, max_buflen=20)
    try:
        info = inlet.info(timeout=5)
        channel = info.desc().child("channels").child("channel")
        labels = []
        for i in range(info.channel_count()):
            label = channel.child_value("label") or f"Channel {i + 1}"
            unit = channel.child_value("unit") or "stream units"
            labels.append(f"{label}\n({unit})")
            channel = channel.next_sibling()

        fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False,
                                 figsize=(10, max(4, 1.7 * len(labels))))
        axes = axes[:, 0]
        lines = []
        for ax, label in zip(axes, labels):
            line, = ax.plot([], [], linewidth=0.8)
            lines.append(line)
            ax.set_ylabel(label)
            ax.set_xlim(-WINDOW_SECONDS, 0)
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Seconds relative to newest received sample")
        title = fig.suptitle(f"{info.name()} — waiting for data")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        # Bounded display history; this viewer is not a recording tool.
        history = deque(maxlen=200000)
        last_received = time.monotonic()

        def update(_):
            nonlocal last_received
            # Nonblocking reads keep the Matplotlib window responsive.
            samples, stamps = inlet.pull_chunk(timeout=0.0, max_samples=8192)
            if stamps:
                last_received = time.monotonic()
                history.extend(zip(stamps, samples))
                newest = history[-1][0]
                while history and history[0][0] < newest - WINDOW_SECONDS:
                    history.popleft()
                times = np.array([row[0] for row in history]) - newest
                values = np.asarray([row[1] for row in history], dtype=float)
                for i, (ax, line) in enumerate(zip(axes, lines)):
                    line.set_data(times, values[:, i])
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
                title.set_text(f"{info.name()} — live (unfiltered values)")
            elif time.monotonic() - last_received > 2:
                title.set_text(f"{info.name()} — no new samples for "
                               f"{time.monotonic() - last_received:.0f}s")
            return lines

        animation = FuncAnimation(fig, update, interval=50, blit=False,
                                  cache_frame_data=False)
        plt.show()
    finally:
        inlet.close_stream()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
