# collect_genx320_events_csv_stream_updated.py
# Save raw GENX320 events to CSV as x,y,t_us,p
# and keep the IDE preview streaming using event histogram rendering.

import csi
import image
import time
import uos
from ulab import numpy as np

# ========= TUNABLES =========================================================
LABEL = "Type_B"
N_SAMPLES = 200
DURATION_MS = 200            # event window per sample
EVT_RES = 8192             # must be power of 2 between 1024 and 65536
BASE = "/sdcard"

# Preview / histogram
SHOW_HISTOGRAM = True
HIST_CONTRAST = 16
HIST_BRIGHTNESS = 128
HIST_CLEAR_EACH_BATCH = True
FLUSH_EVERY_MS = 40         # throttle preview pushes (~25 FPS)
# ===========================================================================

def makedirs(path):
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            uos.mkdir(cur)
        except OSError:
            pass

def now_us(e):
    # Compose absolute timestamp (us) from one event row
    return (int(e[1]) * 1000000) + (int(e[2]) * 1000) + int(e[3])

SAVE_ROOT = BASE + "/Events"
CLASS_DIR = SAVE_ROOT + "/" + LABEL
makedirs(CLASS_DIR)

# Surface to draw histogram image on
img = image.Image(320, 320, image.GRAYSCALE)
img.clear()

# Stores camera events
# Columns:
# [0] event type / polarity
# [1] seconds
# [2] milliseconds
# [3] microseconds
# [4] x
# [5] y
events = np.zeros((int(EVT_RES), 6), dtype=np.uint16)

# Initialize sensor
csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])

clock = time.clock()

def collect_one_csv(path, dur_ms):
    f = open(path, "w")
    f.write("x,y,t_us,p\n")

    start_abs_us = None
    end_abs_us = None
    deadline_us = None
    last_flush_ms = time.ticks_ms()

    while True:
        clock.tick()

        # Read up to EVT_RES events
        event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

        if event_count < 0:
            print("READ_ERROR:", event_count)
            continue

        # Keep preview alive even if quiet
        if event_count == 0:
            if SHOW_HISTOGRAM:
                now_ms = time.ticks_ms()
                if time.ticks_diff(now_ms, last_flush_ms) >= int(FLUSH_EVERY_MS):
                    img.clear()
                    img.draw_string(2, 2, "REC...", color=255, scale=1)
                    img.flush()
                    last_flush_ms = now_ms
            time.sleep_ms(1)
            continue

        # Write CSV rows
        for i in range(event_count):
            e = events[i]
            abs_t = now_us(e)

            if start_abs_us is None:
                start_abs_us = abs_t
                deadline_us = start_abs_us + (int(dur_ms) * 1000)

            rel_us = abs_t - start_abs_us
            x = int(e[4])
            y = int(e[5])
            p = int(e[0])

            f.write("{},{},{},{}\n".format(x, y, rel_us, p))
            end_abs_us = abs_t

        # Render histogram preview
        if SHOW_HISTOGRAM:
            img.draw_event_histogram(
                events[:event_count],
                clear=HIST_CLEAR_EACH_BATCH,
                brightness=int(HIST_BRIGHTNESS),
                contrast=int(HIST_CONTRAST)
            )

            now_ms = time.ticks_ms()
            if time.ticks_diff(now_ms, last_flush_ms) >= int(FLUSH_EVERY_MS):
                img.draw_string(2, 2, "REC... {}".format(event_count), color=255, scale=1)
                img.flush()
                last_flush_ms = now_ms

        # Stop when last event passes deadline
        if (end_abs_us is not None) and (deadline_us is not None) and (end_abs_us >= deadline_us):
            break

    f.close()

# Main loop
for i in range(int(N_SAMPLES)):
    fn = "%s/%s_%05d.csv" % (CLASS_DIR, LABEL, i)
    t0 = time.ticks_ms()
    collect_one_csv(fn, int(DURATION_MS))
    dt = time.ticks_diff(time.ticks_ms(), t0)
    print("saved:", fn, "dur_ms:", DURATION_MS, "took_ms:", dt, "fps:", clock.fps())

# Final frames so preview does not freeze on exit
for _ in range(5):
    if SHOW_HISTOGRAM:
        img.clear()
        img.draw_string(2, 2, "DONE", color=255, scale=2)
        img.flush()
        time.sleep_ms(50)

print("Done:", CLASS_DIR)
