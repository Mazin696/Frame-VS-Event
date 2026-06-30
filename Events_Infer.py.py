# ei_genx320_events_infer.py

import csi
import image
import time
import ml
import gc
import machine
from ulab import numpy as np

MODEL_PATH = "/sdcard/trained.tflite"
LABELS_PATH = "/sdcard/labels.txt"
LOG_PATH = "/sdcard/genx320_log.csv"

CONFIDENCE_THRESHOLD = 0.60

TARGET_OBJECT_A = "Object_A"
TARGET_OBJECT_B = "Object_B"

# ---- Settings --------------------------------------------------------------
WINDOW_MS = 200
EVT_RES = 8192
HIST_CONTRAST = 64
HIST_BRIGHTNESS = 0
IMG_W = 320
IMG_H = 320
# ---------------------------------------------------------------------------

def kb(x):
    return int(x / 1024)

def abs_us(e):
    return (int(e[1]) * 1000000) + (int(e[2]) * 1000) + int(e[3])

def top2(scores):
    inds = list(range(len(scores)))
    inds.sort(key=lambda i: scores[i], reverse=True)
    a = (inds[0], scores[inds[0]])
    b = (inds[1], scores[inds[1]]) if len(inds) > 1 else (inds[0], 0.0)
    return a, b

# ---------- Load model ----------
gc.collect()
print("Free RAM before load:", kb(gc.mem_free()), "KB")
net = ml.Model(MODEL_PATH, load_to_fb=False)
gc.collect()
print("Free RAM after load:", kb(gc.mem_free()), "KB")

# ---------- Load labels ----------
try:
    labels = [line.rstrip() for line in open(LABELS_PATH)]
except Exception:
    print("WARN: labels.txt not found")
    labels = ["Object_A", "Object_B"]

# ---------- Image ----------
img = image.Image(IMG_W, IMG_H, image.GRAYSCALE)

# ---------- Event buffer ----------
events = np.zeros((int(EVT_RES), 6), dtype=np.uint16)

# ---------- Sensor ----------
csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, events.shape[0])

# ---------- Output ----------
out_pin = machine.Pin("P0", machine.Pin.OUT)

# ---------- Excel CSV ----------
try:
    log = open(LOG_PATH, "w")
    log.write("sep=;\n")
    log.write("win_idx;ts_us;pred_label;confidence;is_target;n_events;window_us;event_rate;fps\n")
    log.flush()
except:
    log = None

# ---------- Timing ----------
clock = time.clock()
window_start_us = None
last_abs_us = None
window_idx = 0

while True:
    clock.tick()

    event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

    if event_count <= 0:
        time.sleep_ms(1)
        continue

    if window_start_us is None:
        window_start_us = abs_us(events[0])

    last_abs_us = abs_us(events[event_count - 1])

    if (last_abs_us - window_start_us) >= (WINDOW_MS * 1000):

        # Histogram
        img.draw_event_histogram(
            events[:event_count],
            clear=True,
            brightness=HIST_BRIGHTNESS,
            contrast=HIST_CONTRAST
        )

        # Inference
        scores = net.predict([img])[0].flatten().tolist()
        (i1, s1), _ = top2(scores)

        conf = float(s1)
        raw = labels[i1] if i1 < len(labels) else "Unknown"

        if conf < CONFIDENCE_THRESHOLD:
            pred_label = "Unknown"
            out_pin.value(0)
            color = 150
        else:
            if raw == TARGET_OBJECT_A:
                pred_label = TARGET_OBJECT_A
                out_pin.value(1)
                color = 255
            elif raw == TARGET_OBJECT_B:
                pred_label = TARGET_OBJECT_B
                out_pin.value(0)
                color = 255
            else:
                pred_label = "Unknown"
                out_pin.value(0)
                color = 150

        # Display
        img.draw_string(2, 2,
            "{} ({:.2f})".format(pred_label, conf),
            color=color, scale=2)
        img.flush()

        # ---------- LOG ONLY Object_A and Object_B ----------
        if pred_label in (TARGET_OBJECT_A, TARGET_OBJECT_B):

            window_us = last_abs_us - window_start_us
            if window_us <= 0:
                window_us = 1

            event_rate = (event_count * 1000000.0) / window_us
            is_target = 1   # both are targets
            fps_val = clock.fps()

            line = "{};{};{};{:.3f};{};{};{};{:.2f};{:.2f}\n".format(
                window_idx,
                last_abs_us,
                pred_label,
                conf,
                is_target,
                event_count,
                window_us,
                event_rate,
                fps_val
            )

            if log:
                log.write(line)
                log.flush()

            print(line.strip())

        # Reset window
        window_idx += 1
        window_start_us = last_abs_us
