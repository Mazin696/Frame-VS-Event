# ei_rt1062_infer_memsafe_autopath.py
import gc, uos, sensor, time, ml, machine

# ---------- Auto-discovery helpers ----------
CANDIDATE_MODEL_NAMES = ("trained.tflite", "ei_model.tflite", "model.tflite")
CANDIDATE_LABEL_NAMES = ("labels.txt",)

SEARCH_ROOTS = ("/sdcard", "/sd", "/flash")

def list_dir_safe(path):
    try:
        return uos.listdir(path)
    except Exception:
        return []

def join(a, b):
    if a.endswith("/"):
        return a + b
    return a + "/" + b

def find_file(names, roots):
    # depth-1 search (root + immediate subdirs)
    for root in roots:
        # root
        for n in names:
            p = join(root, n)
            try:
                uos.stat(p)
                return p
            except Exception:
                pass

        # subdirs
        for d in list_dir_safe(root):
            p_dir = join(root, d)
            try:
                _ = uos.listdir(p_dir)  # fails if not directory
            except Exception:
                continue
            for n in names:
                p = join(p_dir, n)
                try:
                    uos.stat(p)
                    return p
                except Exception:
                    pass
    return None

# ---------- Locate model & labels ----------
MODEL_PATH  = find_file(CANDIDATE_MODEL_NAMES, SEARCH_ROOTS)
LABELS_PATH = find_file(CANDIDATE_LABEL_NAMES, SEARCH_ROOTS)

if MODEL_PATH is None:
    print("Could not find a TFLite model. Looked for:", CANDIDATE_MODEL_NAMES)
    print("Roots:", SEARCH_ROOTS)
    for r in SEARCH_ROOTS:
        print("Contents of", r, "->", list_dir_safe(r))
    raise Exception("Model file not found (copy your .tflite to /sdcard or /flash).")

if LABELS_PATH is None:
    print("WARN: labels.txt not found; defaulting to ['Type_A','Type_B']")
    labels = ["Type_A", "Type_B"]
else:
    labels = [line.rstrip() for line in open(LABELS_PATH)]

print("Using model:", MODEL_PATH)
print("Using labels:", LABELS_PATH if LABELS_PATH else "(default)")

# ---------- Config ----------
CONFIDENCE_THRESHOLD = 0.60
CLOSE_MARGIN = 0.04
SMOOTH_WINDOW = 5

TARGET_TYPE_A = "Type_A"
TARGET_TYPE_B = "Type_B"
PIN_ACTIVE_ON_TYPE_A = True

# Put log in SAME directory as the model
if "/" in MODEL_PATH:
    LOG_DIR = MODEL_PATH.rsplit("/", 1)[0]
else:
    LOG_DIR = "/sd"
LOG_PATH = LOG_DIR + "/rt1062_log.csv"

def kb(x):
    return int(x / 1024)

# ---------- Load model (memory-safe) ----------
gc.collect()
print("Free RAM before load:", kb(gc.mem_free()), "KB")

try:
    net = ml.Model(MODEL_PATH, load_to_fb=False)
except Exception as e:
    raise Exception('Failed to load model "{}": {}'.format(MODEL_PATH, e))

gc.collect()
print("Free RAM after load:", kb(gc.mem_free()), "KB")

# ---------- Camera ----------
sensor.reset()
try:
    sensor.set_framebuffers(1)  # save RAM
except Exception:
    pass

# Use RGB565 so colors can be shown on screen
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)  # 320x240
sensor.skip_frames(time=2000)
sensor.snapshot()

# Optional: lock exposure/gain
try:
    sensor.set_auto_exposure(False, exposure_us=21450)
    sensor.set_auto_gain(False, gain_db=23)
    sensor.set_auto_whitebal(False)
    print("Exposure locked:", sensor.get_exposure_us())
    print("Gain locked:", sensor.get_gain_db())
except Exception as e:
    print("Exposure/Gain lock not applied:", e)

# ---------- Inference utils ----------
def top2(scores):
    if not scores:
        return (0, 0.0), (0, 0.0)
    inds = list(range(len(scores)))
    inds.sort(key=lambda i: scores[i], reverse=True)
    a = (inds[0], scores[inds[0]])
    b = (inds[1], scores[inds[1]]) if len(inds) > 1 else (inds[0], 0.0)
    return a, b

def majority_vote(hist):
    if not hist:
        return "Unknown"
    c = {}
    for h in hist:
        c[h] = c.get(h, 0) + 1
    return sorted(c.items(), key=lambda kv: (kv[1], kv[0] != "Unknown"), reverse=True)[0][0]

def get_label_color(label):
    if label == TARGET_TYPE_A:
        return (0, 0, 255)       # Blue
    elif label == TARGET_TYPE_B:
        return (255, 0, 0)       # Red
    else:
        return (128, 128, 128)   # Gray

out_pin = machine.Pin("P0", machine.Pin.OUT)
vote_hist = []
clock = time.clock()

# ---------- CSV log setup ----------
try:
    log = open(LOG_PATH, "w")
    # Each item will be in a separate Excel column
    log.write("frame_idx,ts_us,pred_label,confidence,is_target,fps\n")
    log.flush()
    print("Logging to:", LOG_PATH)
except Exception as e:
    log = None
    print("ERROR: could not open log file:", e)

frame_idx = 0

print("Starting EI inference (auto-paths)…")
while True:
    clock.tick()
    ts_us = time.ticks_us()

    img = sensor.snapshot()
    scores = net.predict([img])[0].flatten().tolist()
    (i1, s1), (i2, s2) = top2(scores)

    raw = labels[i1] if i1 < len(labels) else "Unknown"
    pred_label = raw if raw in (TARGET_TYPE_A, TARGET_TYPE_B) else "Unknown"
    conf = float(s1)

    # Apply confidence and close-margin rule
    if (conf < CONFIDENCE_THRESHOLD) or (abs(s1 - s2) <= CLOSE_MARGIN):
        pred_label = "Unknown"

    vote_hist.append(pred_label)
    if len(vote_hist) > SMOOTH_WINDOW:
        vote_hist.pop(0)
    voted = majority_vote(vote_hist)

    # Digital output based on voted result
    out_pin.value(
        1 if (PIN_ACTIVE_ON_TYPE_A and voted == TARGET_TYPE_A) else
        (0 if PIN_ACTIVE_ON_TYPE_A else (1 if voted == TARGET_TYPE_B else 0))
    )

    # On-screen info with class-specific color
    text_color = get_label_color(pred_label)
    img.draw_string(
        0, 0,
        "pred:{} ({:.2f})".format(pred_label, conf),
        color=text_color,
        scale=2
    )

    # Debug print
    print("FPS:{:.1f} | pred:{} p={:.2f} | vote:{} | P0={}".format(
        clock.fps(), pred_label, conf, voted, out_pin.value()))

    # ---------- Logging ----------
    # Log Type_A, Type_B, and low-confidence Unknown
    if (pred_label == TARGET_TYPE_A or
        pred_label == TARGET_TYPE_B or
        (pred_label == "Unknown" and conf < 0.9)):

        # 1 for Type_A and Type_B, 0 for Unknown
        is_target = 1 if pred_label in (TARGET_TYPE_A, TARGET_TYPE_B) else 0
        fps_val = clock.fps()

        line = "{},{},{},{:.3f},{},{:.2f}\n".format(
            frame_idx,    # frame_idx
            ts_us,        # ts_us
            pred_label,   # pred_label
            conf,         # confidence
            is_target,    # is_target
            fps_val       # fps
        )

        if log is not None:
            try:
                log.write(line)
                log.flush()
            except Exception as e:
                print("LOG_WRITE_ERROR:", e)

        print(line.strip())
        frame_idx += 1
