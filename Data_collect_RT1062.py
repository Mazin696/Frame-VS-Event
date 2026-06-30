import sensor, image, time, uos

# ===================== CONFIG =====================
BASE         = "/sdcard"
SAVE_ROOT    = BASE + "/frame_dataset"
LABEL        = "Wrong"
CONDITION    = "L0_S0"
FIX_EXP_US   = 21450
FIX_GAIN_DB  = 23
MAX_SHOTS    = 500
DELAY_MS     = 200
# ===================================================

# ===================== CREATE DIRECTORIES =====================
def makedirs(path):
    parts = path.strip("/").split("/")
    cur = ""
    for p in parts:
        if not p:
            continue
        cur += "/" + p
        try:
            uos.mkdir(cur)
        except OSError:
            pass

SAVE_DIR = SAVE_ROOT + "/" + LABEL + "/" + CONDITION
makedirs(SAVE_DIR)
print("Saving images to:", SAVE_DIR)
# ===============================================================

# ===================== SENSOR INITIALIZATION =====================
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.set_framesize(sensor.QVGA)

sensor.skip_frames(time=2000)   # let auto run briefly
sensor.snapshot()               # register sync

sensor.set_auto_exposure(False, exposure_us=FIX_EXP_US)
sensor.set_auto_gain(False, gain_db=FIX_GAIN_DB)
sensor.set_auto_whitebal(False)

print("Exposure locked:", sensor.get_exposure_us())
print("Gain locked:", sensor.get_gain_db())
# ==============================================================

# ===================== CAPTURE LOOP =====================
img_id = 0
print("Starting capture…")

while img_id < MAX_SHOTS:
    img = sensor.snapshot()
    filepath = "%s/%s_%06d.jpg" % (SAVE_DIR, LABEL, img_id)
    img.save(filepath, quality=90)
    print("Saved:", filepath)
    img_id += 1
    time.sleep_ms(DELAY_MS)

print("DONE — captured", MAX_SHOTS, "images.")
# =======================================================
