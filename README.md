# Event-Based vs. Frame-Based Embedded Vision  
Micropython Pipelines for Data Collection, Inference, and Logging

This repository contains Micropython and Python code used to collect data, run inference, and log results for two embedded vision systems:

- **Frame-based system:** OpenMV Cam RT1062 with the OV5640 CMOS sensor  
- **Event-based system:** GENX320 event-driven vision sensor  

The project evaluates both systems under varying lighting and motion conditions and compares their performance in industrial inspection scenarios. All code and datasets used in the experiments are included for reproducibility.

---

## 📁 Repository Structure


Each system has separate folders for data collection, inference, and helper utilities. Logged data from both pipelines is stored in the `datasets/` directory.

---

## 📸 Systems Overview

### Frame-Based Vision (OpenMV RT1062 + OV5640)
This pipeline captures RGB frames using the OV5640 CMOS sensor.  
Micropython scripts handle:

- Image acquisition  
- Preprocessing  
- Threshold-based detection  
- Logging of inference results  

This system serves as the baseline for conventional embedded vision.

### Event-Based Vision (GENX320)
The GENX320 sensor outputs asynchronous events.  
Micropython and Python scripts perform:

- Event aggregation  
- Time-window histogram generation  
- Lightweight inference  
- Logging of event-based detections  

This pipeline demonstrates the advantages of event-driven sensing under fast motion and challenging illumination.

---

## 🧪 Data Collection

Both systems record:

- Lighting condition  
- Object speed  
- Raw sensor data (frames or event histograms)  
- Inference outputs  
- Timestamps  

Logs are stored in `.csv` or `.txt` format inside the `datasets/` directory.

---

## ⚙️ Inference

Inference scripts run directly on the microcontroller and include:

- Threshold-based detection  
- Event histogram analysis  
- Logging of TP, FP, TN, FN  
- Accuracy computation  

These scripts were used to generate the results presented in the associated research work.

---

## 🟦 Event → Frame Conversion (Python)

A Python script is included to convert raw GENX320 events into frame-like histograms using a fixed time window.  
This allows event data to be processed using conventional image-based pipelines.

### **event_to_frame.py**

```python
import numpy as np

def events_to_frame(events, width, height, window_us=5000):
    """
    Convert a list of events into a frame histogram using a time window.

    events: list of (x, y, t, polarity)
    width, height: sensor resolution
    window_us: time window in microseconds
    """

    if len(events) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    # Sort events by timestamp
    events = sorted(events, key=lambda e: e[2])

    # Determine window start
    t_start = events[0][2]
    t_end = t_start + window_us

    frame = np.zeros((height, width), dtype=np.uint8)

    for x, y, t, pol in events:
        if t_start <= t <= t_end:
            frame[y, x] += 1

    # Normalize to 0–255
    frame = np.clip(frame, 0, 255)

    return frame
