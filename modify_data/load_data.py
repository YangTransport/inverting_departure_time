import json
import datetime
import re
import numpy as np

def load_data():
    """Loads the data, and returns two arrays, one with the actual
    data (in hours) and one with the hours passed from the first data point

    """
    with open("data.txt", "r", encoding="utf-8") as file:
        text = file.read()
    json_match = re.search(r'modelData\s*=\s*(\{.*\});?', text, re.DOTALL)
    json_text = json_match.group(1)
    data = json.loads(json_text)
    travel_times = np.array([entry["G"] for entry in data["ReportData"]["B"]]) / 60
    x = [entry["Y"] for entry in data["ReportData"]["B"]]
    x = np.array([datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S%z").timestamp() for t in x])
    x -= x[0]
    x /= 3600
    return x, travel_times
