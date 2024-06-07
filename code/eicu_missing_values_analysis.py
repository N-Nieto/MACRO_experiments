
# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

root_dir = "/home/nnieto/Nico/MODS_project/data/eicu-collaborative-research-database-2.0/"              # noqa


usecols = ["patientunitstayid", "observationoffset",
           "heartrate", "systemicsystolic", "systemicdiastolic"]
heart_data_org = pd.read_csv(root_dir + "vitalPeriodic_selected_MACRO.csv.gz",                  # noqa
                             usecols=usecols)

heart_data_org = heart_data_org[heart_data_org['observationoffset'] > -1440]

plausible_range = [(20, 200, "heartrate"),
                   (20, 200, "systemicsystolic"),
                   (20, 200, "systemicdiastolic")]

# Iterate over each range in plausible_range
for min_val, max_val, column in plausible_range:
    # Replace values outside the range with NaN for the specified column
    heart_data_org[column] = heart_data_org[column].where((heart_data_org[column] >= min_val) & (heart_data_org[column] <= max_val), other=np.nan)      # noqa

# %%
dystolic = []
systolyc = []
HR = []
patients = []
minutes = 144
time_range = np.linspace(0, minutes, 100, dtype=int)

for time in time_range:
    # Remove the data after one Hour
    heart_data = heart_data_org[heart_data_org['observationoffset'] < time]                  # noqa
    # Remove data older than one day
    heart_data = heart_data.groupby("patientunitstayid").median().reset_index()
    dystolic.append(heart_data.isna().sum()["systemicdiastolic"])
    HR.append(heart_data.isna().sum()["heartrate"])
    patients.append(heart_data["patientunitstayid"].nunique())


# %%
plt.plot(time_range, 100*(np.array(patients)-np.array(HR))/1633)
plt.plot(time_range, 100*(np.array(patients)-np.array(dystolic))/1633)
plt.ylim([0, 100])
plt.xlim([0, minutes])
plt.ylabel("Percentage Patient with information")
plt.xlabel("Minutes after admission")
plt.legend(["Heart Rate", "Blood preasure"])
plt.grid()

# %%
plt.plot(time_range, np.array(patients))
plt.plot(time_range, 1633*np.ones_like(time_range))
plt.plot(time_range, (np.array(patients)-np.array(dystolic)))
plt.plot(time_range, (np.array(patients)-np.array(HR)))
plt.ylabel("Patients with information")
plt.xlabel("Minutes after admission")
plt.legend(["Heart Rate", "Blood preasure"])

plt.grid()
plt.ylim([0, 1700])
plt.xlim([0, minutes])
# %%
