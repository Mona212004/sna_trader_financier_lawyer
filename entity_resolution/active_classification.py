import pandas as pd
import dedupe
import os

# classification method 3: active learning classification using dedupe python library
training_file = r"D:\commo\code\entity_resolution\dedupe_training.json"
settings_file = r"D:\commo\code\entity_resolution\dedupe_settings"

fields = [dedupe.variables.String("name")]
deduper = dedupe.Dedupe(fields)

# format df123 into dict struct {row_id: {field: value}}
df123 = pd.read_csv(r"D:\commo\code\entity_resolution\df123_train.csv")
data_d = {}
for idx, row in df123.iterrows():
    if isinstance(row["name"], str) and row["name"].strip():
        data_d[idx] = {"name": row["name"]}

if os.path.exists(training_file):
    print("reading labeled examples from ", training_file)
    with open(training_file, "rb") as f:
        deduper.prepare_training(data_d, f)
else:
    deduper.prepare_training(data_d)
# ## Active learning
# Dedupe will find the next pair of records
# it is least certain about and ask you to label them as duplicates
# or not.
# use 'y', 'n' and 'u' keys to flag duplicates
# press 'f' when you are finished
print("starting active labeling...")
try:
    dedupe.console_label(deduper)
except KeyboardInterrupt:
    print("\nInterrupted! Attempting to save partial labels before exiting...")

# train and save regardless of whether labeling finished normally or was interrupted
try:
    deduper.train()
    with open("dedupe_training.json", "w") as tf:
        deduper.write_training(tf)
    with open("dedupe_settings", "wb") as sf:
        deduper.write_settings(sf)
    print("Training and settings saved successfully.")
except Exception as e:
    print(f"Could not save: {e}")
