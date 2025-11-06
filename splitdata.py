import json

# Read the original training.json file
with open('training.json', 'r') as file:
    data = json.load(file)

# Split the data
training1_data = data[:6140]
training2_data = data[6140:]

# Write training1.json
with open('training1.json', 'w') as file:
    json.dump(training1_data, file, indent=2)

# Write training2.json
with open('training2.json', 'w') as file:
    json.dump(training2_data, file, indent=2)

print(f"Split complete. training1.json has {len(training1_data)} items and training2.json has {len(training2_data)} items.")