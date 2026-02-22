import json

json_path = "/data7/tyx/DiffSVS/data/metadata.json"
json_file = json.load(open(json_path, 'r', encoding='utf-8'))

cnt = 0
for key, value in json_file.items():
    print(f"ID: {key}")
    print(f"Data: {value}")
    print("="*30)
    cnt += 1
    if cnt == 10:
        break
