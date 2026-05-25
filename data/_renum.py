import json

with open('data.json', encoding='utf-8') as f:
    obj = json.loads(f.read())

for i, d in enumerate(obj, 1):
    d["id"] = i

with open('data_out.json', "w", encoding='utf-8') as f:
    f.write(json.dumps(obj, ensure_ascii=False, indent=2))