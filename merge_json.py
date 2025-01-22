import json
from tqdm import tqdm
with open('database\You.Shou.Yan.Patch400.embed_vit_h14.json', 'r') as f:
    data1 = json.load(f)
    print(data1[0]['file'])
with open('database\output_vit_h14.json', 'r') as f:
    data2 = json.load(f)
    print(data2[0]['file'])
data1.extend(data2)
print(len(data1))

class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        answers = '[\n'
        for item in tqdm(obj):
            file = item['file']
            embedding = item['embedding']
            tmp = '''    {
    "file": ''' + f'''"{file}",
    "embedding": {embedding}
''' + '''},\n'''
            answers += tmp
        
        return answers[:-2] + '\n]'

# Example data
# data1 = [
#     {"file": "file1.txt", "embedding": [0.1, 0.2, 0.3, 0.4]},
#     {"file": "file2.txt", "embedding": [0.5, 0.6, 0.7, 0.8]}
# ]

encoder = CustomJSONEncoder()
# print(encoder.encode(data1))

# Write JSON with the custom encoder
with open('output.json', 'w', encoding='utf-8') as f:
    f.write(encoder.encode(data1))