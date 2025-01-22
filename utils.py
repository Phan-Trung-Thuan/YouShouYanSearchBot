import nltk
import math
import re

def get_shortest_string(strings):
    if not strings:  # Check if the list is empty
        return None
    return min(strings, key=len)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

def remove_chinese(text):
    pattern = re.compile('[\u4e00-\u9fff]+')
    return re.sub(pattern, '', text)
    
def rule_based_lemmatizer(word):
    # Plural to singular (basic rule for nouns)
    if word.endswith('ies') and len(word) > 3:
        return word[:-3] + 'y'
    elif word.endswith('es') and len(word) > 2:
        return word[:-2]
    elif word.endswith('s') and len(word) > 1:
        return word[:-1]

    # Past tense to base form (basic rule for verbs)
    if word.endswith('ed') and len(word) > 2:
        return word[:-2]
    elif word.endswith('ing') and len(word) > 4:
        return word[:-3]

    # Comparative and superlative forms (adjectives/adverbs)
    if word.endswith('er') and len(word) > 2:
        return word[:-2]
    elif word.endswith('est') and len(word) > 3:
        return word[:-3]

    # Default: return the word itself
    return word

def get_text_matching_score(input_text: str, document:str):
    input_text = input_text.lower()
    document = document.lower()
    document = document.replace(document[:5], '') if document[:4].isnumeric() else document # Remove chapter_id
    document = document.replace('\n', ' ')

    input_text = remove_punctuation(input_text)
    document = remove_punctuation(document)

    input_text = ' '.join([rule_based_lemmatizer(word) for word in input_text.split(' ')])
    document = ' '.join([rule_based_lemmatizer(word) for word in document.split(' ')])

    keywords = list(re.findall(r'\w+', input_text))

    counter = 0
    for i in range(len(keywords)):
        for j in range(i, len(keywords)):
            counter += document.count(' '.join(keywords[i:j+1])) ** 0.5 * (j - i + 1) ** 3
    
    confidence_score = counter / len(input_text.split())

    return confidence_score

def get_semantic_matching_score(input_text: str, document: str):
    pass

# Example usage
if __name__ == "__main__":
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

