from constant import *
import json as js
from tqdm import tqdm
from utils import remove_chinese, rule_based_lemmatizer
import re

with open(YSY_COMIC_KEYWORD_PATH, 'r', encoding="utf8") as f:
    comic_keywords = js.load(f)

def count_keywords_all(k, comic_keywords):
    k = k.lower()
    counter = 0
    for chapter_id, keywords_list in comic_keywords:
        if k in keywords_list:
            counter += 1
    return counter

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', '', text)

# tmp = []
# for chapter_id, keywords_list in comic_keywords:
#     keywords_set = set()
#     for keyword in keywords_list:
#         keywords_set.add(rule_based_lemmatizer(remove_punctuation(remove_chinese(keyword.lower())).strip()))
#     # keywords_set
#     tmp.append((chapter_id, list(keywords_set)))
# comic_keywords = tmp
# with open(YSY_COMIC_KEYWORD_PATH, 'w', encoding="utf8") as f:
#     js.dump(comic_keywords, f, indent=4)

keywords_set = set()
for chapter_id, keywords_list in comic_keywords:
    for keyword in keywords_list:
        keywords_set.add(rule_based_lemmatizer(keyword))

keywords_set.remove('')

print(len(keywords_set))
# print([k for k in keywords_set if len(k) <= 3])

keywords_count = []
for keyword in tqdm(keywords_set):
    keywords_count.append((keyword, count_keywords_all(keyword, comic_keywords)))

keywords_count.sort(key=lambda x: (x[1], x[0]), reverse=True)
with open('database/You.Shou.Yan.keywords_count.json', 'w', encoding="utf8") as f:
    js.dump(keywords_count, f, indent=4)