from constant import *
import json as js

with open(YSY_COMIC_SUMMARY_PATH, 'r', encoding='utf8') as f:
    comic_summary = js.load(f)

tmp = []
for item in comic_summary:
    chapter_id = item['id']
    summary = item['summary']
    if 'chapter' in summary:
        i = summary.find(', ')
        summary = summary[i+2:]
    tmp.append((chapter_id, summary))
comic_summary = tmp
        
with open(YSY_COMIC_SUMMARY_PATH, 'w', encoding='utf8') as f:
    js.dump(comic_summary, f, indent=4)