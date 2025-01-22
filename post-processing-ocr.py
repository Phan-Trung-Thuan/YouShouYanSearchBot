import os
import json as js
import re
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from transformers import pipeline

def normalize_punctuation(text):
    return (
        text.replace('，', ',')
            .replace('！', '!')
            .replace('。', '.')
            .replace('？', '?')
            .replace('；', ';')
            .replace('：', ':')
            .replace('（', '(')
            .replace('）', ')')
            .replace('【', '[')
            .replace('】', ']')
            .replace('《', '<')
            .replace('》', '>')
            .replace('“', '"')
            .replace('”', '"')
            .replace('‘', "'")
            .replace('’', "'")
    )

def string_similarity(s1, s2):
    dist = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:  # Avoid division by zero if both strings are empty
        return 1.0
    return 1 - (dist / max_len)

def contains_mostly_numbers_and_punctuation(s, p=0.8):
    # Define the characters to consider as numbers, spaces, or punctuation
    pattern = r'[\d\s\-.,:—!?@&³˘^<>()*+#%$¥~`]'  # Adjust punctuation as needed
    total_chars = len(s)
    
    if total_chars == 0:
        return False  # Avoid division by zero, handle empty string as not valid

    # Count how many characters match the pattern
    valid_chars = len(re.findall(pattern, s))
    
    # Calculate the proportion of valid characters
    proportion = valid_chars / total_chars

    # Check if at least p proportion of the string consists of valid characters
    return proportion >= p

def remove_chinese_characters(text):
    # Regex to match Chinese characters (ranges for CJK Unified Ideographs)
    chinese_pattern = r'[\u4e00-\u9fff]'
    cleaned_text = re.sub(chinese_pattern, '', text)
    return cleaned_text.strip()

def remove_emojis(text):
    # Emoji pattern covering most emoji ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
        "\U0001F680-\U0001F6FF"  # Transport and Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)

def remove_links(text):
    # Regex to match URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    cleaned_text = re.sub(url_pattern, '', text)
    return cleaned_text.strip()

def find_overlap(s1, s2):
    max_overlap_len = min(len(s1), len(s2))  # The maximum possible overlap
    for i in range(max_overlap_len, 0, -1):  # Check for overlap from largest to smallest
        if s1[-i:] == s2[:i]:
            return i  # Length of the overlap
    return 0  # No overlap

def find_common_prefix(s1, s2):
    prefix = []
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            prefix.append(c1)
        else:
            break
    return ''.join(prefix)

def merge_strings(strings):
    merged_text = strings[0]
    for next_text in strings[1:]:
        overlap_length = find_overlap(merged_text, next_text)
        # Append the non-overlapping part of next_text to merged_text
        merged_text += ' ' + next_text[overlap_length:]
    return merged_text.strip()

def merge_short_string(paragraphs):
    for i, p in enumerate(paragraphs):
        if i == 0 or i == len(paragraphs)-1 or len(p) > 40 or len(p) == 0:
            continue
        elif len(p) < 20:
            paragraphs[i-1] += '. ' + paragraphs[i]
            paragraphs[i] = ''
                
    paragraphs = [p for p in paragraphs if p != '']
    paragraphs = [p[4:] if p.startswith('. . ') else p for p in paragraphs]
    paragraphs = [p[2:] if p.startswith('. ') else p for p in paragraphs]
    return paragraphs

def split_by_paragraph(merged_text):
    # Split into paragraphs and normalize whitespace
    paragraphs = [p.strip() for p in merged_text.split("\n\n") if p.strip()]
    paragraphs = [p.replace('\n', ' ').replace('   ', ' ').replace('  ', ' ').replace('Pi Xiux', 'Pi Xiu') for p in paragraphs]
    paragraphs = [remove_links(p) for p in paragraphs]
    paragraphs = [p[:p.find('Copyright')] if p.find('Copyright') != -1 else p for p in paragraphs]
    paragraphs = [remove_chinese_characters(p) for p in paragraphs]
    paragraphs = merge_short_string(paragraphs)
    paragraphs = merge_short_string(paragraphs)
    paragraphs = merge_short_string(paragraphs)
    paragraphs = [p for p in paragraphs if p != '' and p != ' ' and p != '.']

    for i in range(2, len(paragraphs)):
        common_prefix = find_common_prefix(paragraphs[i], paragraphs[i-2])
        if string_similarity(paragraphs[i], paragraphs[i-2]) >= 0.8 or len(common_prefix) > 10 or len(common_prefix) / max(len(paragraphs[i]), len(paragraphs[i-2])) > 0.8:
            if len(paragraphs[i]) >= len(paragraphs[i-2]):
                paragraphs[i-2] = ''
            else:
                paragraphs[i] = ''

    for i in range(1, len(paragraphs)):
        common_prefix = find_common_prefix(paragraphs[i], paragraphs[i-1])
        if string_similarity(paragraphs[i], paragraphs[i-1]) >= 0.8 or len(common_prefix) > 10:
            if len(paragraphs[i]) > len(paragraphs[i-1]):
                paragraphs[i-1] = ''
            else:
                paragraphs[i] = ''
    
    paragraphs = [p.replace('Translation and stuff: Wehafse. ', '')
                  .replace('translation and everything: Wehafse. ', '') 
                  .replace('There is no visible English text in the image', '')
                  .replace('There is no English text visible in the image', '')
                  .replace('The text visible in the image is:', '')
                  .replace(' . ', '. ').replace('. . ', '. ').replace('.  .', '').replace('..  .', '..')
                  for p in paragraphs]
    paragraphs = [p[4:] if p.startswith('. . ') else p for p in paragraphs]
    paragraphs = [p[2:] if p.startswith('. ') else p for p in paragraphs]
    paragraphs = [p for p in paragraphs if p != '' and p != ' ' and p != '.']
    return paragraphs
    
def process_ocr_raw(listStr):
    merged_string = merge_strings(listStr)
    splited_string = split_by_paragraph(merged_string)
    return splited_string

classifier = pipeline("zero-shot-classification", 
                      model="facebook/bart-large-mnli", 
                      device='cuda')
labels = ["meaningful", "meaningless"]

ocr_raw_path = '/kaggle/input/ysy-en-ocr-raw/YSY-comic-en-ocr'
list_file_path = os.listdir(ocr_raw_path)
list_file_path.sort()
results = []
for file in tqdm(list_file_path):
    chapter_id = file[13:17]
    file_path = os.path.join(ocr_raw_path, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = js.load(f)

    # Process the data
    processed_data = process_ocr_raw(data)
    for paragraph in processed_data:
        paragraph = normalize_punctuation(paragraph)
        paragraph = remove_emojis(paragraph)
        if contains_mostly_numbers_and_punctuation(paragraph, p=0.7):
            print('Meaningless:', paragraph)
            continue
        
        results.append(f'{chapter_id}: {paragraph}\n')
    results.append("="*40 + "\n")

with open('ysy_ocr_1_975.txt', 'w', encoding='utf-8') as f:
    f.writelines(results)