import nltk
import math
import re
import json

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

def round_floats(obj, precision=6):
    """Recursively round all float values in a JSON object to the specified precision."""
    if isinstance(obj, float):
        return round(obj, precision)
    elif isinstance(obj, list):
        return [round_floats(item, precision) for item in obj]
    elif isinstance(obj, dict):
        return {key: round_floats(value, precision) for key, value in obj.items()}
    return obj  # Return as-is for other data types (int, str, bool, None)

def custom_json_dumps(obj, *args, keys_to_skip_indent=(), **kwargs):
    indent = kwargs.pop("indent", 0)
    separators = kwargs.get("separators", (", ", ": "))

    if isinstance(obj, dict):
        res = "{" + "\n" * int(indent != 0)
        for k, v in obj.items():
            if k in keys_to_skip_indent or indent == 0:
                encoded = json.dumps(v, *args, **kwargs)
            else:
                encoded = custom_json_dumps(v, *args, indent=indent, keys_to_skip_indent=keys_to_skip_indent, **kwargs)
            res += "\"{}\"".format(k) + separators[1] + encoded + separators[0] + "\n" * int(indent != 0)
        return res[:res.rindex(separators[0])].replace("\n", "\n" + " " * indent) + "\n" * int(indent != 0) + "}"

    elif isinstance(obj, list):
        if all(isinstance(item, dict) for item in obj) and indent != 0:
            # Proper indentation for lists of dictionaries
            res = "[\n"
            for item in obj:
                encoded = custom_json_dumps(item, *args, indent=indent, keys_to_skip_indent=keys_to_skip_indent, **kwargs)
                res += " " * indent + encoded + separators[0] + "\n"
            return res[:res.rindex(separators[0])] + "\n" + " " * (indent - len(separators[0])) + "]"
        else:
            return json.dumps(obj, *args, **kwargs)

    else:
        return json.dumps(obj, *args, **kwargs)

# Example usage
if __name__ == "__main__":
    query = "Diting's not here yet?"
    doc = """0093: ... Chapter 93: Soft Heart
0093: Diting's not here yet? Isn't he normally the one that's most excited to play?
0093: Unless something happened... oh he's here.
0093: knock knock Oh my, Diting, what happened!
0093: Did you get hurt? Why'd you shrink?!
0093: No. Today is the Ghost Festival, all the ghosts from the underworld are coming to earth to celebrate. They're usually very scared of me, but they feel more comfortable when I turn small.
0093: I see now... you're so considerate towards your employees... haha so soft.
0093: His tail is so cool and slick!. touc h. Don't touch me.. Alright... even
0093: Push down Alright... even small Diting is super dignified.... Push down. Birthday. Look out!
0093: Hellfire Express, an express delivery company from the Look out! What's this...?. Fall from the sky
0093: Diting is responsible for the exchange of goods between the underworld and earth. He often receives complaints about burns.. Woah, it's here.
0093: My birthday falls on the Ghost Festival, every year
0093: Look out! What's this...?. Fall from the sky
0093: Hellfire Express, an express delivery company from the underworld.
0093: Diting is responsible for the exchange of goods between the underworld and earth. He often receives complaints about burns.. Woah, it's here.
0093: My birthday falls on the Ghost Festival, every year Dizang Buddha* sends me a birthday gift.
0093: *tl/n: aka Ksitigarbha bodhisattva. Known as an overlord of hell in Chinese Buddhism. Happy birthday!. Glee
0093: Small clothes this year?
0093: If you want Why didn't you tell us earlier, we didn't prepare anything...
0093: Well, in that case... Honestly... just let me die.... Hu.
0093: Ran out of melon seeds
0093: Everyone was forced to play Mahjong with Diting until the end of time."""
    doc = """0697: CHAPTER 697: THE DARKEST PLACE
0697: CLACK, CLACK CLACK. (DON'T BE SO PROUD OF YOURSELF FOR COPYING MY DESIGNS.)
0697: GURGLE, GURGLE, GURGLE. I SWEAR ON MY NAME, I WILL DEFINITELY CREATE A HIGH-END BRAND THAT NO ONE CAN COPY!
0697: AFTER THE KUMAOMAO INCIDENT, XIAOLU DECIDED TO OVERHAUL HER WHOLE BUSINESS.
0697: CLACK! CLACK, CLACK, CLACK, CLACK, CLACK, (DONE! THIS MUST BE WHAT IT FEELS LIKE TO BE A GOD TRAPPED IN PURGATORY. THE NEXT STEP IS THE MATERIAL...)
0697: SOAKED IN ANGER AND HATRED, A NEW BRAND WAS BORN.
0697: 19 interesting facts about hell!  The Darkest Place in Hell
0697: SQUEAK, SQUEAK, SQUEAK, ( THE BOOK SAYS THAT IT'S IN THE DARKEST PLACE IN HELL, SEALED WITH THE CORPSE OF THE ETERNAL SINFUL BEAST.)
0697: THE DARKEST PLACE IN HELL IS THE NINETEENTH CIRCLE OF HELL, WHICH IS A RESTRICTED AREA THAT CANNOT BE ENTERED AT WILL.
0697: CLACK CLACK, CLACK! ( PREPARE TO SET OUT, MY SKELETON ARMY! WE WILL NOT STOP UNTIL WE BRING BACK THE BEST SKIN!)
0697: DITING'S PASSPORT  -IDENTITY VERIFICATION SUCCESSFUL PLEASE PASS.. DITING'S PASSPORT. PONG DITING
0697: WELCOME, MASTER DITING, WELCOME, MASTER DITING TO THE DARKEST PLACE IN HELL.. PANDORA'S BOX
0697: FOLLOWING THE GUIDANCE OF ANCIENT BOOKS,. CLACK!
0697: THE SKELETON ARMY ARRIVED AT THE PLACE WHERE THE SKELETONS OF ANCIENT SINFUL BEASTS WERE SAID TO BE SEALED. CLACK CLACK? CLACK! CHOW DO WE GET IN? OPEN SESAME! T/N: GATE IS COVERED IN RIPPED "FULLU": TALISMAN STICKERS MADE TO WARD OFF EVIL.
0697: "CHOW DO WE GET IN? OPEN SESAME!"
0697: CLACK, CLACK CLACK, CLACK? CIT DIDN'T WORK, AND THE SEALS CAN'T BE REMOVED. WHAT SHOULD I DO?
0697: SQUEAK, SQUEAK, SQUEAK. PLEASE COME THIS WAY, MS. LL. WE HAVE DUG A RAT HOLE! WITH THE SPIRIT OF MADNESS AND DEATH.... SQUEAK! !
0697: ...THE SKELETON ARMY SUCCESSFULLY SNUCK INTO THE FORBIDDEN AREA.
0697: T/N: SO MUCH COOL STUFF TWT... ALL ON THE FEIRENZAI FLAGSHIP STORE... ALL LONG GONE AND OUT OF STOCK....
0697: Extruded text visible in the image:
0697: 1. "FABU" 2. "" 3. "" 4. "  (3)" 5. " (5)" Winter Warmth Blanket (each 2 sizes). Disposable Mask (5)
0697: Tianlu Rug & Tianlu Irregular Rug (2)"""
    print(get_text_matching_score(query, doc))

