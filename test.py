# pre = 0
# def get_document_by_id(file_path, target_id):
#     with open(file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split(",", 1)

#             if len(parts) != 2:
#                 continue

#             doc_id = parts[0].strip()
#             if doc_id == target_id:
#                 return parts[1].strip()
#     return None

# text_dir = "data\\quran\\quran_ayahs_tail764.txt" 
# print(get_document_by_id(text_dir, "38"))
# lis = []
# for i in range(2, 762):
#     try:
#         if not (a[str(i)]):
#         # print (a[str(i+1)])
#         # if not (a[str(i+1)]):
#             print("#####")
#     except:
#         doc_text = get_document_by_id(text_dir, str(i))
#         lis.append(str(i)+", "+doc_text)
#         pre=pre+1
# print(pre, " pre")
# for i in lis:
#     print(i)

from hazm import Normalizer, word_tokenize
normalizer = Normalizer()
from typing import Set, Dict, Tuple, List, Optional, Any


def extract_raw_phrases_ar_fa(text: str) -> Set[str]:

    phrases = set()

    text = normalizer.normalize(text)

    tokens = word_tokenize(text)

    # unigram
    for tok in tokens:
        if len(tok) >= 2:
            phrases.add(tok)

    # bigram
    for i in range(len(tokens)-1):
        phrase = f"{tokens[i]} {tokens[i+1]}"
        phrases.add(phrase)

    # trigram
    for i in range(len(tokens)-2):
        phrase = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        phrases.add(phrase)

    return phrases

import re
_ARABIC_SCRIPT = re.compile(r'[\u0600-\u06FF]')



a = extract_raw_phrases_ar_fa("فِي جَنَّـٰتٖ يَتَسَآءَلُونَ,[Who will be] in gardens, questioning each other")
b = extract_raw_phrases_ar_fa("فِي جَنَّـٰتٖ يَتَسَآءَلُونَ")
c = extract_raw_phrases_ar_fa("[Who will be] in gardens, questioning each other")
# for i in c:
#     print(i)

def split_id_arabic_english(line: str):
    comma1 = line.index(',')
    ctx_id = line[:comma1].strip()
    rest = line[comma1 + 1:]

    ar_positions = [m.start() for m in _ARABIC_SCRIPT.finditer(rest)]
    if not ar_positions:
        return ctx_id, "", rest.strip()

    last_ar = max(ar_positions)
    ar_raw = rest[:last_ar + 1]
    en_raw = rest[last_ar + 1:]

    arabic_text = ar_raw.rstrip(',').strip().strip('"').strip()
    english_text = en_raw.strip().strip('"').strip()

    return ctx_id, arabic_text, english_text

line = '10,فِي جَنَّـٰتٖ يَتَسَآءَلُونَ,"[Who will be] in gardens, questioning each other"'
ctx_id, arabic_text, english_text = split_id_arabic_english(line)
print(ctx_id)
print(arabic_text)
print(english_text)
