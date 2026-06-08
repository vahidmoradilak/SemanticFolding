import json
import csv

# مسیر فایل JSONL دیتاست
input_file = "multilingual_quran.jsonl"
output_file = "quran_ayahs.csv"

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", encoding="utf-8", newline='') as f_out:

    writer = csv.writer(f_out)
    # نوشتن هدر (اختیاری)
    writer.writerow(["line_number", "arabic_ayah", "english_translation"])

    for i, line in enumerate(f_in, 1):
        data = json.loads(line)
        # print(data)
        # if (i==3):
            # break
        arabic = data.get("verse_ar", "").replace("\n", " ").strip()
        english = data.get("verse_en", "").replace("\n", " ").strip()
        writer.writerow([i, arabic, english])

print(f"Saved {i} rows to {output_file}")