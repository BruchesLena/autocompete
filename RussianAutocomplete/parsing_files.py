import xml.etree.ElementTree as ET
import os
import io



dir = "/Users/bruches/Documents/Typing/fb2-03-110688-121287-RUSSIAN/"
writing_dir = "/Users/bruches/Documents/Typing/texts/"

def extract_clear_text(file):
    prefix = '{http://www.gribuser.ru/xml/fictionbook/2.0}'
    tree = ET.parse(dir+file)
    root = tree.getroot()
    writing_file = io.open(writing_dir+file, 'w', encoding='utf-8')
    for paragraph in root.iter(prefix+'p'):
        text = paragraph.text
        try:
            writing_file.write(text+u'\n')
        except TypeError:
            continue

i = 0
files = os.listdir(dir)
for file in files:
    extract_clear_text(file)
    print(str(i))
    i+=1

