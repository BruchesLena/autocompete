import xml.etree.ElementTree as ET
import io

tree = ET.parse('D:\\Typing\\smsCorpus\\smsCorpus_en_xml_2015.03.09_all\\smsCorpus_en_2015.03.09_all.xml')
root = tree.getroot()

with io.open('D:\\Typing\\smsCorpus\\smsCorpus.txt', 'w') as f:
    for text in root.iter('text'):
        try:
            f.write(text.text.decode('utf-8')+u'\n')
        except UnicodeEncodeError:
            continue