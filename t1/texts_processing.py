import gzip
import os


# MARKS = [',', ':', '-', '(', ')', ';', '"', '*']
# write_file = open('D:\\Typing\\wiki.txt', 'w')
# path = 'D:\\Typing\\wiki\\'
# files = os.listdir(path)
# for file in files:
#     with gzip.open(path+file, 'r') as f:
#         content = f.read().split('\n')
#         for line in content:
#             sentences = line.split('.')
#             if len(sentences) < 2:
#                 continue
#             for sent in sentences:
#                 if len(sent.strip().split(' ')) < 3:
#                     continue
#                 clean = True
#                 for mark in MARKS:
#                     if mark in sent:
#                         clean = False
#                         break
#                 if clean:
#                     write_file.write(sent+'\n')


def augment_words():
    words = load_words_from_clusters('D:\\Typing\\data\\models_hier\\model_6\\clusters_30.txt')
    file_to_write = open('D:\\Typing\\data\\models_hier\\model_6\\from_wiki.txt', 'w')
    path = 'D:\\Syntax\\wiki\\'
    files = os.listdir(path)
    sentences_counter = 0
    for file in files:
        with gzip.open(path + file, 'r') as f:
            content = f.read().split('\n')
            for line in content:
                counter = 0
                for word in words.keys():
                    if word in line:
                        if words[word] > 300:
                            continue
                        i = line.count(word)
                        words[word] += i
                        counter += 1
                if counter == 0:
                    continue
                file_to_write.write(line+'\n')
                sentences_counter += 1
                print str(sentences_counter)
                if sentences_counter == 350000:
                    write_stat(words)
                    return


def write_stat(words):
    file_to_write = open('D:\\Typing\\data\\models_hier\\model_4\\stat_from_wiki.txt', 'w')
    with open('D:\\Typing\\data\\models_hier\\model_4\\clusters_23.txt', 'r') as f:
        clusters = f.read().split('\n')
        for cluster in clusters:
            parts = cluster.split(';')
            for word in parts[1].split(','):
                file_to_write.write(word+':'+str(words[word])+'\n')
            file_to_write.write('\n')
    file_to_write.close()



def load_words_from_clusters(path):
    words = {}
    with open(path, 'r') as f:
        clusters = f.read().split('\n')
        for cluster in clusters:
            parts = cluster.split(';')
            for part in parts[1].split(','):
                if part == '' or part == '-':
                    continue
                words[part] = 0
    return words

augment_words()



