
path_to_write = 'D:\\Typing\\data\\corpus.txt'
with open(path_to_write, 'w') as w_file:
    with open('D:\\Typing\\data\\friends-final.txt', 'r') as r_file:
        text = r_file.read().split('\n')
        for line in text:
            parts = line.split('\t')
            try:
                t = parts[5].replace('\\', '')
                w_file.write(parts[5]+'\n')
            except IndexError:
                continue
