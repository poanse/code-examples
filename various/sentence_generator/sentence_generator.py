import matplotlib.pyplot as plt
import numpy as np

class generator:
    def __init__(self):
        pass

    def learn(self, path = "./speeches.txt"):
        file = open(path).read()
        for x in ('\n', '?', '!', '...','..','…'):
            file = file.replace(x, '.')
        for x in ('«','»', ',', '-','–', ':', ';','’','"'):
            file = file.replace(x, '')
        file = file.replace('  ',' ')
        file = file.lower()
        file = file.split('.')
        file = list(filter(None, file))

        # create a list of sentences which are lists of words with start and end symbols
        self.start = '^start^'
        self.end = '^end^'
        for i in range(len(file)):
            file[i] = file[i].strip()
            file[i] = file[i].split(' ')
            file[i].insert(0, self.start)
            file[i].append(self.end)

        # create a set of all words and turn it to list
        words = set()
        for sentence in file:
            for word in sentence:
                words.add(word)
        words = list(words)

        # create a table
        n = len(words)
        table = np.zeros((n,n))

        # iterate over every word pair in every sentence and increment corresponding cell
        for sentence in file:
            for i in range(len(sentence) - 1):
                k = words.index(sentence[i]) # get index of sentence[i]
                l = words.index(sentence[i+1]) # get index of sentence[i+1]
                table[k,l] += 1 # increment corresponding cell

        # get probabilities
        for i in range(n):
            temp = np.sum(table[i,:])
            if temp != 0:
                table[i,:] = table[i,:]/temp
        self.words = words
        self.table = table

    def generate(self, seq = []):
        #start = '^start^'
        #end = '^end^'
        if seq == []:
            seq = [self.start]
        else:
            seq.insert(0,self.start)
        #if len(seq) == 1:
        #    seq.append(np.random.choice(words))
        while seq[-1] != self.end:
            word = seq[-1]
            #words2 = seq[-2]
            index = self.words.index(word)
            #index2 = words.index(word2)
            seq.append(np.random.choice(self.words, 1, p = list(self.table[index,:]))[0])
        print(seq[1],end = ' ')
        for x in seq[2:-1]:
            print(x, sep = ' ', end = ' ')
        print('\n',end='')