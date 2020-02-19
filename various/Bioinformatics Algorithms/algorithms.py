import math

def prod(iterable):
    prod = 1
    for i in iterable:
        prod = prod*i
    return prod

def Count(Text,Pattern):
    count=0
    startPos=[]
    i=0
    while i <= len(Text)-len(Pattern):
        if Text[i:i+len(Pattern)]==Pattern:
            count=count+1
            startPos.append(i)
        i=i+1
    return count, startPos

def FrequentWords(Text,k):
    patterns=[]
    frequency=[]
    i=0
    while i <= len(Text)-k:
        word = Text[i:i+k]
        j=0
        duplicate=0
        for j in patterns:
            if word==j:
                duplicate=1
        if duplicate == 0:
            patterns.append(word)
            frequency.append(Count(Text,word))
        i=i+1
    maxFreq=max(frequency)
    toprint=[]
    for t in list(range(len(frequency))):
        if frequency[t]==maxFreq:
            toprint.append(patterns[t])
    return toprint

def ReverseComplement(Strand):
    y=''
    for i in list(range(1,len(Strand)+1)):
        n=Strand[-i]
        if n=='A':
            s='T'
        elif n=='C':
            s='G'
        elif n=='G':
            s='C'
        elif n=='T':
            s='A'
        y=y+s
    return y

def HammingDistance(s,t):
    count=0
    if len(s)!= len(t):
        print('Error: len(s) != len(t)')
    for i in list(range(len(s))):
        if s[i]!= t[i]:
            count=count+1
    return count

def Fn(n,k):
    f=[1,1]
    i=3
    while i<=n:
        f.append(f[i-3]*k+f[i-2])
        i=i+1
    return f[n-1]

def Fn2(n,m):
    f=[0,1]
    i=2
    while i < n:
        if i-m-1 >= 0:
            f.append(f[i-1]+f[i-2]-f[i-m-1])
        elif i-m-1 == -1:
            f.append(f[i-1]+f[i-2]-1)
        else:
            f.append(f[i-1]+f[i-2])
        i=i+1
    return f[n-1]+f[n-2]

def PatternToNumber(Pattern):
    var = list(Pattern)
    for i in range(len(var)):
        if var[i]=='A':
            var[i] = 0
        elif var[i]=='C':
            var[i] = 1
        elif var[i]=='G':
            var[i] = 2
        elif var[i]=='T':
            var[i] = 3
        else:
            print('ERROR: not a DNA string')
    index = sum([var[i]*4**(len(var)-i-1) for i in list(range(len(var)))])
    return index

def NumberToPattern(Number, length):
    counter = [0] * length
    letters = [0] * length
    Pattern = ''
    for i in range(length):
        while Number >= 4**(length - i - 1):
            Number = Number - 4**(length - i - 1)
            counter[i] += 1
    for j in range(length):
        if counter[j] == 0:
            letters[j] = 'A'
        elif counter[j] == 1:
            letters[j] = 'C'
        elif counter[j] == 2:
            letters[j] = 'G'
        elif counter[j] == 3:
            letters[j] = 'T'
        else:
            print('ERROR: not a DNA string')
        Pattern = Pattern + letters[j]
    return Pattern

def Frequencies(Text,k):
    FrequencyArray = [0]*4**k
    for i in range(len(Text)-k+1):
        Pattern = Text[i:i+k]
        j = PatternToNumber(Pattern)
        FrequencyArray[j] = FrequencyArray[j] + 1
    return FrequencyArray

def FindingFrequentWordsBySorting(Text , k):
    FrequentPatterns = ''
    Index = [0]*(len(Text)-k+1)
    Count = [0]*(len(Text)-k+1)
    for i in range(len(Text)-k+1):
        Pattern = Text[i:i+k]
        Index[i] = PatternToNumber(Pattern)
        Count[i] = 1
    SortedIndex = sorted(Index)
    for i in range(1, len(Text)-k+1):
        if SortedIndex[i] == SortedIndex[i-1]:
            Count[i] = Count[i-1] + 1
    maxCount = max(Count)
    for i in range(len(Text)-k+1):
        if Count[i] == maxCount:
            Pattern = NumberToPattern(SortedIndex[i],k)
            FrequentPatterns = FrequentPatterns + Pattern + ' '
    return FrequentPatterns

def ClumpFinding(Genome, k, L, t):
    FrequentPatterns = ''
    Clump = [0]*4**int(k)
    Text = Genome[0:L]
    FrequencyArray = Frequencies(Text, k)
    for i in range(4**k):
        if FrequencyArray[i] >= t:
            Clump[i] = 1
    for i in range(1, len(Genome)-L):
        FirstPattern = Genome[i:i+k]
        index = PatternToNumber(FirstPattern)
        FrequencyArray[index] -= 1
        LastPattern = Genome[i+L-k:i+L]
        index = PatternToNumber(LastPattern)
        FrequencyArray[index] += 1
        if FrequencyArray[index] >= t:
            Clump[index] = 1
    for i in range(4**k):
        if Clump[i] == 1:
            Pattern = NumberToPattern(i, k)
            FrequentPatterns = FrequentPatterns + Pattern + ' '
    return FrequentPatterns

def Skew(Genome):
    Skew = [0]*(len(Genome)+1)
    Skew[0] = 0
    for i in range(1, len(Genome)+1):
        if Genome[i-1] == 'G':
            Skew[i] = Skew[i-1] + 1
        elif Genome[i-1] == 'C':
            Skew[i] = Skew[i-1] - 1
        else:
            Skew[i] = Skew[i-1]
    return Skew

def minSkew(Skew):
    minvalue = min(Skew)
    output = ''
    for i in range(len(Skew)):
        if Skew[i] == minvalue:
            output = output + str(i) + ' '
    return output

def APM(text, pattern, d): # Approximate Pattern Matching
    output = ''
    for i in range(len(text)-len(pattern)+1):
        if HammingDistance(pattern, text[i:i+len(pattern)]) <= d:
            output = output + str(i) + ' '
    return output

def APC(text,pattern,d): # Approximate Pattern Count
    count = 0
    for i in range(len(text)-len(pattern)+1):
        if HammingDistance(pattern, text[i:i+len(pattern)]) <= d:
            count += 1
    return count
            
def FWwM(text, k, d): # Frequent Words with Mismatches
    k=int(k)
    d=int(d)
    frequency = [0]*4**k
    output=''
    for i in range(len(text)-k+1):
        word = text[i:i+k]
        for pattern in Neighbours(word, d):
            number = PatternToNumber(pattern)
            frequency[number] = frequency[number] + 1
    maxfrequency = max(frequency)
    for number in range(len(frequency)):
        if frequency[number] == maxfrequency:
            output = output + str(NumberToPattern(number,k))+' '
    return output

def FWwMRC(text, k, d): # Frequent Words with Mismatches and Reverse Complements
    k=int(k)
    d=int(d)
    frequency = [0]*4**k
    output=''
    for i in range(len(text)-k+1):
        word = text[i:i+k]
        for pattern in Neighbours(word, d):
            number = PatternToNumber(pattern)
            frequency[number] = frequency[number] + 1
    frequency_wRC = [frequency[number]+
                     frequency[PatternToNumber(ReverseComplement(NumberToPattern(number,k)))] for number in range(len(frequency))]
    maxfrequency_wRC = max(frequency_wRC)
    for number in range(len(frequency)):
        if frequency_wRC[number] == maxfrequency_wRC:
            output = output + str(NumberToPattern(number,k))+' '
    return output

def Neighbours(Pattern,d):
    d=int(d)
    if d == 0:
        return Pattern
    letters = ['A','C','G','T']
    if len(Pattern) == 1:
        return letters
    else:
        Neighbourhood = []
        SuffixNeighbours = Neighbours(Pattern[1:],d)
        for text in SuffixNeighbours:
            if HammingDistance(text, Pattern[1:]) < d:
                for letter in letters:
                    Neighbourhood.append(letter + text)
            elif HammingDistance(text, Pattern[1:]) == d:
                Neighbourhood.append(Pattern[0] + text)
    return Neighbourhood

def MotifEnumeration(Dna, k, d):
    Patterns = []
    for line in Dna:
        for i in range(len(line)-k+1):
            Pattern = line[i:i+k]
            for Pattern2 in Neighbours(Pattern,d):
                counter = [0]*len(Dna)
                for neighbour in Neighbours(Pattern2,d):
                    for j in range(len(Dna)):
                        line2 = Dna[j]
                        if Count(line2,neighbour)[0] >= 1:
                            counter[j] = counter[j] + 1
                if min(counter) >= 1:
                    Patterns.append(Pattern2)
    return set(Patterns)

def log2(x):
    if x != 0:
        y = math.log(x,2)
    else:
        y=0
    return y

def Entropy(motifs):
    import math
    entropy = 0
    N = len(motifs)
    count = [[0 for i in range(len(motifs[0]))] for j in range(4)]
    for i in range(N):
        line = motifs[i]
        for j in range(len(line)):
            letter = line[j]
            if letter == 'A':
                count[0][j] += 1/N
            elif letter == 'C':
                count[1][j] += 1/N
            elif letter == 'G':
                count[2][j] += 1/N
            elif letter == 'T':
                count[3][j] += 1/N
            else:
                print('Error: not a DNA sring')
    for j in range(len(line)):
        entropy = entropy -sum([count[i][j]*log2(count[i][j]) for i in range(4)])
    return entropy

def DistanceBetweenPatternAndStrings(Pattern, Dna):
    k = len(Pattern)
    distance = 0
    for line in Dna:
        mind = 99999999
        for i in range(len(line)-k+1):
            word = line[i:i+k]
            d = HammingDistance(Pattern,word)
            if d < mind:
                mind = d
        distance += mind
    return distance

def MedianString(Dna,k):
    distance = 9999
    for i in range(4**k):
        pattern = NumberToPattern(i,k)
        if DistanceBetweenPatternAndStrings(pattern, Dna) < distance:
            distance = DistanceBetweenPatternAndStrings(pattern, Dna)
            median = pattern
    return median

def PmPkmer(text,k,profile): #Profile-most Probable k-mer
    maxprobability = -1
    maxpattern = ''
    for i in range(len(text)-k+1):
        pattern = text[i:i+k]
        patternN = list(pattern)
        for j in range(len(patternN)):
            if   patternN[j] == 'A':
                patternN[j] = 0
            elif patternN[j] == 'C':
                patternN[j] = 1
            elif patternN[j] == 'G':
                patternN[j] = 2
            elif patternN[j] == 'T':
                patternN[j] = 3
            else:
                print('Error: not a DNA sring')
        probability = prod([profile[patternN[i]][i] for i in range(len(pattern))])
        if probability > maxprobability:
            maxprobability = probability
            maxpattern = pattern
    return maxpattern

def Profile(motifs, Laplass):
    N = len(motifs)
    count = [[Laplass for i in range(len(motifs[0]))] for j in range(4)]
    for i in range(N):
        line = motifs[i]
        for j in range(len(line)):
            letter = line[j]
            if letter == 'A':
                count[0][j] += 1/N
            elif letter == 'C':
                count[1][j] += 1/N
            elif letter == 'G':
                count[2][j] += 1/N
            elif letter == 'T':
                count[3][j] += 1/N
            else:
                print('Error: not a DNA sring')
    return count

def ConsensusString(motifs, Laplass):
    profile = Profile(motifs,Laplass)
    consensusstringN = [0]*len(motifs[0])
    consensusstring = ['']*len(motifs[0])
    for i in range(len(motifs[0])):
        maxfreq = max([profile[j][i] for j in range(4)])
        for j in range(4):
            if profile[j][i] == maxfreq:
                consensusstringN[i] = j
    for z in range(len(motifs[0])):
        if consensusstringN[z] == 0:
            consensusstring[z] = 'A'
        elif consensusstringN[z] == 1:
            consensusstring[z] = 'C'
        elif consensusstringN[z] == 2:
            consensusstring[z] = 'G'
        elif consensusstringN[z] == 3:
            consensusstring[z] = 'T'
        else:
            print('Error: not a DNA sring')
    cstring = ''
    for letter in consensusstring:
        cstring = cstring + letter
    return cstring
    
def Score(motifs,Laplass):
    cstring = ConsensusString(motifs,Laplass)
    HD = 0
    for motif in motifs:
        HD = HD + HammingDistance(cstring,motif)
    return HD

def GreedyMotifSearch(Dna,k,t,Laplass):
    BestMotifs = [line[:k] for line in Dna]
    motif = ['' for i in range(t)]
    for i in range(len(Dna[0])-k+1):
        motif[0] = Dna[0][i:i+k]
        for j in range(1,t):
            motif[j] = PmPkmer(Dna[j],k,Profile(motif[0:j], Laplass))
        if Score(motif,Laplass) < Score(BestMotifs,Laplass):
            BestMotifs = motif.copy()
    return BestMotifs

lines = open('J:/YandexDisk/Stepik/Bioinformatics Algorithms/dataset_160_9.txt').readlines()
for i in range(len(lines)):
    while lines[i][-1] == '\n':
        lines[i] = lines[i][:-1]
#text = lines[0]
#k = lines[1]
#d = [lines[i].split() for i in range(2,len(lines))]
#profile = [[float(d[j][i]) for i in range(len(d[0]))] for j in range(len(d))]
##print(text)
##print(k)
##print(profile)
##print(PmPkmer(text,int(k),profile))

k = int(lines[0].split()[0])
t = int(lines[0].split()[1])
Dna = lines[1:]
#for word in GreedyMotifSearch(Dna,k,t,1):
#    print(word)
Dna=['CTCGATGAGTAGGAAAGTAGTTTCACTGGGCGAACCACCCCGGCGCTAATCCTAGTGCCC',
     'GCAATCCTACCCGAGGCCACATATCAGTAGGAACTAGAACCACCACGGGTGGCTAGTTTC',
     'GGTGTTGAACCACGGGGTTAGTTTCATCTATTGTAGGAATCGGCTTCAAATCCTACACAG']
print(MedianString(Dna,7))
#answer = MotifEnumeration(text, k, d)
#output=''
#for i in answer:
#    output = output + i + ' '
#print(output)
#print(len(ClumpFinding(Genome, 9, 500, 3).split()))
#values=''
#for value in Skew('GAGCCACCGCGATA'):
#    values=values+str(value)+' '
#print(minSkew(Skew(Genome)))
#print(APM(t,s,int(d)))
#print(APC(text,pattern,int(d)))

#print(FWwMRC(text,k,d))
