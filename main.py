import json
import math

from tqdm import tqdm
from numba import njit, typed, prange


@njit
def toBaseN(n, N):
    strbin = ""
    while n != 0:
        strbin += str(n % N)
        n = n // N
    return strbin[::-1]


wordsG = None
w_freq = None
with open('./freq_map.json') as f:
    w_freq = json.load(f)

with open('wordlist.txt') as f:
    wordsG = f.readlines()

wordC = len(wordsG)

for i in range(wordC):
    wordsG[i] = wordsG[i].strip()


@njit
def matchPattern(pattern, word):
    # known
    for i in range(5):
        if pattern[0][i] != '':
            if word[i] != pattern[0][i]: return False

    # contains
    for l in pattern[1]:
        if not l in word and l != '*': return False

    # do not have
    for l in pattern[2]:
        if l in word: return False

    return True


@njit
def genPattern(pattern, word):
    p = [
        ['', '', '', '', ''],
        ['*'],
        ['*']
    ]

    i = 0
    for l in pattern:
        if l == '2':
            p[0][i] = word[i]
        elif l == '1':
            p[1].append(word[i])
        else:
            p[2].append(word[i])
        i += 1

    return p


@njit
def getMatches(pattern, words):
    r = []
    for w in words:
        if matchPattern(pattern, w):
            r.append(w)
    return r


@njit(parallel=True)
def calculateSum(patterns, word, words):
    s = 0
    for p in prange(243):
        pattern = genPattern(patterns[p], word)
        matched = getMatches(pattern, words)
        px = len(matched) / len(words)
        if px == 0: continue
        Ix = math.log(1.0 / px) / math.log(2.0)
        s += px * Ix
    return s

def checkWordEI(word, words):
    patterns = [toBaseN(p, 3).zfill(5) for p in range(243)]
    return calculateSum(typed.List(patterns), word, words)


def sortByInformation(words=wordsG):
    # return sorted(wordsG, key=lambda x: checkWordEI(x, typed.List(wordsG)))
    a = words.copy()
    tl = typed.List(words)
    for i in tqdm(range(len(a)), desc='analyzing wordlist', unit='words'):
        a[i] = [a[i], checkWordEI(a[i], tl)]

    a.sort(key=lambda x: x[1], reverse=True)
    return a


def sortByFreqAndEI(words=wordsG, step=0):
    # temp = [[v, v[1]*(5-step)+w_freq[v[0]]*2000, w_freq[v[0]]*2000] for v in words]
    # print(sorted(temp, key=lambda l: l[2], reverse=True))
    return sorted(words, key=lambda v: v[1] * (5 - step) + w_freq[v[0]] * (3000 * step), reverse=True)


@njit
def sortW(words, pattern, word):
    return getMatches(genPattern(pattern, word), words)


opener = 'crane'
if __name__ == "__main__":

    pr = input(f'pattern ({opener})>> ')
    print('matching and compiling...')
    wordsN = sortW(typed.List(wordsG), pr, opener)
    for score in range(1, 6):
        r = sortByInformation(wordsN)

        if len(r) == 0:
            print('There are no more options')
            exit()

        pw = [e[0] for e in sortByFreqAndEI(r, score)]

        if len(pw) == 1:
            print(f'the answer is definitely: {pw[0]}')
            exit()

        print('preferred words: ', *pw)
        wordsN = list([e[0] for e in r])
        print(f'possible words: {", ".join(wordsN)}', f'best guess: {wordsN[0]}', f'suggested answer: {pw[0]}',
              sep='\n')
        print(str(len(wordsN)) + '/' + str(wordC))
        sw = input('selected word >> ')
        while not sw in wordsN:
            sw = input('selected word (from list) >> ')
        isw = wordsN.index(sw)
        pr = input(f'pattern ({wordsN[isw]})>> ')
        print('matching...')
        wordsN = sortW(typed.List(wordsN), pr, wordsN[isw])
        try:
            wordsN.remove(sw)
        except:
            pass