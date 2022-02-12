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
w_memo = None
with open('./freq_map.json') as f:
    w_freq = json.load(f)

with open('./words_memory.json') as f:
    w_memo = json.load(f)

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
        if pattern[3][i] != '':
            if word[i] == pattern[3][i]: return False


    # contains
    for l in pattern[1]:
        if not l in word and l != '*': return False

    # do not have
    for l in pattern[2]:
        if l in word: return False

    return True


@njit()
def genPattern(pattern, word):
    p = [
        ['', '', '', '', ''],
        ['*'],
        ['*'],
        ['', '', '', '', '']
    ]
    po = []

    i = 0
    for l in pattern:
        if word[i] in po:
            i += 1
            continue
        po.append(word[i])
        if l == '2':
            p[0][i] = word[i]
        elif l == '1':
            p[1].append(word[i])
            p[3][i] = word[i]
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


def sortByInformation(words=wordsG, skipTqdm=False):
    # return sorted(wordsG, key=lambda x: checkWordEI(x, typed.List(wordsG)))
    a = words.copy()
    tl = typed.List(words)
    for i in tqdm(range(len(a)), desc='analyzing wordlist', unit='words', disable=skipTqdm):
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

def deepGraph(word, words=wordsG, memo=w_memo, isTop=True, depth=0, md=6):
    if depth >= md+1:
        return
    if depth<=3: print('E', word, depth, f'{len(words)} words')
    patterns = [toBaseN(p, 3).zfill(5) for p in range(243)]

    for p in patterns:
        wordsN = sortW(typed.List(words), p, word)

        r = None

        if isTop and word in w_memo and p in w_memo[word]:
            print('pattern found in memory!')
            r = w_memo[word][p]
        else:
            r = sortByInformation(wordsN, True)

        if not word in memo:
            memo[word] = {}

        if len(r) != 1:
            memo[word][p] = {'m': r, 'v': {}}
            i,m=0,len(r)
            for wf in r:
                if depth == 0: print(f'{i}/{m}')
                w = wf[0]
                memo[word][p]['v'][w] = {}
                # print('DG', w, wordsN)
                deepGraph(w, wordsN, memo[word][p]['v'][w], False, depth+1, md)
                i+=1
        else:
            # print('OR', r[0][0])
            memo[word][p] = r[0][0]

        if depth == 0:
            with open('./words_memory_deep.json', 'w') as f:
                json.dump(w_memo, f)
            print('saved data')


def generateAllForWord(word, words=wordsG):
    patterns = [toBaseN(p, 3).zfill(5) for p in range(243)]

    for p in patterns:
        print('Matching ' + p + ' for ' + word)
        wordsN = sortW(typed.List(words), p, word)
        print('Analyzing ' + p + ' for ' + word)
        r = sortByInformation(wordsN)
        if not word in w_memo:
            w_memo[word] = {}
        w_memo[word][p] = r
        with open('./words_memory.json', 'w') as f:
            json.dump(w_memo, f)

opener = 'crane'
lastword = opener
if __name__ == "__main__":

    pr = input(f'pattern ({opener})>> ')
    print('matching and compiling...')
    wordsN = sortW(typed.List(wordsG), pr, opener)
    for score in range(1, 6):

        r = None

        if score == 1 and lastword in w_memo and pr in w_memo[lastword]:
            print('pattern found in memory!')
            r = w_memo[lastword][pr]
        else:
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

        if score == 1:
            if not lastword in w_memo:
                w_memo[lastword] = {}
            w_memo[lastword][pr] = r
            with open('./words_memory.json', 'w') as f:
                json.dump(w_memo, f)

        print(str(len(wordsN)) + '/' + str(wordC))
        sw = input('selected word >> ')
        while not sw in wordsN:
            sw = input('selected word (from list) >> ')
        isw = wordsN.index(sw)
        lastword = sw
        pr = input(f'pattern ({wordsN[isw]})>> ')
        print('matching...')
        wordsN = sortW(typed.List(wordsN), pr, wordsN[isw])
        try:
            wordsN.remove(sw)
        except:
            pass