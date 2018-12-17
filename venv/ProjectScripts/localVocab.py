from nltk import tokenize
from LanguageModel import LanguageModel

import re
import math

class KLSum():

    def __call__(self, text, sentencesCount):
        sentences = tokenize.sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            sentences[i] = re.sub(r"[^a-zA-Z0-9]+", ' ', sentence)
        sentences = [sent for sent in sentences if len(sent) > 2]
        #for i, sentence in enumerate(sentences):
            #print("Sentence: ", sentence)
        ratings = self.getScores(sentences)
        sortedRatings = sorted(ratings.items(), key=lambda value: value[1])
        for sent, val in sortedRatings:
            print(sent, val)

    def getScores(self, sentences):
        ratings = self.computeScores(sentences)
        return ratings

    def computeScores(self, sentences):
        wordFreq, wordList = self.computeTF(sentences)
        lm = LanguageModel()
        wordFreqCorpus, corpusLength = lm()

        ratings = {}
        summary = []
        summaryWordList = []

        docLength = 0
        for s in wordList:
            for w in s:
                docLength += 1

        # Removes one sentence per iteration by adding to summary
        while len(sentences) > 0:
            print(len(sentences))
            # will store all the kls values for this pass
            kls = []

            # converts summary to word list
            if summary:
                for sent in summary:
                    if tokenize.word_tokenize(sent):
                        for word in tokenize.word_tokenize(sent):
                            summaryWordList.append(word)

            sentenceCount = 0
            for s in wordList:
                weight = self.computeSentenceWeight(sentences[sentenceCount], wordFreq, wordFreqCorpus, docLength, corpusLength)
                sentenceCount += 1

                # calculates the joint frequency through combining the word lists
                jointFreq = self.computeJointFreq(s, summaryWordList)

                # adds the calculated kl divergence to the list in index = sentence used
                #print(self.klDivergence(jointFreq, wordFreq))
                kls.append(self.klDivergence(jointFreq, wordFreq, weight))

            # to consider and then add it into the summary
            indexToRemove = self.getIndexOfBestSentence(kls)
            bestSentence = sentences.pop(indexToRemove)
            del wordList[indexToRemove]
            summary.append(bestSentence)

            # value is the iteration in which it was removed multiplied by -1 so that the first sentences removed (the most important) have highest values
            ratings[bestSentence] = -1 * len(ratings)

        return ratings

    def computeTF(self, sentences):
        results = []
        for s in sentences:
            results.append(tokenize.word_tokenize(s))
        #stop words? normalize?
        wordFrequency = self.computeWordFreq(results)
        print("Results: ", results)
        return wordFrequency, results

    def computeWordFreq(self, results):
        wordFrequency = {}
        for sentence in results:
            for word in sentence:
                word = word.lower()
                if wordFrequency.get(word) == None:
                    wordFrequency[word] = 1
                else:
                    currFreq = wordFrequency.get(word)
                    wordFrequency[word] = currFreq + 1
        return wordFrequency

    def computeWordFreqSingleList(self, results):
        wordFrequency = {}
        for word in results:
            word = word.lower()
            if wordFrequency.get(word) == None:
                wordFrequency[word] = 1
            else:
                currFreq = wordFrequency.get(word)
                wordFrequency[word] = currFreq + 1
        return wordFrequency

    def computeSentenceWeight(self, sentence, wordFreqDoc, wordFreqCorpus, docLength, corpusLength):
        sentProbCorpus = 0
        sentProbDoc = 0
        sentProb = 0
        counter = 0
        words = tokenize.word_tokenize(sentence)
        for word in words:
            counter += 1
            word = word.lower()
            if not word in wordFreqDoc:
                wordFreqDoc[word] = 0
            if not word in wordFreqCorpus:
                wordFreqCorpus[word] = 0
            sentProbDoc += math.log((wordFreqDoc.get(word) + 0.5 * (wordFreqDoc.get(word)/docLength))/(docLength + 0.5))
            sentProbCorpus += math.log((wordFreqDoc.get(word) + 0.5 * (wordFreqCorpus.get(word)/corpusLength))/(docLength + 0.5))

        sentProbDoc /= counter
        sentProbCorpus /= counter
        sentProb = sentProbDoc - sentProbCorpus
        return sentProb

    def computeJointFreq(self, sentenceList, summaryWordList):
        # combined length of the word lists
        total_len = len(sentenceList) + len(summaryWordList)

        # word frequencies within each list
        wcSentenceList = self.computeWordFreqSingleList(sentenceList)
        wcsSummaryWordList = self.computeWordFreqSingleList(summaryWordList)

        # inputs the counts from the first list
        joint = wcSentenceList.copy()

        # adds in the counts of the second list
        for k in wcsSummaryWordList:
            if k in joint:
                joint[k] += wcsSummaryWordList[k]
            else:
                joint[k] = wcsSummaryWordList[k]

        # divides total counts by the combined length
        for k in joint:
            joint[k] /= float(total_len)

        return joint

    def klDivergence(self, summaryFreq, docFreq, weight):
        sum_val = 0
        for w in summaryFreq:
            frequency = docFreq.get(w)
            #print(type(frequency))
            if frequency:  # missing or zero = no frequency
                #print(frequency / summaryFreq[w])
                sum_val += frequency * math.log(frequency / summaryFreq[w])

        sum_val = weight * sum_val
        return sum_val

    def getIndexOfBestSentence(self, kls):
        return kls.index(min(kls))


            '''raw = parser.from_file("C:/Users/John/PycharmProjects/SummarizationTests/venv/data/" + file)

            tempRaw = raw['content'].splitlines()
            #print(tempRaw)
            newRaw = ''
            counter = 0
            for i in tempRaw:
                if i is '':
                    tempRaw.remove('')

            for i in tempRaw:
                newRaw = newRaw + '\n' + i

            sentences = tokenize.sent_tokenize(newRaw)
            for i, _sentence in enumerate(sentences):
                sentences[i] = re.sub(r"[^a-zA-Z0-9]+", ' ', _sentence)
            sentences = [sent for sent in sentences if len(sent) > 2]

            for s in sentences:
                for w in tokenize.word_tokenize(s):
                    corpusLength += 1

            wordFreqTF = self.computeTF(sentences, wordFreqTF)

        return wordFreqTF, corpusLength

    def computeTF(self, sentences, wordFreqTF):
        results = []
        for s in sentences:
            results.append(tokenize.word_tokenize(s))
        #stop words? normalize?
        wordFrequency = self.computeWordFreq(results, wordFreqTF)
        return wordFrequency

    def computeWordFreq(self, results, wordFreqTF):
        for sentence in results:
            for word in sentence:
                word = word.lower()
                if wordFreqTF.get(word) == None:
                    wordFreqTF[word] = 1
                else:
                    currFreq = wordFreqTF.get(word)
                    wordFreqTF[word] = currFreq + 1
        return wordFreqTF'''