# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

import math
import operator

from _summarizer import AbstractSummarizer
from LanguageModel import LanguageModel


class KLSum(AbstractSummarizer):
    """
    Method that greedily adds sentences to a summary so long as it decreases the
    KL Divergence.
    Source: http://www.aclweb.org/anthology/N09-1041
    """

    stop_words = frozenset()

    def __call__(self, document, sentences_count):
        ratings, selectedSentences = self._get_ratings(document)
        return self._get_best_sentences(selectedSentences, sentences_count, ratings)

    def _get_ratings(self, document):
        sentences = document.sentences
        ratings, selectedSentences = self._compute_ratings(sentences)
        return ratings, selectedSentences

    def _get_all_words_in_doc(self, sentences):
        return [w for s in sentences for w in s.words]

    def _get_content_words_in_sentence(self, sentence):
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        return normalized_content_words

    def _normalize_words(self, words):
        return [self.normalize_word(w) for w in words]

    def _filter_out_stop_words(self, words):
        return [w for w in words if w not in self.stop_words]

    def _compute_word_freq(self, list_of_words):
        word_freq = {}
        for w in list_of_words:
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def _get_all_content_words_in_doc(self, sentences):
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words

    def compute_tf(self, sentences):
        """
        Computes the normalized term frequency as explained in http://www.tfidf.com/

        :type sentences: [sumy.models.dom.Sentence]
        """
        content_words = self._get_all_content_words_in_doc(sentences)
        content_words_count = len(content_words)
        content_words_freq = self._compute_word_freq(content_words)
        content_word_tf = dict((w, f / content_words_count) for w, f in content_words_freq.items()) #2
        return content_word_tf, content_words_count, content_words_freq

    def _joint_freq(self, word_list_1, word_list_2):
        # combined length of the word lists
        total_len = len(word_list_1) + len(word_list_2)

        # word frequencies within each list
        wc1 = self._compute_word_freq(word_list_1)
        wc2 = self._compute_word_freq(word_list_2)

        # inputs the counts from the first list
        joint = wc1.copy()

        # adds in the counts of the second list
        for k in wc2:
            if k in joint:
                joint[k] += wc2[k]
            else:
                joint[k] = wc2[k]

        # divides total counts by the combined length
        #for k in joint:
            #joint[k] /= float(total_len)

        return joint

    def _kl_divergence(self, summary_freq, doc_freq, docLength):
        """
        Note: Could import scipy.stats and use scipy.stats.entropy(doc_freq, summary_freq)
        but this gives equivalent value without the import
        """
        sum_val = 0
        for w in doc_freq:
            frequency = doc_freq.get(w)
            if not w in summary_freq:
                summary_freq[w] = 0
            wordProb = ((summary_freq[w] + 5000 * doc_freq.get(w)) / (docLength + 5000))
            sum_val += frequency * math.log(frequency / wordProb)

        return sum_val

    def _find_index_of_best_sentence(self, kls):
        """
        the best sentence is the one with the smallest kl_divergence
        """
        return kls.index(min(kls))

    def computeSentenceWeight(self, sentence, wordTF, wordFreqCorpus, docLength):
        LANGUAGE = "english"
        sentProbCorpus = 0
        sentProbDoc = 0
        sentProb = 0
        counter = 0

        for word in sentence:
            counter += 1
            word = word.lower()
            if not word in wordFreqCorpus:
                wordFreqCorpus[word] = 0
            if not word in wordTF:
                wordTF[word] = 0
            sentProbDoc += math.log((wordTF.get(word) + 5000 * wordFreqCorpus.get(word))/(docLength + 5000))
            sentProbCorpus += math.log(wordFreqCorpus.get(word))
            #print(((wordTF.get(word) + 5000 * wordFreqCorpus.get(word))/(docLength + 5000))/(wordFreqCorpus.get(word)))

        sentProb = sentProbDoc - sentProbCorpus
        return sentProb

    def _compute_ratings(self, sentences):
        wordFreq, docLength, wordTF = self.compute_tf(sentences)
        ratings = {}
        summary = []
        lm = LanguageModel()
        wordFreqCorpus = lm()

        sortedWordFreqCorpus = sorted(wordFreqCorpus.items(), key=operator.itemgetter(1))
        sortedWordFreq = sorted(wordFreq.items(), key=operator.itemgetter(1), reverse=True)

        diffProb = {}
        for w in sortedWordFreqCorpus:
            for w2 in sortedWordFreq:
                if w[0] == w2[0]:
                    diffProb[w[0]] = w2[1] - w[1]
                    break

        sortedDiffProb = sorted(diffProb.items(), key=operator.itemgetter(1), reverse=True)
        for w in sortedDiffProb:
            print(w)


        # make it a list so that it can be modified
        sentences_list = list(sentences)

        # get all content words once for efficiency
        sentences_as_words = [self._get_content_words_in_sentence(s) for s in sentences]

        listOfSentWeightTuples = []
        for words, sent in zip(sentences_as_words, sentences):
            weight = self.computeSentenceWeight(words, wordTF, wordFreqCorpus, docLength)
            sentPlusWeight = (sent, weight)
            listOfSentWeightTuples.append(sentPlusWeight)

        listOfSentWeightTuples.sort(key=operator.itemgetter(1), reverse=True)
        del listOfSentWeightTuples[100:]

        sent_list = [sent[0] for sent in listOfSentWeightTuples]

        #for sent in listOfSentWeightTuples:
            #print(sent)

        print()

        sentencesInOrder = []
        for s in sentences:
            for sent in listOfSentWeightTuples:
                if s == sent[0]:
                    sentencesInOrder.append(sent[0])

        sent_as_words = [self._get_content_words_in_sentence(s) for s in sent_list]

        # Removes one sentence per iteration by adding to summary
        while len(sent_list) > 0:
            # will store all the kls values for this pass
            kls = []

            # converts summary to word list
            summary_as_word_list = self._get_all_words_in_doc(summary)

            sentenceCount = 0
            for s in sent_as_words:
                # calculates the joint frequency through combining the word lists
                joint_freq = self._joint_freq(s, summary_as_word_list)

                # adds the calculated kl divergence to the list in index = sentence used
                kls.append(self._kl_divergence(joint_freq, wordFreq, docLength)) #continue to use chapter freq or switch to the sentences remaining freq

            # to consider and then add it into the summary
            indexToRemove = self._find_index_of_best_sentence(kls)
            best_sentence = sent_list.pop(indexToRemove)
            del sent_as_words[indexToRemove]
            summary.append(best_sentence)

            # value is the iteration in which it was removed multiplied by -1 so that the first sentences removed (the most important) have highest values
            ratings[best_sentence] = -1 * len(ratings)

        return ratings, sentencesInOrder