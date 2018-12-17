from tika import parser
from sumy.parsers.plaintext import PlaintextParser
from nltk import tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from _compat import to_unicode


import os
import math
import re
import io

class LanguageModel():

    def __call__(self):
        wordFreqCorpus, corpusLength = self.computeSentenceProbability()
        return wordFreqCorpus, corpusLength

    def computeSentenceProbability(self):

        global corpusLength
        corpusLength = 0
        global content_words_count
        content_words_count = 0

        LANGUAGE = "english"

        pdfFiles = [i for i in os.listdir("C:/Users/John/PycharmProjects/SummarizationTests/venv/data") if i.endswith("pdf")]
        wordFreqTF = {}
        for file in pdfFiles:

            raw = parser.from_file("C:/Users/John/PycharmProjects/SummarizationTests/venv/data/" + file)

            fileSplit = file.split(".")

            txtfile = fileSplit[0] + ".txt"

            text_file = open(txtfile, "w", encoding="utf-8")
            text_file.write(raw['content'])
            text_file.close()

        txtFiles = [i for i in os.listdir("C:/Users/John/PycharmProjects/SummarizationTests/venv/data") if
                 i.endswith("txt")]
        for file in txtFiles:

            _parser = PlaintextParser.from_file("C:/Users/John/PycharmProjects/SummarizationTests/venv/data/" + file, Tokenizer(LANGUAGE))

            stemmer = Stemmer(LANGUAGE)

            #summarizer = KLSum(stemmer)
            stop_words = get_stop_words(LANGUAGE)

            sentences = _parser.document.sentences

            wordFreqTF = self.compute_tf(sentences, wordFreqTF, stop_words)

        content_word_tf = dict((w, f / content_words_count) for w, f in wordFreqTF.items())

        return content_word_tf, corpusLength

    def compute_tf(self, sentences, wordFreqTF, stop_words):
        """
        Computes the normalized term frequency as explained in http://www.tfidf.com/

        :type sentences: [sumy.models.dom.Sentence]
        """
        content_words = self._get_all_content_words_in_doc(sentences, stop_words)
        global content_words_count
        content_words_count += len(content_words)
        content_words_freq = self._compute_word_freq(content_words, wordFreqTF)
        return content_words_freq

    def _get_all_words_in_doc(self, sentences):
        return [w for s in sentences for w in s.words]

    def _get_content_words_in_sentence(self, sentence):
        normalized_words = self._normalize_words(sentence.words)
        normalized_content_words = self._filter_out_stop_words(normalized_words)
        return normalized_content_words

    def _normalize_words(self, words):
        return [to_unicode(w).lower() for w in words]

    def _filter_out_stop_words(self, words, stop_words):
        return [w for w in words if w not in stop_words]

    def _compute_word_freq(self, list_of_words, wordFreqTF):
        word_freq = wordFreqTF
        for w in list_of_words:
            global corpusLength
            corpusLength += 1
            word_freq[w] = word_freq.get(w, 0) + 1
        return word_freq

    def _get_all_content_words_in_doc(self, sentences, stop_words):
        all_words = self._get_all_words_in_doc(sentences)
        content_words = self._filter_out_stop_words(all_words, stop_words)
        normalized_content_words = self._normalize_words(content_words)
        return normalized_content_words











