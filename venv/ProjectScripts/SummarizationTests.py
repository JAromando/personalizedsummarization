from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
#from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from tika import parser

from kl import KLSum
from klTwo import KLSummarizer
from klParsimon import KLSumParsim

LANGUAGE = "english"
SENTENCES_COUNT = 7
FILE_LOCATION = "C:/Users/John/Desktop/01boolANSI.txt"

#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterThreeMitchellSP.txt"
#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterThreeMitchellGrobid.txt"

#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterFiveJefferiesSP.txt"
#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterFiveJefferiesGrobid.txt"

#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterOneYokleySP.txt"
#FILE_LOCATION = "C:/Users/John/PycharmProjects/SummarizationTests/chapterOneYokleyGrobid.txt"


def lsaSummarization(_parser):

    #url = ""
    #parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))

    stemmer = Stemmer(LANGUAGE)

    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

def lexRankSummarization(_parser):

    #stemmer = Stemmer(LANGUAGE)

    summarizer = LexRankSummarizer()
    #summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

def reductionSummarization(_parser):

    stemmer = Stemmer(LANGUAGE)

    summarizer = ReductionSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

def klSummarization(_parser):

    stemmer = Stemmer(LANGUAGE)

    summarizer = KLSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

def klSummarizationAltered(_parser):

    stemmer = Stemmer(LANGUAGE)

    summarizer = KLSum(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

    '''summarizer = KLSum();
    summary = summarizer(_parser, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)'''

def ParsimoniousSummarization(_parser):

    stemmer = Stemmer(LANGUAGE)

    summarizer = KLSumParsim(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    summary = summarizer(_parser.document, SENTENCES_COUNT)

    for sentence in summary:
        print(sentence)

if __name__ == "__main__":

    '''raw = parser.from_file("C:/Users/John/PycharmProjects/SummarizationTests/venv/data/chapterFocus/07system.pdf")
    #raw = parser.from_file("C:/Users/John/Desktop/JAromandoVirginiaTechPersonalStatement.pdf")

    tempRaw = raw['content'].splitlines()
    print(tempRaw)
    newRaw = ''
    counter = 0
    for i in tempRaw:
        if i is '':
            tempRaw.remove('')

    print(tempRaw)

    for i in tempRaw:
        newRaw = newRaw + '\n' + i'''

    raw = parser.from_file("C:/Users/johna/PycharmProjects/personalizedsummarization/venv/data/chapterFocus/03dict.pdf")

    txtfile = "C:/Users/johna/PycharmProjects/personalizedsummarization/venv/data/chapterFocus/03dict.txt"

    text_file = open(txtfile, "w", encoding="raw_unicode_escape")
    text_file.write(raw['content'])
    text_file.close()

    parser = PlaintextParser.from_file(txtfile, Tokenizer(LANGUAGE))

    txtfile = "C:/Users/johna/PycharmProjects/personalizedsummarization/venv/data/chapterFocus/03dictjson.txt"

    with open(txtfile, "w") as myfile:
        for sent in parser.document.sentences:
            #if not str(sent).endswith("."):
                #myfile.write('"' + str(sent) + '.",\n')
                myfile.write('"' + str(sent) + '",\n')

    myfile.close()

    print("CLOSED")

    '''print()
    lsaSummarization(parser)
    print()
    print()
    print()
    lexRankSummarization(parser)
    print()
    print()
    print()
    reductionSummarization(parser)
    print()
    print()
    print()'''
    klSummarization(parser)
    print()
    print()
    print()
    klSummarizationAltered(parser)
    print()
    print()
    print()
    ParsimoniousSummarization(parser)
