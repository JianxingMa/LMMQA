from nltk import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger
from nltk import pos_tag
from data import Data
from config import Config
from nltk.parse.stanford import StanfordDependencyParser
from stanfordcorenlp import StanfordCoreNLP


class BasicDataProcessor:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        self.tagger = StanfordNERTagger(model_filename=self.config.ner_model_path)
        self.postagger = StanfordPOSTagger(path_to_jar=self.config.pos_jar_path,
                                           model_filename=self.config.pos_model_path)
        self.dependency_parser = StanfordDependencyParser(path_to_jar=self.config.parser_jar_path,
                                                          path_to_models_jar=self.config.parser_model_path)
        self.nlp = StanfordCoreNLP("C:/UOM/SML/stanford-corenlp-full-2018-02-27/stanford-corenlp-full-2018-02-27")
        self.punc = r"""!"#&'()*+;<=>?[]^`{}~"""


    def preprocess_questions(self, questions):
        return [self.preprocess_question(q) for q in questions]

    def process_docs(self, docs):
        return [self.preprocess_doc(doc) for doc in docs]

    def preprocess_question(self, question):
        normal_tokens = word_tokenize(question.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        lower_tokens = self.lower_tokens(remove_punc_in_tokens)
        remove_stop_tokens = self.remove_stop_words(lower_tokens)
        for i in range(len(remove_stop_tokens)):
            if remove_stop_tokens[i] == 'where':
                remove_stop_tokens[i] = 'location'
            if remove_stop_tokens[i] == 'when':
                remove_stop_tokens[i] = 'time'
            if remove_stop_tokens[i] == 'who' or remove_stop_tokens[i] == 'whom':
                remove_stop_tokens[i] = 'person'
            if remove_stop_tokens[i] == 'why':
                remove_stop_tokens[i] = 'reason'
        lemmatized_tokens = self.lemmatize_tokens(remove_stop_tokens)
        return lemmatized_tokens

    def is_pure_puncs(self, token):
        if all([c in punctuation for c in token]):
            return True
        return False

    # remove punctuations within a token
    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    # remove punctuations within a token if the punctuation is not in puc set
    def remove_punc_in_token_for_rule(self, token):
        return ''.join([x for x in token if x not in self.punc]).strip()

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in stopwords.words("english")]

    def lemmatize_tokens(self, words):
        return [self.lemmatize(word.lower()) for word in words]

    def lemmatize(self, word):
        word = word.lower()
        lemma = self.lemmatizer.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.lemmatizer.lemmatize(word, 'n')
        return lemma

    def preprocess_doc(self, doc):
        normal_tokens = [word_tokenize(par.replace(u"\u200b", '').replace(u"\u2014", '')) for par in
                         doc]
        remove_punc_tokens = [[token for token in tokens if not self.is_pure_puncs(token)] for
                              tokens in normal_tokens]
        remove_punc_in_tokens = [[self.remove_punc_in_token(token) for token in tokens] for tokens
                                 in remove_punc_tokens]
        lower_tokens = [self.lower_tokens(tokens) for tokens in remove_punc_in_tokens]
        remove_stop_tokens = [self.remove_stop_words(tokens) for tokens in lower_tokens]
        lemmatized_tokens = [self.lemmatize_tokens(tokens) for tokens in remove_stop_tokens]
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def process_sent(self, sens):
        normal_tokens = word_tokenize(sens.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        ner_tags = self.sens_ner_tagging(remove_punc_in_tokens)
        replaced_tokens = [
            'number' if tup[1] == 'NUMBER' else 'person' if tup[1] == 'PERSON' else 'location' if
            tup[1] == 'LOCATION' else tup[0].lower() for tup in ner_tags]
        lower_tokens = self.lower_tokens(replaced_tokens)
        remove_stop_tokens = self.remove_stop_words(lower_tokens)
        lemmatized_tokens = self.lemmatize_tokens(remove_stop_tokens)
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def sens_ner_tagging(self, sent):
        ner_sents = self.tagger.tag_sents([sent])
        pos_sent = pos_tag(sent)
        ner_sent = ner_sents[0]
        processed_ner_sent = []
        for j in range(len(ner_sent)):
            span, tag = ner_sent[j]
            _, pos = pos_sent[j]
            if span.isdigit() or pos == 'CD':
                processed_ner_sent.append((span, 'NUMBER'))
            elif tag == 'PERSON':
                processed_ner_sent.append((span, 'PERSON'))
            elif tag == 'LOCATION':
                processed_ner_sent.append((span, 'LOCATION'))
            elif tag == 'ORGANIZATION':
                processed_ner_sent.append((span, 'OTHER'))
            elif j != 0 and tag == 'O' and span[0].isupper():
                processed_ner_sent.append((span, 'OTHER'))
            else:
                processed_ner_sent.append((span, tag))
        return processed_ner_sent

    def lemmatize_entity_name(self, entity_name):
        tokens = entity_name.split()
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)


if __name__ == '__main__':
    data = Data()
    config = Config()
    bdp = BasicDataProcessor(config, data)
