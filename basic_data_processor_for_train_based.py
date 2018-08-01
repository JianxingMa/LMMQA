from nltk import word_tokenize
from nltk import sent_tokenize
from string import punctuation
from nltk.corpus import stopwords
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from data import Data
from config import Config
from scnn import Trainer
import pickle


class BasicDataProcessorForTrain:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.lemmatizer = WordNetLemmatizer()
        self.word2vec_model = KeyedVectors.load_word2vec_format(self.config.word2vec_model_path,
                                                                binary=True)
        self.trn = Trainer()

    ''' generate dev question embeddings, trancate the question if  the question length is 
        larger than 30 and complement it if the length is smaller than 30
    '''

    def generate_dev_qs_embeddings(self):
        question_vectors = []
        qs = []
        total = int(len(self.data.dev_qs_processed))
        for i in range(total):
            if i % 1000 == 0:
                print(i, '/', total)
            dev_qs = self.data.dev_qs_processed[i]
            q_vector = []
            for token in dev_qs:
                if token in self.word2vec_model.vocab:
                    q_vector.append(list(self.word2vec_model[token]))
            if len(q_vector) < 30:
                for k in range(30 - len(q_vector)):
                    q_vector.append([0] * 300)
            elif len(q_vector) > 30:
                q_vector = q_vector[:30]
            question_vectors.append(q_vector)
            qs.append(dev_qs)
        return question_vectors, qs
        with open('ner.pkl') as f:
            answer_tags = pickle.load(f)
        print
        len(answer_tags)
        print
        answer_tags[0]
        print
        np.array(question_vectors).shape
        self.trn.load_ner_data(question_vectors, answer_tags)

    ''' generate training vectors which contains 30-length-embedding sentences
        for each question there is a positive sentence sample and several negative 
        sentence samples
    '''

    def generate_training_embeddings(self):
        question_vectors = []
        sent_vectors = []
        label = []
        total = int(len(self.data.train_qs_processed))
        for i in range(total):
            if i % 1000 == 0:
                print(i, '/', total)
            train_qs = self.data.train_qs_processed[i]
            train_answer = self.data.train_answers[i]
            train_par_id = self.data.train_answer_par_ids[i]
            train_doc_id = self.data.train_doc_ids[i]
            train_par = self.data.doc_texts[train_doc_id][train_par_id]
            train_sents = sent_tokenize(train_par)
            q_vector = []
            for token in train_qs:
                if token in self.word2vec_model.vocab:
                    q_vector.append(list(self.word2vec_model[token]))

            if len(q_vector) > 30:
                continue

            if len(q_vector) < 30:
                for k in range(30 - len(q_vector)):
                    q_vector.append([0] * 300)

            for sent_id in range(len(train_sents)):
                if train_answer in train_sents[sent_id]:
                    sent_vector = self.generate_sent_embedding(train_sents[sent_id])
                    if sent_vector:
                        question_vectors.append(q_vector)
                        sent_vectors.append(sent_vector)
                        label.append(1)

                        for wrong_id in range(len(train_sents)):
                            if wrong_id == sent_id:
                                continue
                            else:
                                wrong_sent_vector = self.generate_sent_embedding(
                                    train_sents[wrong_id])
                                if wrong_sent_vector:
                                    question_vectors.append(q_vector)
                                    sent_vectors.append(wrong_sent_vector)
                                    label.append(0)
                        break
        with open(self.config.sentence_embedding_pkl, 'wb') as f:
            pickle.dump([question_vectors, sent_vectors, label], f)

    '''
        ignore too long sentence and complement short sentences 
    '''

    def generate_sent_embedding(self, sent):
        sent_vector = []
        sent_tokens = self.process_sent(sent)
        for sent_token in sent_tokens:
            if sent_token in self.word2vec_model.vocab:
                sent_vector.append(list(self.word2vec_model[sent_token]))
        if len(sent_vector) > 30:
            return []
        elif len(sent_vector) < 30:
            for k in range(30 - len(sent_vector)):
                sent_vector.append([0] * 300)
        return sent_vector

    def preprocess_questions(self, questions):
        return [self.preprocess_question(q) for q in questions]

    def process_docs(self, docs):
        return [self.preprocess_doc(doc) for doc in docs]

    def preprocess_question(self, question):
        normal_tokens = word_tokenize(question.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        remove_punc_in_tokens = [self.remove_punc_in_token(token) for token in remove_punc_tokens]
        lower_tokens = self.lower_tokens(remove_punc_in_tokens)
        lemmatized_tokens = self.lemmatize_tokens(lower_tokens)
        return lemmatized_tokens

    def is_pure_puncs(self, tokens):
        if all([token in punctuation for token in tokens]):
            return True
        return False

    def remove_punc_in_token(self, token):
        return ''.join([x for x in token if x not in punctuation]).strip()

    def remove_stop_words(self, words):
        return [word for word in words if word.lower() not in stopwords.words("english")]

    def lemmatize_tokens(self, words):
        return [self.lemmatize(word) for word in words]

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
        lemmatized_tokens = [self.lemmatize_tokens(tokens) for tokens in lower_tokens]
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def process_sent(self, sens):
        normal_tokens = word_tokenize(sens.replace("\u200b", '').replace("\u2014", ''))
        remove_punc_tokens = [token for token in normal_tokens if not self.is_pure_puncs(token)]
        replace_numbers = []
        for token in remove_punc_tokens:
            if any([x.isdigit() for x in token]):
                replace_numbers.append('number')
            else:
                replace_numbers.append(token)
        lower_tokens = self.lower_tokens(replace_numbers)
        lemmatized_tokens = self.lemmatize_tokens(lower_tokens)
        return lemmatized_tokens

    def lower_tokens(self, words):
        return [word.lower() for word in words]

    def ner_tagging(self, sents):
        ner_sents = self.tagger.tag_sents(sents)
        processed_ners = []
        for i in range(len(ner_sents)):
            sent = sents[i]
            pos_sent = pos_tag(sent)
            ner_sent = ner_sents[i]
            processed_ner_sent = []
            for j in range(len(ner_sent)):
                span, tag = ner_sent[j]
                _, pos = pos_sent[j]
                if span.isdigit() or pos == 'CD':
                    processed_ner_sent.append((span, 'NUMBER'))
                elif tag == 'ORGANIZATION':
                    processed_ner_sent.append((span, 'OTHER'))
                elif j != 0 and tag == 'O' and span[0].isupper():
                    processed_ner_sent.append((span, 'OTHER'))
                else:
                    processed_ner_sent.append((span, tag))
            processed_ners.append(processed_ner_sent)
        return processed_ners


if __name__ == '__main__':
    data = Data()
    config = Config()
    bdp = BasicDataProcessor(config, data)
