from config import Config
import numpy as np
import pickle
from collections import defaultdict
import unicodecsv as csv
from file_loader import FileLoader
from data import Data
from basic_data_processor import BasicDataProcessor
from bm25 import BM25
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk import sent_tokenize


class RuleBasedQA:
    def __init__(self):
        # switch of train, dev, test model
        train = 0
        dev = 0
        test = 1

        # switch of loading data from pkl or reprocessing
        load_processed_doc = 1
        load_doc_from_pkl = 1

        # switch of testing BM25 accuracy
        test_BM25 = 0

        self.data = Data()
        self.config = Config()
        self.fileLoader = FileLoader(self.config, self.data)
        self.bdp = BasicDataProcessor(self.config, self.data)
        self.bm25 = BM25(self.config, self.data)

        # not used ner tags, will merge them together with 'O' tag
        self.other = ['SET', "MISC", 'EMAIL', 'URL', 'TITLE', 'IDEOLOGY', 'CRIMINAL_CHARGE']

        self.fileLoader.load_doc()

        # load doc data
        if load_processed_doc:
            if load_doc_from_pkl:
                with open(self.config.doc_processed_path, 'rb') as f:
                    self.data.doc_processed = pickle.load(f)
            else:
                self.data.doc_processed = self.bdp.process_docs(self.data.doc_texts)
                with open(self.config.doc_processed_path, 'wb') as f:
                    pickle.dump(self.data.doc_processed, f)

        # load train data
        if train:
            self.fileLoader.load_training_data()
            if test_BM25:
                self.bm25.test_training_BM25_accuracy(10)
                return

            # predict answer
            # self.predict_with_bm25_pars_sents(0)
            self.predict_with_bm25_sents(0)

        # load dev data
        if dev:
            self.fileLoader.load_dev_data()
            if test_BM25:
                self.bm25.test_BM25_par_on_dev()
                return

            # predict answer
            self.predict_with_bm25_pars_sents(1)
            # self.predict_with_bm25_sents(1)

        # load test data
        if test:
            self.fileLoader.load_test_data()

            # predict answer
            # self.predict_with_bm25_pars_sents(2)
            self.predict_with_bm25_sents(2)

    ''' extract wh word from questions
        return wh word if found otherwise return -1
    '''

    def extract_wh_word(self, words):
        for word in words:
            if word.lower() in self.config.WH_words or word.lower() == 'whom':
                return word
        return -1

    '''  identify question types based on rules
    return ranked ner tags and classified type
    '''

    def identify_question_type(self, wh, q_words):
        lower = self.bdp.lower_tokens(q_words)
        # open_words = self.dataProcessor.remove_stop_words(lower)
        raw_q_sent = ' '.join(lower)
        if 'rank' in raw_q_sent:
            return ['ORDINAL'], 'rank'
        elif 'average' in raw_q_sent:
            return ['NUMBER', 'MONEY'], 'average'
        elif wh == 'what':
            if 'what century' in raw_q_sent:
                return ['ORDINAL'], 'century'
            if 'what language' in raw_q_sent:
                return ['NATIONALITY'], 'language'
            if 'nationality' in raw_q_sent:
                return ['NATIONALITY', 'PERSON'], 'nationality'
            if 'length' in raw_q_sent:
                return ['NUMBER'], 'length'
            if 'what year' in raw_q_sent:
                return ['DATE'], 'year'
            if 'what date' in raw_q_sent:
                return ['DATE'], 'date'
            if 'what percent' in raw_q_sent or 'what percentage' in raw_q_sent:
                return ['PERCENT'], 'percentage'
            if 'number' in raw_q_sent:
                return ['NUMBER'], 'number'
            if 'in what place' in raw_q_sent:
                return ['ORDINAL'], 'order'
            if 'what country' in raw_q_sent:
                return ['COUNTRY'], 'country'
            if 'what city' in raw_q_sent:
                return ['STATE_OR_PROVINCE', 'CITY', 'LOCATION'], 'city'
            if 'what region' in raw_q_sent:
                return ['NATIONALITY'], 'region'
            if 'location' in raw_q_sent:
                return ['LOCATION'], 'place'
            if 'population' in raw_q_sent:
                return ['PERCENT', 'NUMBER'], 'population'
            if 'fraction' in raw_q_sent:
                return ['ORDINAL'], 'fraction'
            if 'what age' in raw_q_sent:
                return ['NUMBER'], 'age'
            if 'what decade' in raw_q_sent:
                return ['DATE'], 'decade'
            if 'temperature' in raw_q_sent:
                return ['NUMBER'], 'temperature'
            if 'abundance' in raw_q_sent:
                return ['PERCENT'], 'abundance'
            if 'capacity' in raw_q_sent:
                return ['NUMBER'], 'capacity'
            else:
                return ['O', 'OTHER', 'PERSON', 'LOCATION', 'NUMBER'], 'else'
        elif wh == 'when':
            return ['DATE', 'TIME', 'NUMBER'], 'time'
        elif wh == 'who' or wh == 'whom':
            return ['PERSON', 'ORGANIZATION', 'OTHER'], 'person'
        elif wh == 'where':
            if 'headquarter' in raw_q_sent or 'capital' in raw_q_sent:
                return ['CITY'], 'headquarter'
            return ['LOCATION', 'ORDINAL', 'OTHER'], 'location'
        elif wh == 'how':
            if 'old' in raw_q_sent or 'large' in raw_q_sent:
                return ['NUMBER'], 'number'
            elif 'how long' in raw_q_sent:
                return ['DURATION', 'NUMBER'], 'length'
            elif 'how far' in raw_q_sent or 'how fast' in raw_q_sent:
                return ['NUMBER', 'TIME', 'PERCENT'], 'length'
            elif 'how many' in raw_q_sent:
                return ['NUMBER'], 'times'
            elif 'how much money' in raw_q_sent:
                return ['MONEY', 'PERCENT', 'NUMBER'], 'money'
            elif 'how much' in raw_q_sent:
                return ['MONEY', 'PERCENT', 'NUMBER'], 'money'
            elif 'how tall' in raw_q_sent:
                return ['number'], 'tall'
            else:
                return ['O', 'NUMBER', 'LOCATION', 'PERSON', 'ORGANIZATION'], 'else'
        elif wh == 'which':
            if 'which language' in raw_q_sent:
                return ['NATIONALITY'], 'language'
            if 'which year' in raw_q_sent:
                return ['TIME', 'NUMBER'], 'year'
            if 'which country' in raw_q_sent:
                return ['COUNTRY'], 'country'
            if 'which city' in raw_q_sent:
                return ['CITY'], 'country'
            if 'place' in raw_q_sent or 'location' in raw_q_sent or 'site' in raw_q_sent:
                return ['LOCATION', 'ORGANIZATION', 'OTHER', 'PERSON'], 'place'
            if 'person' in raw_q_sent:
                return ['PERSON', 'ORGANIZATION', 'OTHER', 'LOCATION'], 'person'
            else:
                return ['O', 'OTHER', 'LOCATION', 'PERSON', 'NUMBER'], 'else'
        elif 'activism' in raw_q_sent or 'philosophy' in raw_q_sent or 'ideology' in raw_q_sent:
            return ['IDEOLOGY'], 'ideology'
        elif 'war' in raw_q_sent or 'blood' in raw_q_sent:
            return ['CAUSE_OF_DEATH'], 'war'
        else:
            return ['O', 'OTHER', 'LOCATION', 'PERSON', 'NUMBER'], 'else'

    def pred_answer_type(self, entities, qs_processed,
                         possible_qs_type_rank, qs_type):

        # doubt!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # remove doc entities appeared in the question
        not_in_qs_entities = self.remove_entity_in_qs(qs_processed, entities)
        # get entity dict by ner tags
        ner_type_to_entities_dict = self.get_ner_type_to_entities_dict(not_in_qs_entities)
        # get lemmatized eneities strings
        grouped_entities_strings_lemmatized = [self.bdp.lemmatize_entity_name(tup[0]) for tup in
                                               entities]
        if not possible_qs_type_rank:
            return -1, []

        # iterate possible answer ner tag in likelihood order
        for type in possible_qs_type_rank:
            if len(ner_type_to_entities_dict[type]) != 0:
                assert ner_type_to_entities_dict[type]
                # get all this kind tag entities
                one_type_entities = ner_type_to_entities_dict[type]
                one_type_grouped_entities_strings = [x[0] for x in one_type_entities]
                # if the type is 'O', only get 'NN' pos tag entities
                if type == 'O':
                    one_type_grouped_entities_strings = [x[0] for x in
                                                         pos_tag(one_type_grouped_entities_strings)
                                                         if 'NN' in x[1]]
                # distance between candidate answer entity to all question tokens
                # in the text
                distance = []
                # candidate answer entity position
                possible_entity_pos = []
                # position of question token in text
                qs_token_in_entity_pos = []

                # position of question token in text
                for qs_token in qs_processed:
                    if qs_token in grouped_entities_strings_lemmatized:
                        for i in range(len(grouped_entities_strings_lemmatized)):
                            entity_string = grouped_entities_strings_lemmatized[i]
                            if entity_string.lower() in qs_token:
                                qs_token_in_entity_pos.append(i)
                # calculate distance between candidate answer entity to all question tokens
                # in the text
                for entity in one_type_grouped_entities_strings:
                    for j in range(len(entities)):
                        word = entities[j][0]
                        if word.lower() == entity.lower():
                            sum_dist = 0
                            for k in qs_token_in_entity_pos:
                                sum_dist += (abs(j - k))
                            distance.append(sum_dist)
                            possible_entity_pos.append(j)
                            break
                assert len(possible_entity_pos) == len(distance)

                if distance:
                    # choose the entities with the minimum distance to the question tokens
                    min_idx = np.argmin(distance)
                    best_entity = one_type_grouped_entities_strings[min_idx]
                    # if the question type is year, choose a 4-length-number entity with
                    # minimum distance
                    if qs_type == 'year':
                        while len(best_entity) != 4 and len(distance) > 1:
                            distance.remove(distance[min_idx])
                            min_idx = np.argmin(distance)
                            best_entity = one_type_grouped_entities_strings[min_idx]
                        return best_entity.lower(), one_type_grouped_entities_strings
                    return best_entity.lower(), one_type_grouped_entities_strings
        return -1, []

    '''  combine neighbouring same kind of ner tag together except 'O'
    '''

    def get_combined_entities(self, ner_par):
        entities = []
        ner_group = []
        prev_ner_type = ''
        for ner_tuple in ner_par:
            current_ner_type = ner_tuple[1]
            if not prev_ner_type:
                ner_group.append(ner_tuple)
                prev_ner_type = current_ner_type
            else:
                if current_ner_type == prev_ner_type:
                    ner_group.append(ner_tuple)
                else:
                    entities += self.process_combined_entity(ner_group, prev_ner_type)
                    ner_group = [ner_tuple]
                    prev_ner_type = current_ner_type
        entities += self.process_combined_entity(ner_group, prev_ner_type)
        return entities

    '''  combine neighbouring same kind of ner tag together except 'O'
    '''

    def process_combined_entity(self, ner_group, ner_type):
        entities = []
        if ner_type == 'O':
            for ner_tuple in ner_group:
                entities.append(ner_tuple)
        else:
            entity = [ner_tuple[0] for ner_tuple in ner_group]
            entity_item = [' '.join(entity), ner_type]
            entities.append(entity_item)
        return entities

    def remove_entity_in_qs(self, qs, entities):
        valid_entities = []
        for entity in entities:
            entity_words = entity[0].split()
            for word in entity_words:
                word = word.lower()
                if self.bdp.lemmatize(word) not in qs:
                    valid_entities.append(entity)
                    break
        return valid_entities

    def get_ner_type_to_entities_dict(self, entities):
        ner_type_to_entities_dict = defaultdict(list)
        for entity in entities:
            ner_type = entity[1]
            ner_type_to_entities_dict[ner_type].append(entity)
        return ner_type_to_entities_dict

    '''  preprocess questions and return tokens
    '''

    def preprocess_questions(self, raw_qs):
        # remove special characters
        raw_split = word_tokenize(raw_qs.replace("\u200b", '').replace("\u2014", ''))
        # remove pure punctuation tokens
        remove_pure_punc = [token for token in raw_split if not self.bdp.is_pure_puncs(token)]
        # remove punctuations within a token
        remove_punc_in_words = [self.bdp.remove_punc_in_token(token) for token in remove_pure_punc]
        lemmatized = self.bdp.lemmatize_tokens(remove_punc_in_words)
        return lemmatized

    ''' input string of text
     return processed combined ner tags
     '''

    def ner_process(self, text):
        # get ner tags
        ner_par = self.bdp.nlp.ner(text)
        original_ner = []
        for tup in ner_par:
            tup = list(tup)
            # change tags in 'OTHER' set to 'O'
            if tup[1] in self.other:
                tup[1] = 'O'
            # remove certain kind of punctuations ina token
            tup[0] = self.bdp.remove_punc_in_token_for_rule(tup[0])
            original_ner.append(tup)
        # combine neighbouring same kind of ner tag together except 'O'
        original_ner = self.get_combined_entities(original_ner)
        # remove pure punctuation tokens
        original_ner = [item for item in original_ner if not self.bdp.is_pure_puncs(item[0])]
        # remove stop word tokens
        original_ner = [item for item in original_ner if
                        item[0].lower() not in stopwords.words("english")]
        return original_ner

    '''  predict answers by using bm25 finding answer sentence
    '''

    def predict_with_bm25_sents(self, type):
        # count correctly predicted questions
        correct = 0
        # count correctly predicted paragraphs
        correct_id = 0
        # save already processed doc entities for reuse to improve performance
        doc_entity_temp = {}
        # save already separated sentences of docs
        doc_text_temp = {}
        doc_all = self.data.doc_texts
        qs_all = []
        doc_id_all = []
        answer_all = []
        answer_par_id_all = []
        if type == 0:  # train
            qs_all = self.data.train_questions
            doc_id_all = self.data.train_doc_ids
            answer_all = self.data.train_answers
            answer_par_id_all = self.data.train_answer_par_ids
            fname = self.config.predict_train_output_path
        elif type == 1:  # dev
            qs_all = self.data.dev_questions
            doc_id_all = self.data.dev_doc_ids
            answer_all = self.data.dev_answers
            answer_par_id_all = self.data.dev_answer_par_ids
            fname = self.config.predict_dev_output_path
        else:  # test
            qs_all = self.data.test_questions
            doc_id_all = self.data.test_doc_ids
            test_ids = self.data.test_ids
            fname = self.config.predict_test_output_path
        total = int(len(qs_all))
        with open(fname, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            if type == 0 or type == 1:
                csv_writer.writerow(
                    ['W/R', 'query', 'predicted_id_R/W', 'actual_id', 'predicted_answer',
                     'actual_answer',
                     'predicted_answer_type', 'predicated_candidates'])
            else:
                csv_writer.writerow(['id', 'answer'])
            for i in range(total):
                # for i in range(20):
                print(i, " / ", total)

                qs = qs_all[i]
                doc_id = doc_id_all[i]
                doc = doc_all[doc_id]
                if type == 0 or type == 1:
                    answer = answer_all[i]
                    answer_par_id = answer_par_id_all[i]
                # preprocess questions and return tokens
                qs_processed = self.preprocess_questions(qs)
                doc_processed = self.data.doc_processed[doc_id]

                # get doc entities saving in format of
                # [sentence...[entitiy...]]
                doc_entities = []
                # get doc sentences saving in format of
                # [sentence1, sentence2..]
                doc_sents_text = []
                if doc_id in doc_entity_temp:
                    doc_entities = doc_entity_temp[doc_id]
                    doc_sents_text = doc_text_temp[doc_id]
                else:
                    # iterate paragraphs of that doc
                    for par in doc:
                        sents_text = sent_tokenize(par)
                        doc_sents_text += sents_text
                        # iterate sentences of the paragraph
                        for sent in sents_text:
                            doc_entities.append(self.ner_process(sent))
                    doc_entity_temp[doc_id] = doc_entities
                    doc_text_temp[doc_id] = doc_sents_text

                # extract wh word
                wh = self.extract_wh_word(qs_processed)
                # identify answer ner tag ranks and question type
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, qs_processed)
                pred_answer = 'unknown'
                # predicted answer
                predict_answer = 'unknown'
                # predicated answer ner tags
                answer_types = []
                # predicated paragraph id
                pred_par_id = -1
                # finded candidate answers
                candidate_answers = ''
                if possible_qs_type_rank:
                    self.bm25.k1 = 1.2
                    self.bm25.b = 0.75
                    # tokenize sentences
                    sent_tokens = self.bdp.preprocess_doc(doc_sents_text)
                    # rank sentences based on bm25 scores
                    bm25_sent_tokens_rank = self.bm25.sort_by_bm25_score(qs_processed, sent_tokens)
                    bm25_sent_tokens_rank_ids = [x[0] for x in bm25_sent_tokens_rank]
                    # iterate sentences from higher bm25 score to lower
                    for sent_id in bm25_sent_tokens_rank_ids:
                        # find a answer and candidate answers
                        temp_answer, temp_candidate_answers = self.pred_answer_type(
                            doc_entities[sent_id],
                            qs_processed,
                            possible_qs_type_rank,
                            qs_type)
                        # if find a answer, break out
                        if temp_answer != -1:
                            pred_answer = temp_answer
                            answer_types = possible_qs_type_rank
                            pred_sent_id = sent_id
                            candidate_answers = '; '.join(temp_candidate_answers)
                            break
                if type == 0 or type == 1:
                    if pred_sent_id != -1:
                        for par_id in range(len(doc)):
                            if doc_sents_text[pred_sent_id] in doc[par_id]:
                                pred_par_id = par_id
                                break
                    candidate_answers = '; '.join(temp_candidate_answers)

                    types = ' '.join(answer_types)
                    if pred_par_id == answer_par_id:
                        correct_id += 1
                    if answer == pred_answer:
                        csv_writer.writerow(
                            ["##right##", qs, pred_par_id, answer_par_id, pred_answer, answer,
                             types,
                             candidate_answers])
                        correct += 1
                    else:
                        csv_writer.writerow(
                            ["##wrong##", qs, pred_par_id, answer_par_id, pred_answer, answer,
                             types,
                             candidate_answers])
                    print(answer, " ; ", pred_answer)
                    # print "correct :", correct
                else:
                    csv_writer.writerow([test_ids[i], pred_answer])
            if type == 0 or type == 1:
                csv_writer.writerow([str(correct), str(correct * 100.0 / total)])
                csv_writer.writerow([str(correct_id), str(correct_id * 100.0 / total)])
                csv_writer.writerow([str(total)])
                print(correct * 100.0 / total)
                print(correct_id * 100.0 / total)
                print("best : 19.470455279302552")

    '''  predict answers by using bm25 firstly finding answer paragraph
     then within that paragraph finding answer sentence
     '''

    def predict_with_bm25_pars_sents(self, type):
        # count correctly predicted questions
        correct = 0
        # count correctly predicted paragraphs
        correct_id = 0
        # save already processed doc entities for reuse to improve performance
        doc_entity_temp = {}
        # save already separated doc sentences
        doc_text_temp = {}
        doc_all = self.data.doc_texts
        qs_all = []
        doc_id_all = []
        answer_all = []
        answer_par_id_all = []
        if type == 0:  # train
            qs_all = self.data.train_questions
            doc_id_all = self.data.train_doc_ids
            answer_all = self.data.train_answers
            answer_par_id_all = self.data.train_answer_par_ids
            fname = self.config.predict_train_output_path
        elif type == 1:  # dev
            qs_all = self.data.dev_questions
            doc_id_all = self.data.dev_doc_ids
            answer_all = self.data.dev_answers
            answer_par_id_all = self.data.dev_answer_par_ids
            fname = self.config.predict_dev_output_path
        else:  # test
            qs_all = self.data.test_questions
            doc_id_all = self.data.test_doc_ids
            test_ids = self.data.test_ids
            fname = self.config.predict_test_output_path
        total = int(len(qs_all))
        with open(fname, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            if type == 0 or type == 1:
                csv_writer.writerow(
                    ['W/R', 'query', 'predicted_id_R/W', 'actual_id', 'predicted_answer',
                     'actual_answer',
                     'predicted_answer_type', 'predicated_candidates'])
            else:
                csv_writer.writerow(['id', 'answer'])
            for i in range(total):
                # for i in range(20):
                print(i, " / ", total)

                qs = qs_all[i]
                doc_id = doc_id_all[i]
                doc = doc_all[doc_id]
                if type == 0 or type == 1:
                    answer = answer_all[i]
                    answer_par_id = answer_par_id_all[i]
                # preprocess questions and return tokens
                qs_processed = self.preprocess_questions(qs)
                doc_processed = self.data.doc_processed[doc_id]

                # get doc entities saving in format of
                # [paragraph...[sentence...[entitiy...]]]
                doc_entities = []
                if doc_id in doc_entity_temp:
                    doc_entities = doc_entity_temp[doc_id]
                else:
                    # iterate paragraphs of that doc
                    for par in doc:
                        par_entities = []
                        sent_text = sent_tokenize(par)
                        # iterate sentences of the paragraph
                        for sent in sent_text:
                            par_entities.append(self.ner_process(sent))
                        doc_entities.append(par_entities)
                    doc_entity_temp[doc_id] = doc_entities

                # extract wh word
                wh = self.extract_wh_word(qs_processed)
                # identify answer ner tag ranks and question type
                possible_qs_type_rank, qs_type = self.identify_question_type(wh, qs_processed)
                # predicted answer
                predict_answer = 'unknown'
                # predicated answer ner tags
                answer_types = []
                # predicated paragraph id
                pred_par_id = -1
                # finded candidate answers
                candidate_answers = ''
                if possible_qs_type_rank:
                    self.bm25.k1 = 1.2
                    self.bm25.b = 0.75
                    # rank paragraphs based on bm25 scores
                    bm25_rank = self.bm25.sort_by_bm25_score(qs_processed, doc_processed)
                    bm25_rank_par_ids = [x[0] for x in bm25_rank]
                    # iterate paragraphs from higher bm25 score to lower
                    for par_id in bm25_rank_par_ids:
                        par_text = doc[par_id]
                        sents_text = sent_tokenize(par_text)
                        # tokenize sentences of the paragraph
                        sent_tokens = self.bdp.preprocess_doc(sents_text)
                        # rank sentences based on bm25 scores
                        bm25_sent_tokens_rank = self.bm25.sort_by_bm25_score(qs_processed,
                                                                             sent_tokens)
                        bm25_sent_tokens_rank_ids = [x[0] for x in bm25_sent_tokens_rank]
                        # iterate sentences from higher bm25 score to lower
                        for sent_id in bm25_sent_tokens_rank_ids:
                            # find a answer and candidate answers
                            temp_answer, temp_candidate_answers = self.pred_answer_type(
                                doc_entities[
                                    par_id][
                                    sent_id],
                                qs_processed,
                                possible_qs_type_rank,
                                qs_type)
                            # if find a answer, break out
                            if temp_answer != -1:
                                predict_answer = temp_answer
                                answer_types = possible_qs_type_rank
                                pred_par_id = par_id
                                candidate_answers = '; '.join(temp_candidate_answers)
                                break
                        # if find a answer, break out
                        if temp_answer != -1:
                            break

                if type == 0 or type == 1:
                    types = ' '.join(answer_types)
                    if pred_par_id == int(answer_par_id):
                        correct_id += 1
                    if predict_answer == answer:
                        csv_writer.writerow(
                            ["##right##", qs, pred_par_id, answer_par_id, predict_answer, answer,
                             types,
                             candidate_answers])
                        correct += 1
                    else:
                        csv_writer.writerow(
                            ["##wrong##", qs, pred_par_id, answer_par_id, predict_answer, answer,
                             types,
                             candidate_answers])
                    print(predict_answer, " ; ", answer)
                    # print "correct :", correct
                else:
                    csv_writer.writerow([test_ids[i], predict_answer])

            if type == 0 or type == 1:
                csv_writer.writerow([str(correct), str(correct * 100.0 / total)])
                csv_writer.writerow([str(correct_id), str(correct_id * 100.0 / total)])
                csv_writer.writerow([str(total)])
                print(correct * 100.0 / total)
                print(correct_id * 100.0 / total)
                print("best : 19.470455279302552")


if __name__ == '__main__':
    rule_based_QA = RuleBasedQA()
