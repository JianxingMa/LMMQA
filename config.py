import time


class Config:
    def __init__(self):
        self.doc_file_name = 'Data/documents.json'
        self.train_file_name = 'Data/training.json'
        self.dev_file_name = 'Data/devel.json'
        self.test_file_name = 'Data/testing.json'

        self.n_filters = 30
        self.max_sent_len = 30
        self.filter_size = 4
        self.word_emb_dim = 300

        # self.word2vec_model_path = 'model/pruned.word2vec.txt'
        self.word2vec_model_path = 'model/GoogleNews-vectors-negative300.bin'
        # self.ner_model_path = 'stanford/stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.ner_model_path = 'stanford/stanford_ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
        self.ner_jar_path = 'stanford/stanford_ner/stanford-ner.jar'
        self.pos_model_path = 'stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'
        self.pos_jar_path = 'stanford/stanford-postagger/stanford-postagger.jar'
        self.parser_model_path = 'stanford/stanford-parser/stanford-parser-3.9.1-models.jar'
        self.parser_jar_path = 'stanford/stanford-parser/stanford-parser.jar'
        self.predict_train_output_path = 'csv/train_result_sents' + time.strftime(
            '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '.csv'
        self.predict_dev_output_path = 'csv/dev_result_sents' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                                                              time.localtime(
                                                                                  time.time())) + '.csv'
        self.predict_test_output_path = 'csv/test_results_sents' + time.strftime(
            '%Y-%m-%d_%H-%M-%S', time.localtime(time.time())) + '.csv'
        self.model_save_path = 'model/answer.model'

        self.doc_processed_path = 'pkl/doc_processed.pkl'
        self.train_qs_processed_path = 'pkl/train_qs_processed.pkl'
        self.dev_qs_processed_path = 'pkl/dev_qs_processed.pkl'
        self.test_qs_processed_path = 'pkl/test_qs_processed.pkl'
        self.dev_rule_pkl = 'pkl/dev_rule_pkl.pkl'
        self.sentence_embedding_pkl = 'pkl/sentence_embedding.pkl'
        self.training_ner_pkl = 'pkl/training_ner.pkl'
        self.dev_ner_pkl = 'pkl/dev_ner.pkl'
        self.train_answer_ner = 'pkl/dev_ner.pkl'

        self.WH_words = ['how', 'what', 'where', 'when', 'who', 'which']
