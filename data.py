# class for saving all raw data loaded from json and preprocessed data

class Data:
    def __init__(self):
        self.doc_ids = list()
        self.doc_texts = list()
        self.doc_processed = list()

        self.train_doc_ids = list()
        self.train_questions = list()
        self.train_answers = list()
        self.train_answer_par_ids = list()
        self.train_qs_processed = list()

        self.dev_doc_ids = list()
        self.dev_questions = list()
        self.dev_answers = list()
        self.dev_answer_par_ids = list()
        self.dev_qs_processed = list()

        self.test_doc_ids = list()
        self.test_questions = list()
        self.test_ids = list()
        self.test_qs_processed = list()
