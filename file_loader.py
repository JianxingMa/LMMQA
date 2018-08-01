import codecs
import json


'''  File loader class for loading doc, train, dev and test data from json files and save
    in data class
'''
class FileLoader:
    def __init__(self, config, data):
        self.config = config
        self.data = data

    def load_training_data(self):
        f = codecs.open(self.config.train_file_name, 'rb', encoding='utf-8')
        print(self.config.train_file_name + ' done')
        data = json.load(f)
        for item in data:
            self.data.train_questions.append(item['question'])
            self.data.train_doc_ids.append(item['docid'])
            self.data.train_answers.append(item['text'])
            self.data.train_answer_par_ids.append(item['answer_paragraph'])

    def load_doc(self):
        f = codecs.open(self.config.doc_file_name, 'rb', encoding='utf-8')
        print(self.config.doc_file_name + ' done')
        data = json.load(f)
        for item in data:
            self.data.doc_ids.append(item['docid'])
            self.data.doc_texts.append(item['text'])

    def load_dev_data(self):
        f = codecs.open(self.config.dev_file_name, 'rb', encoding='utf-8')
        print(self.config.dev_file_name + ' done')
        data = json.load(f)
        for item in data:
            self.data.dev_questions.append(item['question'])
            self.data.dev_doc_ids.append(item['docid'])
            self.data.dev_answers.append(item['text'])
            self.data.dev_answer_par_ids.append(item['answer_paragraph'])

    def load_test_data(self):
        f = codecs.open(self.config.test_file_name, 'rb', encoding='utf-8')
        print(self.config.test_file_name + ' done')
        data = json.load(f)
        for item in data:
            self.data.test_questions.append(item['question'])
            self.data.test_doc_ids.append(item['docid'])
            self.data.test_ids.append(item['id'])
