from collections import defaultdict
import unicodecsv as csv
from math import log


class BM25:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.k1 = 0.5
        self.b = 0.5

    # the function return the paragraph(sentence) in document that matches best to the query
    # based on bm25
    # returning a list ranked by descending order
    # like [(3, 21.44802474790178), (18, 3.1944542200601345), (0, 2.4847094773431535),...]
    def sort_by_bm25_score(self, query, doc):
        # the number of paragraphs that contains a given word
        idf_dict = defaultdict(int)
        # term frequency in each paragraph. like{'word1':{1:5,2:4,3:0},'word2':{1:2..}...}
        tf_par_dict = defaultdict(dict)
        # term frequency in query
        tf_query_dict = defaultdict(int)
        acc_bm25_dict = defaultdict(float)

        # total number of paragraphs in this doc
        pars_count = len(doc)
        # average paragraph length in this doc
        parLength_avg = self.avg_length(doc)
        for j in range(pars_count):
            par = doc[j]
            for term in par:
                if j not in list(tf_par_dict[term].keys()):
                    tf_par_dict[term][j] = 0
                tf_par_dict[term][j] += 1
            for term in set(par):
                idf_dict[term] += 1

        for term in query:
            tf_query_dict[term] += 1

        for i in range(pars_count):
            acc_bm25_score = 0.0
            par_Length = len(doc[i])
            for term in query:
                if term in list(idf_dict.keys()):
                    ft = idf_dict[term]
                else:
                    ft = 0
                if term in list(tf_par_dict.keys()):
                    if i in list(tf_par_dict[term].keys()):
                        fdt = tf_par_dict[term][i]
                    else:
                        fdt = 0
                else:
                    fdt = 0
                fqt = tf_query_dict[term]
                acc_bm25_score += self.cal_bm25(pars_count, ft, fdt, fqt, par_Length, parLength_avg,
                                                self.k1, self.b)
            acc_bm25_dict[i] = acc_bm25_score
        sorted_acc_bm25_dict = sorted(list(acc_bm25_dict.items()), key=lambda item: item[1],
                                      reverse=True)
        return sorted_acc_bm25_dict

    # calculate bm25 scores of a word in a query with a document
    # the bm25 should be accumulated to rank the best document to a query
    # N: number of documents(sentences)
    # ft: number of documents than contain the term
    # fdt: number of a given term in a document
    # fqt: number of a given term in a query
    # Ld: length of the document
    # Ld_avg: average length of the document
    def cal_bm25(self, N, ft, fdt, fqt, Ld, Ld_avg, k1=1.2, b=0.75, k3=0):
        idf_component = log((N - ft + 0.5) / (ft + 0.5))
        doc_tf_component = ((k1 + 1) * fdt) / ((k1 - k1 * b + k1 * b * Ld / Ld_avg) + fdt)
        query_tf_component = ((k3 + 1) * fqt) / (k3 + fqt)
        bm25_score = idf_component * doc_tf_component * query_tf_component
        # print(str(k1), str(b))
        return bm25_score

    def avg_length(self, doc):
        N = len(doc)
        totalLen = 0.0
        for par in doc:
            totalLen += len(par)
        return totalLen / N

    # test bm25 accuracy of finding answer paragraph on training data
    # return accuracy of ranking the answer paragraph in the top max_tolerant_num order
    def test_training_BM25_accuracy(self, max_tolerant_num, fname):
        n_accuary = defaultdict(int)
        total = len(self.data.train_qs_processed)
        for i in range(len(self.data.train_qs_processed)):
            print(i, ' / ', total)
            qs = self.data.train_qs_processed[i]
            doc_id = self.data.train_doc_ids[i]
            answer_par_id = self.data.train_answer_par_ids[i]
            doc = self.data.doc_processed[doc_id]

            ranked_pars = self.sort_by_bm25_score(qs, doc)

            if ranked_pars:
                count_N = -1
                for k, val in ranked_pars:
                    count_N += 1
                    if count_N < max_tolerant_num:
                        if k == answer_par_id:
                            for m in range(max_tolerant_num, count_N, -1):
                                n_accuary[m] += 1
                    else:
                        break
        with open(fname, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['total', str(total)])
            csv_writer.writerow(['N', "correct", 'accuracy'])
            for k, v in list(n_accuary.items()):
                csv_writer.writerow([str(k), str(v), str(1.0 * v / total)])

    # test bm25 accuracy of finding answer paragraph on dev data
    # return accuracy of ranking the answer paragraph in the top max_tolerant_num order
    def test_dev_BM25_accuracy(self, max_tolerant_num):
        n_accuary = defaultdict(int)
        total = int(len(self.data.dev_qs_processed) / 10)
        for i in range(total):
            print(i, ' / ', total)
            qs = self.data.dev_qs_processed[i]
            doc_id = self.data.dev_doc_ids[i]
            answer_par_id = self.data.dev_answer_par_ids[i]
            doc = self.data.doc_processed[doc_id]

            ranked_pars = self.sort_by_bm25_score(qs, doc)
            if ranked_pars:
                count_N = -1
                for k, val in ranked_pars:
                    count_N += 1
                    if count_N < max_tolerant_num:
                        if k == answer_par_id:
                            for m in range(max_tolerant_num, count_N, -1):
                                n_accuary[m] += 1
                    else:
                        break
        return 1.0 * n_accuary[1] / total

    # find best BM25 parameters on dev data
    # try k1 between 0.1 and 2.1, b between 0.1 and 2.1
    def test_BM25_par_on_dev(self):
        # n = 1
        k1 = 0.1
        b = 0.1
        best_accuracy = 0
        best_k1 = 0.1
        best_b = 0.1
        fname = 'bm25_dev_' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                            time.localtime(time.time())) + '.csv'
        with open(fname, 'wb') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["k1", 'b'])
            for i in range(21):
                for j in range(21):
                    print(str(k1), str(b))
                    self.k1 = k1
                    self.b = b
                    acc = self.test_dev_BM25_accuracy(2)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_k1 = k1
                        best_b = b
                    csv_writer.writerow([k1, b, acc])
                    b += 0.1
                k1 += 0.1
                b = 0.1
            csv_writer.writerow(["best"])
            csv_writer.writerow([best_accuracy, best_k1, best_b])
        print(str(best_accuracy), str(best_k1), str(best_b))
        return
