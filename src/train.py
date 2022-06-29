from preprocess import *
from text_cnn import *
from path import *

if __name__ == '__main__':
    # read and initialize the data
    fileURL = root_path + data_path + "test_data.xlsx"
    data = read_excel(fileURL, "关联关系")
    corpus_1 = extract_label(data, "机构名称")
    corpus_2 = extract_label(data, "表中文名")
    corpus_3 = extract_label(data, "字段中文（可忽略）")
    labels = extract_label(data, "数据元")

    # segmentation and remove stop words
    corpus_words_1 = segment(corpus_1)
    corpus_words_2 = segment(corpus_2)
    corpus_words_3 = segment(corpus_3)

    all_corpus_words = corpus_words_3 # list(map(lambda a, b, c: a + b + c, corpus_words_1, corpus_words_2, corpus_words_3))

    print(all_corpus_words)

    # compute Word2vec vectors.
    w2v_model = train_word2vec_model(all_corpus_words, vector_size=100)

    # compute x, y
    x, embedding_matrix = construct_x(w2v_model, all_corpus_words, 4)
    y, y_dict = construct_y(labels)

    # 训练，评价，存储
    model, x_test, y_test = train_text_cnn(embedding_matrix, x, y, len(y_dict))
    evaluate(model, x_test, y_test)
    model.save(output_path + "text_cnn.model")
