from preprocess import *
from text_cnn import *
from path import *

if __name__ == '__main__':
    # read and initialize the data
    fileURL = root_path + data_path + "news.xlsx"
    data = read_excel(fileURL, "Sheet2")
    corpus = extract_label(data, "text1")
    labels = extract_label(data, "label")

    # segmentation and remove stop words
    corpus_words = segment_and_clear(corpus)

    # compute Word2vec vectors.
    w2v_model = train_word2vec_model(corpus_words, vector_size=100)

    # compute x, y
    x, embedding_matrix = construct_x(w2v_model, corpus_words, 30)
    y, y_dict = construct_y(labels)

    # 训练，评价，存储
    model, x_test, y_test = train_text_cnn(embedding_matrix, x, y, len(y_dict))
    evaluate(model, x_test, y_test)
    model.save(output_path + "text_cnn.model")
