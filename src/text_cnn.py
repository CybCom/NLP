import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import Input, Model
from keras.layers import Embedding, MaxPooling1D, concatenate, Flatten, Dropout, Dense, Conv1D
from keras.utils import to_categorical, pad_sequences

from sklearn import metrics

from path import *


def construct_x(w2v_model, corpus, max_len=100):
    """
    Construct x, the input for CNN.
    :param max_len:
    :param w2v_model:
    :param corpus:
    :return:
    """
    # 获得词表
    vocab_list = list(w2v_model.wv.index_to_key)
    # print(vocab_list)
    # 构建词典,表达词语与id映射关系，空格初始化id=0 -> in case zero padding.
    embedding_id_dict = {" ": 0}
    # 初始化词向量矩阵
    embedding_matrix = np.zeros((len(vocab_list) + 1, w2v_model.vector_size))
    # 填充词典与词向量矩阵
    for i in range(len(vocab_list)):
        word = vocab_list[i]
        embedding_id_dict[word] = i + 1                        # the word index
        embedding_matrix[i + 1] = w2v_model.wv[word]    # the corresponding vector

    def get_index(sentences):
        sequences = []
        for sentence in sentences:
            sequence = []
            for w in sentence:
                sequence.append(embedding_id_dict[w])
            sequences.append(sequence)
        return sequences
    # 获得全部文本的ID序列
    id_seq = get_index(corpus)
    # 定义每条文本最大词数,超过截断,不足填0
    # max_len = 70
    # 将ID转换为词向量矩阵中的词向量,生成特征矩阵
    x = pad_sequences(id_seq, maxlen=max_len, padding="post", truncating="post")
    print(x)
    print("X build completed.")
    return x, embedding_matrix


def construct_y(labels):
    """
    Construct y, the expected output of CNN.
    :param labels:
    :return:
    """
    label_set = list(set(labels))
    y_dict = dict(zip(label_set, range(0, len(label_set))))

    y_data = np.zeros(len(labels))
    for i in range(len(labels)):
        y_data[i] = y_dict[labels[i]]
    print("Y build completed.")
    return y_data, y_dict


def do_shuffle(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def train_text_cnn(embedding_matrix, x, y, num_classes):
    """
    Finally, build, train and save the text CNN.
    :param embedding_matrix:
    :param x:
    :param y:
    :param num_classes: number of classification choices.
    :return: the trained CNN model.
    """
    # shuffle the data
    x, y = do_shuffle(x, y)
    # cut the data to train and test.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    y_train = to_categorical(y_train, num_classes=num_classes)
    vocabulary_size, embedding_vector_dims = embedding_matrix.shape
    print(embedding_matrix.shape)
    print(x_train.shape)

    # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接

    # 输入层
    num_documents, len_document_processed = x_train.shape
    main_input = Input(shape=(len_document_processed,), dtype='float64')  # every input is a text aka document
    # 词嵌入层（使用预训练的词向量）
    embedder = Embedding(vocabulary_size, embedding_vector_dims, input_length=len_document_processed,
                         weights=[embedding_matrix],
                         trainable=True)
    embed = embedder(main_input)


    # 卷积池化层 词窗大小分别为3,4,5
    cnn1 = Conv1D(embedding_vector_dims, 2, padding='same', strides=1, activation='relu')(embed)
    print(cnn1)
    cnn1 = MaxPooling1D(pool_size=4)(cnn1)
    print(cnn1)
    cnn2 = Conv1D(embedding_vector_dims, 3, padding='same', strides=1, activation='relu')(embed)
    cnn2 = MaxPooling1D(pool_size=3)(cnn2)
    print(cnn2)
    cnn3 = Conv1D(embedding_vector_dims, 4, padding='same', strides=1, activation='relu')(embed)
    cnn3 = MaxPooling1D(pool_size=2)(cnn3)
    print(cnn3)

    # 合并三个模型的输出向量
    cnn = concatenate([cnn1, cnn2, cnn3], axis=-2)
    flat = Flatten()(cnn)
    drop = Dropout(0.5)(flat)
    main_output = Dense(num_classes, activation='softmax')(drop)
    print("here!")
    # compile a model.
    model = Model(inputs=main_input, outputs=main_output)
    adam = keras.optimizers.Adam(learning_rate=0.0001, decay=0.000005)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    # train the model.
    model.fit(x_train, y_train, batch_size=100, epochs=150, validation_split=0.20, verbose=2)
    model.save(root_path + output_path + "text_cnn.model")
    return model, x_test, y_test


def evaluate(model, x_test, y_test):
    """
    Value the train result by correct rate.
    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    y_predict = model.predict(x_test)
    if len(y_predict.shape) > 1:
        y_predict = np.argmax(y_predict, axis=1)
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print('accuracy:' + str(round(accuracy, 4)))
