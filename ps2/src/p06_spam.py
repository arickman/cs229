import collections

import numpy as np

import util
import svm

def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # l = []
    # word = ""
    # for c in message:
    #     if (c != " "): word += c
    #     else: 
    #         l.append(word.lower())
    #         word = ""
    # l = np.array(l)
    # return l
    lowercase = message.lower()
    words = lowercase.split(" ")
    return words
    # *** END CODE HERE ***

def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    mydict = {}
    for message in messages:
        seen = []
        words = get_words(message)
        for word in words:
            if (word not in seen):
                if (word in mydict.keys()):
                    mydict[word] += 1
                else: mydict[word] = 1
                seen.append(word)
    #now we have a dictionary where the value tells the number of messages it was in. 
    final_dict = {}
    index = 0
    for key in mydict:
        if (mydict[key] >= 5): 
            final_dict[key] = index
            index +=1
    return final_dict
    # *** END CODE HERE ***

def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    m = len(messages)
    n = len(word_dictionary)
    t_matrix = np.zeros((m,n))
    message_index = -1
    for message in messages:
        words = get_words(message)
        message_index +=1
        for word in words:
            if (word not in word_dictionary.keys()): continue
            t_matrix[message_index, word_dictionary[word]] += 1
    return t_matrix
    # *** END CODE HERE ***

def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    V = np.shape(matrix)[1]
    m = np.shape(matrix)[0]


    one_vec = np.ones((V,1))
    zero_vec = np.ones((V,1))
    denom_0 = V
    denom_1 = V
    for i in range(m):
        if labels[i] == 1: 
            denom_1 += sum(matrix[i,])
            for j in range(V):
                one_vec[j] += matrix[i,j]
        if labels[i] == 0: 
            denom_0 += sum(matrix[i,])
            for j in range(V):
                zero_vec[j] += matrix[i,j] 
    #Now have numerators of the k-vectors
    one_vec /= denom_1
    zero_vec /= denom_0
        
    return one_vec, zero_vec, phi_y(m, labels)

def phi_y(m, labels):
    phi_y = 0
    for i in range(m):
        if labels[i] == 1: phi_y +=1
    phi_y = (m)**(-1)*phi_y


    return phi_y


    # #phi_k_1
    # num = 1
    # denom = V
    # print(V)
    # phi_1_vec = []
    # for k in range(V):
    #     for i in range(m):
    #         # calculate n_i from the message row
    #         n_i = 0
    #         for j in range(V):
    #             n_i += matrix[i, j]
    #         #print("done with j")
    #         #calculate the denominator and numerator
    #         if (labels[i] == 1):
    #             num += matrix[i, k]
    #             denom += n_i
    #     #print("done with i")
    #     #log rule
    #     num = np.log(num)
    #     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     phi_1_vec.append(num/denom)

    # #phi_k_0
    # num = 1
    # denom = V
    # phi_0_vec = []
    # for k in range(V):
    #     for i in range(m):
    #         # calculate n_i from the message row
    #         n_i = 0
    #         for j in range(V):
    #             n_i += matrix[i, j]
    #         #calculate the denominator and numerator
    #         if (labels[i] == 0):
    #             num += matrix[i, k]
    #             denom += n_i
    #     #log rule
    #     num = np.log(num)
    #     #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     phi_0_vec.append(num/denom)


    # #phi_y
    # phi_y = 0
    # for i in range(m):
    #     if labels[i] == 1: phi_y +=1
    # phi_y = (m)**(-1)*phi_y


    # return phi_1_vec, phi_0_vec, phi_y
    #####################################
    # *** END CODE HERE ***

def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: The trained model
    """
    # *** START CODE HERE ***
    phi_1_vec, phi_0_vec, phi_y = model
    m = np.shape(matrix)[0]
    V = np.shape(matrix)[1]
    prob_vec_spam = []
    prob_vec_not_spam = []
    for i in range(m): #looping through each example
        prob_spam = 0
        prob_not_spam = 0
        for j in range(V):
            #loop through phi_vec and mult all values from words in the dictionary
            prob_spam += np.log(phi_1_vec[j]**(matrix[i,j]))
            prob_not_spam += np.log(phi_0_vec[j]**(matrix[i,j]))

        prob_spam += np.log(phi_y)
        prob_not_spam += np.log(1 - phi_y)
        prob_vec_spam.append(prob_spam)
        prob_vec_not_spam.append(prob_not_spam)

    diff_vec = np.array(prob_vec_spam) - np.array(prob_vec_not_spam)
    predictions = []
    for elem in diff_vec:
        if elem >= 0: predictions.append(1)
        else: predictions.append(0)

    return predictions
    # *** END CODE HERE ***

def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.
    
    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    phi_1_vec, phi_0_vec, phi_y = model
    unsorted = []
    for key in dictionary:
        num = phi_1_vec[dictionary[key]]
        denom = phi_0_vec[dictionary[key]]
        unsorted.append([np.log(num/denom), key])


    unsorted.sort(key=lambda x: x[0])

    sorted_list = []
    for i in range(5):
        sorted_list.append(unsorted[-(i+1)])
    
    return_list = []
    for tup in sorted_list:
        elem = tup[1]
        return_list.append(elem)

    return return_list
    # *** END CODE HERE ***

def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        eval_matrix: The word counts for the validation data
        eval_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    max_accuracy = 0
    rad_opt = 0
    for rad in radius_to_consider:
        svm_prediction = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, rad)
        svm_accuracy = np.mean(svm_prediction == val_labels)
        if svm_accuracy > max_accuracy: 
            max_accuracy = svm_accuracy
            rad_opt = rad

    return rad_opt
    # *** END CODE HERE ***

def main():
    train_messages, train_labels = util.load_spam_dataset('../data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('../data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('../data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('./output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('./output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('./output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('./output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('./output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))

if __name__ == "__main__":
    main()