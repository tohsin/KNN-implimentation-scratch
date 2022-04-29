def trainTestSplit(X_data, labels, percentage_split):
    num_samples, _ = labels.shape
    num_train_set = int(percentage_split * num_samples) # 80 percent split
    num_test_set=num_samples - num_train_set
    print("{} percentage split with {} samples and {} training data \
        and {} number of test".format(percentage_split*100, num_samples, num_train_set,  num_test_set))
    train_data = X_data[:num_train_set]
    train_label =labels[:num_train_set]
    test_data = X_data[num_train_set:]
    test_label = labels[num_train_set:]
    return train_data, train_label, test_data,  test_label
