def multiclass_accuracy(prediction, ground_truth):
    """
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    # TODO: Implement computing accuracy
    true_labels = 0
    prediction_size = prediction.shape[0]
    for i in range(prediction_size):
        if prediction[i] == ground_truth[i]:
            true_labels +=1
    accuracy = true_labels / prediction_size
    return accuracy
