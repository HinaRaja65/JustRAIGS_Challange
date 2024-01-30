import numpy as np
import pandas as pd

data= pd.read_excel(r'./File1.xlsx')


def get_hamming_loss(true_labels, predicted_labels):
    """Calculate the Hamming loss for the given true and predicted labels."""
    # Convert to numpy arrays for efficient computation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate the hamming distance that is basically the total number of mismatches
    Hamming_distance = np.sum(np.not_equal(true_labels, predicted_labels))
    print("Hamming distance", Hamming_distance)

    # Calculate the total number of labels
    total_corrected_labels = true_labels.size

    # Compute the Average Hamming loss
    loss = Hamming_distance / total_corrected_labels
    return loss