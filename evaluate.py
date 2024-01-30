
# step 1
# Go to Ubuntu (or any POSIX system that has docker/gzip/pwd)
# Navigate to the correct dir (e.g. cd /mnt/c/Users/hinar/Downloads/JustRAIGS-challenge-pack-main-final/example-evaluation)

# Run these commands:
# docker build --tag evaluation-method-development .

# This one is optional:
# docker run --rm --network none -v $(pwd)/intermediate:/opt/app/intermediate -v $(pwd)/test/input:/input -v $(pwd)/test/output:/output evaluation-method-development

# docker save evaluation-method-development | gzip -c > evaluation-method-development.tar.gz

# Upload the created .tar(.gz) as evaluation method under admin

import csv
import json
import os.path
from statistics import mean
from pathlib import Path
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc

INPUT_DIRECTORY = "/input"  # You can change this to "test_submission" to run outside Docker, but remember to change it back before building your container
OUTPUT_DIRECTORY = "/output"  # You can also change this to a local directory to run outside Docker, but remember to change it back
INTERMEDIATE_DIRECTORY = Path("/opt/app/intermediate/")

EYE_ID_HEADER = "EYE ID"
TASK_1_HEADERS = ["referable glaucoma", "likelihood referable glaucoma"]
TASK_2_LABEL_HEADERS = [
    "appearance neuroretinal rim superiorly",
    "appearance neuroretinal rim inferiorly",
    "retinal nerve fiber layer defect superiorly",
    "retinal nerve fiber layer defect inferiorly",
    "baring of the circumlinear vessel superiorly",
    "baring of the circumlinear vessel inferiorly",
    "nasalization of the vessel trunk",
    "disc hemorrhages",
    "laminar dots",
    "large cup", 
]

TASK_2_LOOKUP = {k: index for index, k in enumerate(TASK_2_LABEL_HEADERS)}


def main():
    print(os.path.isdir(INTERMEDIATE_DIRECTORY))
    print_inputs()

    predictions = read_predictions()
    stack_info = read_stack_info()

    task_1_results = []
    task_2_results = []

    for job in predictions:
        # We now iterate over each algorithm job for this submission
        # Note that the jobs are not in any order!
        # We work that out from predictions.json

        # This corresponds to one archive item in the archive
        cfp_stack_filename = get_image_name(
            values=job["inputs"], slug="stacked-color-fundus-images"
        )
        # Parse one of the filename to get the batch ID
        batch_id = cfp_stack_filename.split(".")[0]

        pprint(f"Processing batch {batch_id}")

        # Now we can get the locations of users inference output for this archive item
        is_referable_glaucoma_stacked_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="multiple-referable-glaucoma-binary-decisions",
        )
        is_referable_glaucoma_likelihood_stacked_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="multiple-referable-glaucoma-likelihoods",
        )

        referable_glaucomatous_features_stacked_location = get_file_location(
            job_pk=job["pk"],
            values=job["outputs"],
            slug="stacked-referable-glaucomatous-features",
        )

        # Now we load those files to get their content
        is_referable_glaucoma_stacked = load_json_file(
            location=is_referable_glaucoma_stacked_location
        )
        is_referable_glaucoma_likelihood_stacked = load_json_file(
            location=is_referable_glaucoma_likelihood_stacked_location
        )
        referable_glaucomatous_features_stacked = load_json_file(
            location=referable_glaucomatous_features_stacked_location
        )

        current_stack_info = stack_info[batch_id]
        for item in current_stack_info:
            index = item["stack_index"]

            # The follow three data points should be sufficient to map to your groundthruth
            eye_id = os.path.splitext(item["image"])[0]
            referable_glaucoma = is_referable_glaucoma_stacked[index]
            referable_glaucoma_likelihood = is_referable_glaucoma_likelihood_stacked[
                index
            ]

            task_1_results.append(
                [eye_id, optional_bool_to_int(referable_glaucoma), referable_glaucoma_likelihood]
            )

            # Ensure ordering, because JSON has unordered dicts
            label_results = dict(
                sorted(
                    referable_glaucomatous_features_stacked[index].items(),
                    key=lambda i: TASK_2_LOOKUP[i[0]],
                )
            )
            label_result_ordered = list(optional_bool_to_int(v) for v in label_results.values())
            task_2_results.append([eye_id, *label_result_ordered])

    # Sort on Eye ID
    task_1_results.sort(key=lambda i: i[0])
    task_2_results.sort(key=lambda i: i[0])

    INTERMEDIATE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # Save task 1
    with open(INTERMEDIATE_DIRECTORY / "task_1.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow([EYE_ID_HEADER, *TASK_1_HEADERS])

        # Writing the data rows
        writer.writerows(task_1_results)

    # Save task 2
    with open(INTERMEDIATE_DIRECTORY / "task_2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)

        # Writing the header
        writer.writerow([EYE_ID_HEADER, *TASK_2_LABEL_HEADERS])

        # Writing the data rows
        writer.writerows(task_2_results)


    metrics = process(task_1_file = INTERMEDIATE_DIRECTORY / "task_1.csv", task_2_file=INTERMEDIATE_DIRECTORY / "task_2.csv")


    write_metrics(metrics=metrics)

    return 0

def optional_bool_to_int(optional_bool):
    if optional_bool is None:
        return None
    else:
        return int(optional_bool)

def process(task_1_file, task_2_file):
    gtLabelsDF = pd.read_csv('ground_truth/JustRAIGS_5%TestDataset_labels.csv')
    task1DF = pd.read_csv(task_1_file)
    task2DF = pd.read_csv(task_2_file)
    # Mappings
    for index, row in gtLabelsDF.iterrows():
        task1DF.loc[task1DF['EYE ID'] == row["EYE ID"], 'Final Label'] = row["Final Label"]
        task2DF.loc[task2DF['EYE ID'] == row["EYE ID"], ['Final Label','Label G3', 'G1 ANRS', 'G1 ANRI', 'G1 RNFLDS', 'G1 RNFLDI', 'G1 BCLVS',
        'G1 BCLVI', 'G1 NVT', 'G1 DH', 'G1 LD', 'G1 LC', 'G2 ANRS', 'G2 ANRI',
        'G2 RNFLDS', 'G2 RNFLDI', 'G2 BCLVS', 'G2 BCLVI', 'G2 NVT', 'G2 DH',
        'G2 LD', 'G2 LC', 'G3 ANRS', 'G3 ANRI', 'G3 RNFLDS', 'G3 RNFLDI',
        'G3 BCLVS', 'G3 BCLVI', 'G3 NVT', 'G3 DH', 'G3 LD', 'G3 LC']] = [row['Final Label'],
                                                                         row['Label G3'], row['G1 ANRS'], row['G1 ANRI'], row['G1 RNFLDS'],
                                                                         row['G1 RNFLDI'], row['G1 BCLVS'],
                                                                         row['G1 BCLVI'], row['G1 NVT'], row['G1 DH'], row['G1 LD'], row['G1 LC'],
                                                                         row['G2 ANRS'], row['G2 ANRI'],
                                                                         row['G2 RNFLDS'], row['G2 RNFLDI'], row['G2 BCLVS'],
                                                                         row['G2 BCLVI'], row['G2 NVT'], row['G2 DH'],
                                                                         row['G2 LD'], row['G2 LC'], row['G3 ANRS'], row['G3 ANRI'],
                                                                         row['G3 RNFLDS'], row['G3 RNFLDI'],
                                                                         row['G3 BCLVS'], row['G3 BCLVI'], row['G3 NVT'], row['G3 DH'],
                                                                         row['G3 LD'], row['G3 LC']]
    
    # task1DF.to_csv('task_1.csv',index=False)
    # task2DF.to_csv('task_2.csv',index=False)
    
    # Calculating Senisitivity
    mask = task1DF['Final Label'] == 'U'
    task1DF = task1DF[~mask]
    task1DF['Final Label'] = task1DF['Final Label'].map({'RG': 1, 'NRG': 0})
    task1DF['likelihood referable glaucoma'] = task1DF['likelihood referable glaucoma'].fillna(0)
    sensitivity = get_sensitivity(task1DF['likelihood referable glaucoma'], task1DF['Final Label'])

    # Calculating Hamming loss
    count = 0
    sum = 0
    for index, row in task2DF.iterrows():
        #print("row",row)
        pred_labels = task2DF.loc[index, TASK_2_LABEL_HEADERS].to_numpy().tolist()
        #print("pred_labels", pred_labels)
        if row['Final Label'] == 'RG':
            if row['Label G3'] == 'RG':
                count += 1

                #print("Label", row['Label G3'])

                # Putting the column names for labels of ten additional features
                label_columns = ['G3 ANRS', 'G3 ANRI', 'G3 RNFLDS', 'G3 RNFLDS', 'G3 BCLVS',
                                'G3 BCLVI', 'G3 NVT', 'G3 DH', 'G3 LD', 'G3 LC']
                # Extract the values from the DataFrame's label columns and convert them to a list of lists
                true_labels = row[label_columns].values.tolist()
                #print("true_labels", true_labels)
                #print("pred_labels", pred_labels)
                # Assuming true_labels and pred_labels are lists or arrays, index them at 70
                loss = hamming_loss(true_labels, pred_labels)  # Note the brackets to make them iterable
                sum += loss
                print("Average Hamming Loss at index :", loss)
            else:
                count += 1
                # print(row)
                G1_label_columns = ['G1 ANRS', 'G1 ANRI', 'G1 RNFLDS', 'G1 RNFLDS', 'G1 BCLVS',
                                    'G1 BCLVI', 'G1 NVT', 'G1 DH', 'G1 LD', 'G1 LC']
                G1_labels = row[G1_label_columns].values.tolist()
                #print("Grade1 labels", G1_labels)

                G2_label_columns = ['G2 ANRS', 'G2 ANRI', 'G2 RNFLDS', 'G2 RNFLDS', 'G2 BCLVS',
                                    'G2 BCLVI', 'G2 NVT', 'G2 DH', 'G2 LD', 'G2 LC']
                G2_labels = row[G2_label_columns].values.tolist()
                #print("Grade2 labels", G2_labels)

                # find features which have disaggrement
                agreed_features = np.equal(G1_labels, G2_labels)
                print("Agreed features", agreed_features)
                # Select specific columns where disagreed_features is True
                selected_pred_labels = np.array(pred_labels)[agreed_features]
                selected_true_labels = np.array(G2_labels)[agreed_features]

                # Print the selected labels
                print("Selected Predicted Labels:", selected_pred_labels)
                print("Selected G2 Labels:", selected_true_labels)

                # Assuming selected_true_labels and selected_pred_labels are lists or arrays, index them at 70
                loss = hamming_loss(selected_true_labels, selected_pred_labels)
                sum += loss
                print("Average Hamming Loss at index :", loss)

    print(f"Sensitivity = {round(sensitivity,3)}")
    hamming_sum = sum/count
    print(f"Hamming losses Avg = {round(hamming_sum,3)}")
    overall_loss = (1-sensitivity) + hamming_sum
    print(f"Overall loss = {round(overall_loss,3)}")
    return {
        "Total loss": round(overall_loss,3),
        "Sensitivity": round(sensitivity,3),
        "Hamming losses Avg": round(hamming_sum,3),
    }
    # task1DF.to_csv('task_1.csv',index=False)
    # task2DF.to_csv('task_2.csv',index=False)
    # we can directly access the dataframes in the memory as well
    # for task1 get_specificity
    # writing dataframes to csv labels / GT 
    #task1DF.to_csv('task_1.csv',index=False)
    #task2DF.to_csv('task_2.csv',index=False)


def print_inputs():
    # Just for convenience, in the logs you can then see what files you have to work with
    input_files = [str(x) for x in Path(INPUT_DIRECTORY).rglob("*") if x.is_file()]

    print("Input Files:")
    pprint(input_files)
    print("")


def read_predictions():
    # The prediction file tells us the location of the users' predictions
    with open(f"{INPUT_DIRECTORY}/predictions.json") as f:
        return json.loads(f.read())


def hamming_loss(true_labels, predicted_labels):
    """Calculate the Hamming loss for the given true and predicted labels."""
    # Convert to numpy arrays for efficient computation
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate the hamming distance that is basically the total number of mismatches
    Hamming_distance = np.sum(np.not_equal(true_labels, predicted_labels))
    print("Hamming distance", Hamming_distance)
    
    # Calculate the total number of labels
    total_corrected_labels= true_labels.size

    # Compute the Average Hamming loss
    loss = Hamming_distance / total_corrected_labels
    return loss

def get_sensitivity(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :return:
    specificity at 0.95 value of sensitivity
    """
    ## convert NRG -> 0
    ## convert RG -> 1
    # y_pred = FinalLabel
    # y_test = likelihood
    fpr, tpr, thresholds = roc_curve(y_pred, y_test)

    roc_auc = auc(fpr, tpr)

    desired_specificity = 0.95
    # Find the index of the threshold that is closest to the desired specificity
    idx = np.argmax(fpr >= (1 - desired_specificity))
    # Get the corresponding threshold
    threshold_at_desired_specificity = round(thresholds[idx], 4)
    # Get the corresponding TPR (sensitivity)
    sensitivity_at_desired_specificity = round(tpr[idx], 4)

    return sensitivity_at_desired_specificity

def read_stack_info():
    # The mapping tells us which files were in the stack
    with open(f"ground_truth/archive_item_to_content_mapping.json") as f:
        flat_stack_info = json.loads(f.read())

    # Create a lookup dictionary
    stack_info = {}
    for item in flat_stack_info:
        stack_info[item["stack_name"]] = stack_info.get(item["stack_name"], [])
        stack_info[item["stack_name"]].append(item)
    return stack_info


def get_image_name(*, values, slug):
    # This tells us the user-provided name of the input or output image
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["image"]["name"]

    raise RuntimeError(f"Image with interface {slug} not found!")


def get_interface_relative_path(*, values, slug):
    # Gets the location of the interface relative to the input or output
    for value in values:
        if value["interface"]["slug"] == slug:
            return value["interface"]["relative_path"]

    raise RuntimeError(f"Value with interface {slug} not found!")


def get_file_location(*, job_pk, values, slug):
    # Where a job's output file will be located in the evaluation container
    relative_path = get_interface_relative_path(values=values, slug=slug)
    return f"{INPUT_DIRECTORY}/{job_pk}/output/{relative_path}"


def load_json_file(*, location):
    # Reads a json file
    with open(location) as f:
        return json.loads(f.read())


def write_metrics(*, metrics):
    # Write a json document used for ranking results on the leaderboard
    with open(f"{OUTPUT_DIRECTORY}/metrics.json", "w") as f:
        f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
