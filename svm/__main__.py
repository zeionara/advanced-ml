import os

import click
import cv2
import numpy as np
from imutils import paths
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


@click.group()
def main():
    pass


@click.command()
@click.option('--train', default='resources/svm/train/', type=str)
@click.option('--test', default='resources/svm/test/', type=str)
def classify_images(train, test):
    print(f'Training on data from {train}')
    print(f'Set up data...')
    image_paths = sorted(list(paths.list_images(train)))
    data = []
    labels = []

    for (i, imagePath) in enumerate(image_paths):
        image = cv2.imread(imagePath, 1)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]
        hist = extract_histogram(image)
        data.append(hist)
        labels.append(label)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    print(f'Label of {image_paths[0]} is {labels[0]}')

    trainData, testData, trainLabels, testLabels = train_test_split(np.array(data), labels, test_size=0.25,
                                                                    random_state=15)
    model = LinearSVC(random_state=15, C=1.35)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    print("Test classification report:")
    print(classification_report(testLabels, predictions, target_names=le.classes_))
    print(f'f1 score = {f1_score(testLabels, predictions, average="macro")}')

    for coefficient_index in (369, 270, 55):
        print(f'{coefficient_index}th coefficient = {model.coef_[0][coefficient_index]}')

    print(f'Testing on data from {test}')
    print(f'Set up data...')
    image_paths = sorted(list(paths.list_images(test)))
    data = []

    for (i, imagePath) in enumerate(image_paths):
        image = cv2.imread(imagePath, 1)
        hist = extract_histogram(image)
        data.append(hist)

    predictions = model.predict(np.array(data))
    for image_index in (46, 36, 7, 34):
        print(f'Prediction for {image_paths[image_index]} = {predictions[image_index]}')


if __name__ == "__main__":
    main.add_command(classify_images)
    main()
