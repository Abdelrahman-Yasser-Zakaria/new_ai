import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.utils import img_to_array, load_img  # type: ignore
from ultralytics import YOLO
import shutil
import sys


class_labels = [
    "Cucumber__Anthracnose",
    "Cucumber__Bacterial Wilt",
    "Cucumber__Downy Mildew",
    "Cucumber__Fresh Leaf",
    "Cucumber__Gummy_Stem_Blight",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
    "Tomato___mosaic_virus",
]

diseased_labels = [
    "Cucumber__Anthracnose",
    "Cucumber__Bacterial Wilt",
    "Cucumber__Downy Mildew",
    "Cucumber__Gummy_Stem_Blight",
    "Pepper__bell___Bacterial_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus",
]

cropped_images_dir = "/home/abdo/Code/Python/Deep_Learning/plant_village/cropped_images"


def resize_images(original_images_dir, target_size=(416, 416)):
    for filename in os.listdir(original_images_dir):
        if (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".jpeg")
            or filename.endswith(".JPG")
        ):
            img_path = os.path.join(original_images_dir, filename)
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, target_size)
            cv2.imwrite(img_path, resized_image)


def detection(original_images_dir):

    # Load YOLO 8 detection model
    yolo_model = YOLO(
        "/home/abdo/Code/Python/Deep_Learning/plant_village/training/best (6).pt"
    )

    if os.path.exists(cropped_images_dir):
        shutil.rmtree(cropped_images_dir)

    # detect leafs and cropped them using yolo model
    results = yolo_model.predict(source=original_images_dir, conf=0.25)
    for result in results:
        result.save_crop(save_dir=cropped_images_dir)


# Function to preprocess and classify images using tensorflow model
def classify_cropped_images():

    # Load TensorFlow classification model
    classifier_model = load_model(
        "/home/abdo/Code/Python/Deep_Learning/plant_village/training/new_dataset_models/resnet152_98.16%.keras"
    )

    # Create classified images directory
    cropped_images_leaf_dir = os.path.join(cropped_images_dir, "leaf")
    classified_images_dir = (
        "/home/abdo/Code/Python/Deep_Learning/plant_village/classified_images"
    )

    if os.path.exists(classified_images_dir):
        shutil.rmtree(classified_images_dir)

    os.makedirs(classified_images_dir)
    # Create subdirectories for healthy and diseased labels
    os.makedirs(os.path.join(classified_images_dir, "healthy"))
    os.makedirs(os.path.join(classified_images_dir, "diseased"))

    for filename in os.listdir(cropped_images_leaf_dir):

        if (
            filename.endswith(".jpg")
            or filename.endswith(".png")
            or filename.endswith(".jpeg")
            or filename.endswith(".JPG")
        ):
            # Load and preprocess the image
            img_path = os.path.join(cropped_images_leaf_dir, filename)
            image = load_img(img_path, target_size=(256, 256))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # Predict the class
            prediction = classifier_model.predict(image)
            predicted_probability = round((prediction.max()) * 100, 2)
            predicted_class = class_labels[prediction.argmax()]

            # Print the results
            print(
                f"Image: {filename}, Predicted class: {predicted_class}:{predicted_probability}"
            )

            # Load the cropped image using OpenCV
            img_cv2 = cv2.imread(img_path)

            # # Calculate the scaling factor based on the image size
            # height, width, _ = img_cv2.shape
            # scaling_factor = min(width, height) / 500

            # # Put the predicted class text on the image
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # font_scale = 1 * scaling_factor
            # thickness = int(2 * scaling_factor)
            # cv2.putText(
            #     img_cv2,
            #     f"{predicted_class}_{predicted_probability}",
            #     (10, int(30 * scaling_factor)),
            #     font,
            #     font_scale,
            #     (0, 255, 0),
            #     thickness,
            #     cv2.LINE_AA,
            # )

            # Save the image in the corresponding class directory
            if predicted_class in diseased_labels:
                output_path = os.path.join(
                    classified_images_dir,
                    "diseased",
                    f"{predicted_class}_{filename}.jpg",
                )
                cv2.imwrite(output_path, img_cv2)
            else:
                output_path = os.path.join(
                    classified_images_dir,
                    "healthy",
                    f"{predicted_class}_{filename}.jpg",
                )
                cv2.imwrite(output_path, img_cv2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <original_images_path>")
    else:
        original_images_dir = sys.argv[1]
        resize_images(original_images_dir)
        detection(original_images_dir)
        classify_cropped_images()
