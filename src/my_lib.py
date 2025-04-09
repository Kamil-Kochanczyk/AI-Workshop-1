import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

def get_categories():
    return ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]

def get_numpy_image(image_path, size=None):
    # Returned array is of type uint8, with values between 0 and 255
    # It can either be 2D (grayscale) or 3D (RGB), e.g. (256, 256) or (256, 256, 3)
    try:
        if size is None:
            return np.array(Image.open(image_path))
        else:
            return np.array(Image.open(image_path).resize(size))
    except Exception as e:
        print(f"Failed to get image: {e}")
        return None

def get_numpy_images(image_paths, size=None):
    images = []
    for image_path in image_paths:
        image = get_numpy_image(image_path, size)
        if image is not None:
            images.append(image)
    return images

def save_numpy_image_as_png(numpy_image, output_path):
    try:
        Image.fromarray(numpy_image.astype(np.uint8)).save(output_path)
        return True
    except Exception as e:
        print(f"Failed to save image: {e}")
        return False

def lerp_probabilities(pred_score):
    # x coordinate - it's index 0 and corresponds to label False, value is probability of no anomaly
    # y coordinate - it's index 1 and corresponds to label True, value is probability of anomaly
    t = pred_score
    a = np.array([1, 0])
    b = np.array([0, 1])
    return (1 - t) * a + t * b

def get_default_visualization(tested_image_path):
    # Argument can be for example "...\\datasets\\MVTecAD\\bottle\\test\\broken_large\\000.png"
    split_path = tested_image_path.split("\\")
    category_name = split_path[-4]
    last_3 = split_path[-3:]
    to_be_joined = ["results", "Patchcore", "MVTecAD", category_name, "latest", "images"]
    to_be_joined.extend(last_3)
    default_vis_path = "\\".join(to_be_joined)
    try:
        return np.array(Image.open(default_vis_path))
    except FileNotFoundError:
        print(f"File not found: {default_vis_path}")
        return None
    except Exception as e:
        print(f"Error opening file {default_vis_path}: {e}")
        return None

def is_correct_prediction(prediction):
    image_path = prediction.image_path
    pred_label = prediction.pred_label
    is_anomalous = "good" not in str(image_path)
    return (pred_label.item() == True and is_anomalous) or (pred_label.item() == False and not is_anomalous)

def extract_indexes(predictions):
    normal_indexes = []
    anomalous_indexes = []
    correct_indexes = []
    incorrect_indexes = []
    for i, prediction in enumerate(predictions):
        if prediction.pred_label.item() == False:
            normal_indexes.append(i)
        elif prediction.pred_label.item() == True:
            anomalous_indexes.append(i)
        if is_correct_prediction(prediction):
            correct_indexes.append(i)
        else:
            incorrect_indexes.append(i)
    return normal_indexes, anomalous_indexes, correct_indexes, incorrect_indexes

def summarize_predictions(predictions):
    print("Summarizing predictions...")

    print(f"Number of predictions: {len(predictions)}")

    num_normal = sum([p.pred_label.item() == False for p in predictions])
    num_anomalous = sum([p.pred_label.item() == True for p in predictions])
    print(f"Number of normal predictions: {num_normal}")
    print(f"Number of anomalous predictions: {num_anomalous}")

    correct_predictions = sum([is_correct_prediction(p) for p in predictions])
    accuracy = (correct_predictions / len(predictions)) * 100
    print(f"Number of correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")

    normal_indexes, anomalous_indexes, correct_indexes, incorrect_indexes = extract_indexes(predictions)
    print(f"Normal Indexes: {normal_indexes}")
    print(f"Anomalous Indexes: {anomalous_indexes}")
    print(f"Correct Indexes: {correct_indexes}")
    print(f"Incorrect Indexes: {incorrect_indexes}")

def visualize_prediction(prediction):
    image_path = prediction.image_path  # Path to the input image
    anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
    pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
    pred_score = prediction.pred_score  # Image-level anomaly score
    is_correct = is_correct_prediction(prediction)

    print(f"Image Path: {image_path}")
    print(f"Anomaly Map Shape: {anomaly_map.shape}, Min: {anomaly_map.min()}, Max: {anomaly_map.max()}")
    print(f"Predicted Label: {pred_label}")
    print(f"Predicted Score: {pred_score}")
    print(f"Prediction Correct: {is_correct}\n")

    if isinstance(image_path, list):
        image_path = image_path[0]
    
    if isinstance(anomaly_map, torch.Tensor):
        anomaly_map = anomaly_map.detach().cpu().numpy().squeeze()

    original_image = np.array(Image.open(image_path))
    
    # Check if the image is grayscale
    if len(original_image.shape) == 2:  # Grayscale images have only 2 dimensions
        is_grayscale = True
    else:
        is_grayscale = False

    # Normalize anomaly map for visualization
    anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

    # Check if the image is anomalous and if so, find its ground truth anomaly map
    ground_truth_anomaly_map = None
    is_anomalous = "good" not in str(image_path)
    if is_anomalous:
        ground_truth_anomaly_map_path = image_path.replace("test", "ground_truth").replace(".png", "_mask.png")
        ground_truth_anomaly_map = np.array(Image.open(ground_truth_anomaly_map_path))

    # Display default visualization
    # Also display custom visualization using prediction data (the original image, anomaly map, and ground truth (if it exists))

    default_vis = get_default_visualization(image_path)

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 1, 1)
    plt.title(f"{image_path.split('\\')[-4]}, {image_path.split('\\')[-2]}")
    plt.imshow(default_vis)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Original Image")
    if is_grayscale:
        plt.imshow(original_image, cmap="gray")
    else:
        plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.title("Anomaly Map")
    plt.imshow(anomaly_map_normalized)
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Ground Truth")
    if is_anomalous:
        plt.imshow(ground_truth_anomaly_map, cmap="gray")
    else:
        plt.imshow(np.zeros_like(original_image), cmap="gray")
    plt.axis("off")

    plt.suptitle(f"Predicted Label: {'anomalous' if pred_label.item() == True else 'normal'}, Score: {pred_score.item():.2f}, Is Correct: {is_correct}")
    plt.show()
