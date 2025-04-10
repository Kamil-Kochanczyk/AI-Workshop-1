from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine

from pathlib import Path

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, mark_boundaries

import numpy as np
import matplotlib.pyplot as plt

import tempfile
import os

from my_lib import get_numpy_image, get_numpy_images, save_numpy_image_as_png
from my_lib import lerp_probabilities
from my_lib import is_correct_prediction, summarize_predictions, visualize_prediction

import growing_spheres as gs
import sedc_t as sedct
import face as face

# def test():
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         tmpfile_path = os.path.join(tmpdirname, "tmp_img.png")
#         print(tmpfile_path)

#         img_path = f"./datasets/MVTecAD/bottle/test/broken_small/000.png"
#         # img_path = f"./datasets/MVTecAD/zipper/test/broken_teeth/000.png"
#         numpy_image = get_numpy_image(img_path, (256, 256))
#         print(f"{numpy_image.shape}, {numpy_image.min()}, {numpy_image.max()}")

#         save_numpy_image_as_png(numpy_image, tmpfile_path)

#         original_image = Image.open(img_path)
#         tmp_image = Image.open(tmpfile_path)

#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#         axs[0].imshow(original_image)
#         axs[0].set_title("Original Image")
#         axs[0].axis("off")

#         axs[1].imshow(tmp_image)
#         axs[1].set_title("Tmp Image")
#         axs[1].axis("off")

#         plt.tight_layout()
#         plt.show()

class CounterfactualExplanationsWrapper:    
    def __init__(self, category, backbone_index):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = os.path.join(self.tmp_dir.name, "tmp.png")

        self.category = category
        self.ckpt_path = f"./models/{self.category}/model.ckpt"
        
        if backbone_index == 0:
            self.backbone = "wide_resnet50_2"
        elif backbone_index == 1:
            self.backbone = "resnet18"
        
        self.model = Patchcore(backbone=self.backbone, layers=["layer2", "layer3"], coreset_sampling_ratio=0.1)
        self.engine = Engine()
        self.dataset = PredictDataset(path=Path(f"./datasets/MVTecAD/{self.category}/test"))

        self.predictions = self.engine.predict(
            model=self.model,
            dataset=self.dataset,
            ckpt_path=self.ckpt_path
        )

        self.is_dataset_grayscale = False  # we'll have to verify it later
        
        self.cfe_image_res = (256, 256)    # saving memory and reducing complexity of algorithms
        self.cfe_image_shape = None        # will be either 3 dim (rgb) or 2 dim (grayscale), we'll check it out later
        self.cfe_desired_label = None      # we'll figure it out later

        save_numpy_image_as_png(np.zeros((28, 28)), self.tmp_path)    # placeholder to initialize the variable below

        self.tmp_dataset = PredictDataset(path=Path(self.tmp_path))

        self.algorithm_wrappers = {
            "gs": self.growing_spheres,
            "sedct": self.sedc_t,
            "face": self.face
        }

    def __del__(self):
        self.tmp_dir.cleanup()
    
    def loop(self):
        while True:
            try:
                print()
                summarize_predictions(self.predictions)
                print()
                index = int(input("Enter an index of prediction (or -1 to stop): "))
                if index == -1:
                    break
                elif 0 <= index < len(self.predictions):
                    choice = input("What do you want to do?\nv - visualize, cfe - counterfactual explanation: ").strip().lower()
                    if choice == "v":
                        visualize_prediction(self.predictions[index])
                    elif choice == "cfe":
                        algorithm = input("Choose algorithm\ngs - Growing Spheres, sedct - SEDC-T, face - FACE: ").strip().lower()
                        if algorithm in ["gs", "sedct", "face"]:
                            print()
                            self.perform_algorithm(index, algorithm)
                        else:
                            print("Invalid algorithm.")
                    else:
                        print("Invalid choice.")
                else:
                    print("Invalid index.")
            except Exception as e:
                print(f"Exception occurred: {e}")
    
    def perform_algorithm(self, prediction_index, algorithm):
        print(f"Resizing image to {self.cfe_image_res} and predicting one more time...")

        image_path = self.predictions[prediction_index].image_path[0]
        i = get_numpy_image(image_path, self.cfe_image_res)
        
        self.is_dataset_grayscale = len(i.shape) == 2

        if self.is_dataset_grayscale:
            expanded_by_1_dim = i[:, :, np.newaxis]
            numpy_image = np.repeat(expanded_by_1_dim, 3, axis=2)
        else:
            numpy_image = i
        
        self.cfe_image_shape = numpy_image.shape

        save_numpy_image_as_png(numpy_image, self.tmp_path)

        prediction_on_resized = self.engine.predict(
            model=self.model,
            dataset=self.tmp_dataset,
            ckpt_path=self.ckpt_path
        )[0]

        print()
        print(f"Image Path: {prediction_on_resized.image_path}")
        print(f"Anomaly Map Shape: {prediction_on_resized.anomaly_map.shape}, \
                Min: {prediction_on_resized.anomaly_map.min()}, \
                Max: {prediction_on_resized.anomaly_map.max()}")
        print(f"Predicted Label: {prediction_on_resized.pred_label}")
        print(f"Predicted Score: {prediction_on_resized.pred_score}")
        print()

        self.cfe_desired_label = not prediction_on_resized.pred_label.item()

        if self.is_dataset_grayscale:
            plt.imshow(numpy_image, cmap="gray")
        else:
            plt.imshow(numpy_image)
        
        predicted_label = "anomalous" if prediction_on_resized.pred_label.item() == True else "normal"
        score = prediction_on_resized.pred_score.item()
        is_original_correct = is_correct_prediction(self.predictions[prediction_index])
        
        if prediction_on_resized.pred_label.item() == self.predictions[prediction_index].pred_label.item():
            is_correct = is_original_correct
        else:
            is_correct = not is_original_correct
        
        plt.suptitle(f"Predicted Label: {predicted_label}, Score: {score:.2f}, Is Correct: {is_correct}")
        plt.show()

        self.algorithm_wrappers[algorithm](numpy_image)
    
    def growing_spheres(self, numpy_image):
        print("Starting Growing Spheres...")
        print()

        obs = numpy_image.reshape(1, -1)
        obs = obs / 255.0

        gscfe_obj = gs.GSCFE(
            obs_to_interprete=obs,
            prediction_fn=self.predict_for_growing_spheres,
            target_class=self.cfe_desired_label,
            caps=[0.0, 1.0],
            n_in_layer=50,
            layer_shape="ball",
            first_radius=100.0,
            dicrease_radius=2.0,
            sparse=True,
            verbose=True
        )

        cfe, cfe_sparse, displacement_from_obs = gscfe_obj.find_cfe()

        if cfe is None:
            print()
            print("No counterfactual explanation found within given limits")
        else:
            # if sparse=False, normal and sparse will be the same
            cfe_sparse = (cfe_sparse.reshape(1, -1).reshape(self.cfe_image_shape) * 255.0).astype("uint8")
            cfe = (cfe.reshape(1, -1).reshape(self.cfe_image_shape) * 255.0).astype("uint8")
            displacement_from_obs = (displacement_from_obs.reshape(1, -1).reshape(self.cfe_image_shape) * 255.0).astype("uint8")

            print()
            print(cfe_sparse.shape)
            print(cfe.shape)
            print(displacement_from_obs.shape)
            print()

            fig, axs = plt.subplots(2, 2, figsize=(16, 8))

            if self.is_dataset_grayscale:
                axs[0, 0].imshow(numpy_image, cmap="gray")
                axs[0, 1].imshow(cfe_sparse, cmap="gray")
                axs[1, 0].imshow(cfe, cmap="gray")
                axs[1, 1].imshow(displacement_from_obs, cmap="gray")
            else:
                axs[0, 0].imshow(numpy_image)
                axs[0, 1].imshow(cfe_sparse)
                axs[1, 0].imshow(cfe)
                axs[1, 1].imshow(displacement_from_obs)
            
            axs[0, 0].set_title("Original image")
            axs[0, 0].axis("off")

            axs[0, 1].set_title("Sparse cfe")
            axs[0, 1].axis("off")

            axs[1, 0].set_title("Default cfe")
            axs[1, 0].axis("off")

            axs[1, 1].set_title("Difference between original and sparse")
            axs[1, 1].axis("off")

            plt.tight_layout()
            plt.show()
    
    def predict_for_growing_spheres(self, numpy_images_as_rows):
        print("Starting predict for Growing Spheres...")
        print()

        return np.apply_along_axis(self.predictor_for_row_image, axis=1, arr=numpy_images_as_rows)
    
    def predictor_for_row_image(self, row_image):
        print("Starting predictor for row image...")

        img = (row_image * 255.0).astype("uint8")
        save_numpy_image_as_png(img.reshape(self.cfe_image_shape), self.tmp_path)

        prediction = self.engine.predict(
            model=self.model,
            dataset=self.tmp_dataset,
            ckpt_path=self.ckpt_path
        )[0]

        print(f"Predicted label: {prediction.pred_label.item()}")
        print()

        return prediction.pred_label.item()
    
    def sedc_t(self, numpy_image):
        print("Starting SEDC-T...")
        print()

        segments = None
        while True:
            # kernel_size = int(input("Enter kernel size (e.g. 1): "))
            # max_dist = int(input("Enter max_dist (e.g. 100): "))
            # ratio = float(input("Enter ratio (e.g. 0.75): "))
            # segments = quickshift(numpy_image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

            n_segments = int(input("Enter the number of segments (e.g. 100): "))
            compactness = float(input("Enter compactness (try log scale, e.g. 1, 10.0, 100.0): "))
            segments = slic(numpy_image, n_segments=n_segments, compactness=compactness)

            # scale = int(input("Enter scale (e.g. 10): "))
            # sigma = float(input("Enter sigma (e.g. 0.95): "))
            # min_size = int(input("Enter min size (e.g. 5): "))
            # segments = felzenszwalb(numpy_image, scale=scale, sigma=sigma, min_size=min_size)

            print("Number of superpixels: " + str(len(np.unique(segments))))
            print()
            if self.is_dataset_grayscale:
                plt.imshow(mark_boundaries(numpy_image, segments), cmap="gray")
            else:
                plt.imshow(mark_boundaries(numpy_image, segments))
            plt.show()
            choice = input("Enter 0 to test other parameter values, -1 to cancel, or anything else to accept and move on: ")
            print()
            if choice == "0":
                continue
            elif choice == "-1":
                return
            else:
                break

        explanation, segments_in_explanation, perturbation, new_class = sedct.SEDCTCFE.sedc_t(
            image=numpy_image,
            predict_fn=self.predict_for_sedc_t,
            segments=segments,
            target_class=int(self.cfe_desired_label),    # look for the comment in my_lib.py for function lerp_probabilities
            mode="inpaint"
        )

        if explanation is None:
            print()
            print("No counterfactual explanation found within given limits")
        else:
            print(segments_in_explanation)
            print(new_class)
            print()

            fig, axs = plt.subplots(2, 2, figsize=(16, 8))

            if self.is_dataset_grayscale:
                axs[0, 0].imshow(numpy_image, cmap="gray")
                axs[0, 1].imshow(mark_boundaries(numpy_image, segments), cmap="gray")
                axs[1, 0].imshow(explanation, cmap="gray")
                axs[1, 1].imshow(perturbation, cmap="gray")
            else:
                axs[0, 0].imshow(numpy_image)
                axs[0, 1].imshow(mark_boundaries(numpy_image, segments))
                axs[1, 0].imshow(explanation)
                axs[1, 1].imshow(perturbation)
            
            axs[0, 0].set_title("Original image")
            axs[0, 0].axis("off")

            axs[0, 1].set_title("Original image + segments")
            axs[0, 1].axis("off")

            axs[1, 0].set_title("Explanation")
            axs[1, 0].axis("off")

            axs[1, 1].set_title("Perturbation")
            axs[1, 1].axis("off")

            plt.tight_layout()
            plt.show()         

    def predict_for_sedc_t(self, numpy_image):
        print("Starting predict for SEDC-T")
        print()

        save_numpy_image_as_png(numpy_image, self.tmp_path)

        prediction = self.engine.predict(
            model=self.model,
            dataset=self.tmp_dataset,
            ckpt_path=self.ckpt_path
        )[0]

        return lerp_probabilities(prediction.pred_score.item())
    
    def face(self, numpy_image):
        print("Starting FACE...")
        print()

        train_images_dir = f"./datasets/MVTecAD/{self.category}/train/good"
        train_images_paths = [os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir) if img.endswith(".png")]
        train_images = get_numpy_images(train_images_paths, self.cfe_image_res)
        X_train = list(map(lambda x: (x / 255.0).reshape(1, -1), train_images))

        distance_computer = face.DistanceComputer()
        pdf_estimator = face.PDFEstimator(distance_computer=distance_computer, bandwidth=0.25, kernel="tophat")
        feasibility_set = face.FeasibilitySet()

        # Experiment with other parameter values of pdf_estimator as those above work poorly (they lead to overflows and infinite values)

        n_train = int(len(X_train))

        facecfe = face.FACECFE(
            X_train=X_train,
            predict_fn=self.predict_for_face,
            distance_computer=distance_computer,
            pdf_estimator=pdf_estimator,
            feasibility_set=feasibility_set,
            epsilon=float("inf"),
            tp=0.5,
            td=0.001
        )

        cfe, cfe_path_cost, cfe_path = facecfe.get_cfe((numpy_image / 255.0).reshape(1, -1))

        cfe = (cfe.reshape(self.cfe_image_shape) * 255.0).astype("uint8")

        print(f"{cfe.shape}, {cfe.min()}, {cfe.max()}")

        print()
        print(f"cfe_path_cost: {cfe_path_cost}")
        print(f"cfe_path (indexes of images) (0th index isn't important, it's random): {cfe_path}")
        print()

        fig, axs = plt.subplots(1, 2, figsize=(16, 8))

        if self.is_dataset_grayscale:
            axs[0].imshow(numpy_image, cmap="gray")
            axs[1].imshow(cfe, cmap="gray")
        else:
            axs[0].imshow(numpy_image)
            axs[1].imshow(cfe)
            
            axs[0].set_title("Original image")
            axs[0].axis("off")

            axs[1].set_title("Counterfactual Explanation")
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()     
    
    def predict_for_face(self, np_img_row):
        img = (np_img_row * 255.0).astype("uint8")
        save_numpy_image_as_png(img.reshape(self.cfe_image_shape), self.tmp_path)

        prediction = self.engine.predict(
            model=self.model,
            dataset=self.tmp_dataset,
            ckpt_path=self.ckpt_path
        )[0]

        label = prediction.pred_label.item()
        probabilities = lerp_probabilities(prediction.pred_score.item())

        return label, probabilities

if __name__ == "__main__":
    cfew = CounterfactualExplanationsWrapper("hazelnut", 0)
    cfew.loop()
    # cfew.perform_algorithm(21, "face")
    del cfew
