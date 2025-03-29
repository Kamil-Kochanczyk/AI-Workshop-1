import sys

sys.path.insert(1, './growingspheres/')

from growingspheres import counterfactuals as cf

from anomalib.data import PredictDataset
from anomalib.models import Patchcore
from anomalib.engine import Engine

from pathlib import Path

import numpy as np
from PIL import Image
import tempfile
import os
import torch

import matplotlib.pyplot as plt

def get_numpy_image(image_path):
    try:
        arr = np.array(Image.open(image_path))
        original_shape = arr.shape
        reshaped_arr = arr.reshape(1, -1).reshape(original_shape)
        if np.array_equal(arr, reshaped_arr):
            print("The array is the same after reshaping.")
        else:
            print("The array is not the same after reshaping.")
        return arr
    except FileNotFoundError:
        print(f"The file at {image_path} does not exist.")
        return None

def save_numpy_as_png(numpy_array, output_path):
    try:
        image = Image.fromarray(numpy_array.astype(np.uint8))
        image.save(output_path)
    except Exception as e:
        print(f"Failed to save image: {e}")

# def test():
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         tmpfile_path = os.path.join(tmpdirname, "tmp_img.png")
#         print(tmpfile_path)
#         img_path = f"./datasets/MVTecAD/bottle/test/broken_small/000.png"
#         numpy_image = get_numpy_image(img_path)
#         print(f"{numpy_image.shape}, {numpy_image.min()}, {numpy_image.max()}")
#         print(numpy_image)
#         save_numpy_as_png(numpy_image, tmpfile_path)
#         original_image = Image.open(img_path)
#         tmp_image = Image.open(tmpfile_path)

#         fig, axs = plt.subplots(1, 2, figsize=(10, 5))

#         axs[0].imshow(original_image)
#         axs[0].set_title("Original Image")
#         axs[0].axis("off")

#         axs[1].imshow(tmp_image)
#         axs[1].set_title("Output Image")
#         axs[1].axis("off")

#         plt.tight_layout()
#         plt.show()

cat = "zipper"
global_ckpt_path = f"./models/{cat}/model.ckpt"
backbone_index = 1
subdir = "good"
number = 0
global_img_path = f"./datasets/MVTecAD/{cat}/test/{subdir}/{str(number).zfill(3)}.png"
global_numpy_image = get_numpy_image(global_img_path)
global_shape = global_numpy_image.shape
global_tmpdirname = tempfile.TemporaryDirectory()
dict_backbones = {0: "wide_resnet50_2", 1: "resnet18"}
model = Patchcore(
    backbone=dict_backbones[backbone_index],
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1
)
engine = Engine()
print(engine.predict(model=model, dataset=PredictDataset(path=Path(global_img_path)), ckpt_path=global_ckpt_path)[0].pred_label.item())
print("=======================================")

def my_func(row):
    tmpfile_path = os.path.join(global_tmpdirname.name, "tmp_img.png")
    print(row.shape)
    save_numpy_as_png(row.reshape(global_shape), tmpfile_path)
    dataset = PredictDataset(path=Path(tmpfile_path))
    prediction = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=global_ckpt_path
    )
    print(prediction[0].pred_label.item())
    print("========================")
    return prediction[0].pred_label.item()

def predict_numpy_image(numpy_images_reshaped):
    print("hello")
    print(numpy_images_reshaped.shape)
    print(len(numpy_images_reshaped))
    return np.apply_along_axis(my_func, axis=1, arr=numpy_images_reshaped)

def predict_on_category():
    dict_backbones = {0: "wide_resnet50_2", 1: "resnet18"}
    model = Patchcore(
        backbone=dict_backbones[backbone_index],
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1
    )
    engine = Engine()
    dataset = PredictDataset(path=Path(global_img_path))
    prediction = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=global_ckpt_path
    )
    return prediction[0]

# p1 = predict_numpy_image(numpy_image.reshape(1, -1))
# p2 = predict_on_category()

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# print(p1.pred_label)
# print(p2.pred_label)
# print(p1.pred_score)
# print(p2.pred_score)

# axs[0].imshow(p1.anomaly_map.detach().cpu().numpy().squeeze())
# axs[0].set_title("Original Image")
# axs[0].axis("off")

# axs[1].imshow(p2.anomaly_map.detach().cpu().numpy().squeeze())
# axs[1].set_title("Output Image")
# axs[1].axis("off")

# plt.tight_layout()
# plt.show()

obs = global_numpy_image.reshape(1, -1)
print("1")
counterfactual_explanation = cf.CounterfactualExplanation(obs_to_interprete=obs, prediction_fn=predict_numpy_image)
print("2")
counterfactual_explanation.fit(n_in_layer=3, layer_shape='sphere', first_radius=250.0, dicrease_radius=10.0, verbose=True, caps=[0.0, 255.0])
print("3")
e = counterfactual_explanation.enemy.reshape(1, -1)
print("4")
e_star = counterfactual_explanation.e_star.reshape(1, -1)
print("5")
move = counterfactual_explanation.move
print("6")

fig, axs = plt.subplots(1, 3, figsize=(10, 5))

axs[0].imshow(obs.reshape(global_shape))
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(e.reshape(global_shape))
axs[1].set_title("e")
axs[1].axis("off")

axs[2].imshow(e_star.reshape(global_shape))
axs[2].set_title("e star")
axs[2].axis("off")

plt.tight_layout()
plt.show()

global_tmpdirname.cleanup()
