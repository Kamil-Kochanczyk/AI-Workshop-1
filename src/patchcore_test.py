from anomalib.data import PredictDataset
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

from pathlib import Path
import os

from my_lib import get_categories, summarize_predictions, visualize_prediction

categories = get_categories()

def test_on_category(cat, backbone_index):
    dict_backbones = {0: "wide_resnet50_2", 1: "resnet18"}
    model = Patchcore(
        backbone=dict_backbones[backbone_index],
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1
    )
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=cat,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=os.cpu_count()
    )
    engine = Engine()
    results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=f"./models/{cat}/model.ckpt"
    )
    return results

def predict_on_category(cat, backbone_index):
    dict_backbones = {0: "wide_resnet50_2", 1: "resnet18"}
    model = Patchcore(
        backbone=dict_backbones[backbone_index],
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1
    )
    engine = Engine()
    dataset = PredictDataset(path=Path(f"./datasets/MVTecAD/{cat}/test"))
    predictions = engine.predict(
        model=model,
        dataset=dataset,
        ckpt_path=f"./models/{cat}/model.ckpt"
    )
    return predictions

def main():
    chosen_mode = input("Enter a mode (test/predict): ").strip().lower()
    chosen_cat = input("Enter a category: ")
    model_used = input("Enter a model that was used to train the category (0 for wide_resnet50_2, 1 for resnet18): ")
    
    # Wrong model won't correspond properly to the .ckpt file and result in an error (check readme.md)

    if chosen_mode == "test":
        results = test_on_category(chosen_cat, int(model_used))
        print(f"Results for {chosen_cat}:")
        print(results)
    elif chosen_mode == "predict":
        if chosen_cat in categories and model_used in ["0", "1"]:
            predictions = predict_on_category(chosen_cat, int(model_used))

            if predictions is not None:
                summarize_predictions(predictions)
                print()
                while True:
                    try:
                        index = int(input("Enter an index to visualize (or -1 to stop): "))
                        if index == -1:
                            break
                        if 0 <= index < len(predictions):
                            visualize_prediction(predictions[index])
                        else:
                            print("Invalid index.")
                    except ValueError:
                        print("Invalid input.")
        else:
            print("Invalid category or model.")
    else:
        print("Invalid mode.")

if __name__ == '__main__':
    main()
