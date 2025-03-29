from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

import os

from my_lib import get_categories

categories = get_categories()

def train_on_category(cat, back):
    datamodule = MVTecAD(
        root="./datasets/MVTecAD",
        category=cat,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=os.cpu_count()
    )
    model = Patchcore(
        backbone=back,
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1
    )
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
    print(engine.test(model=model, datamodule=datamodule))

def main():
    # Last trained: hazelnut, wide_resnet50_2
    first_cat = 5
    last_cat = 5
    for cat in categories[first_cat:last_cat + 1]:
        train_on_category(cat, "wide_resnet50_2")
        # train_on_category(cat, "resnet18")
        pass

if __name__ == '__main__':
    main()
