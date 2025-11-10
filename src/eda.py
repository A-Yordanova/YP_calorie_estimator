import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def analyze_num_cols(df: DataFrame, to_drop=None):
    if to_drop is not None:
        num_cols_df = df.select_dtypes(exclude='object').drop(to_drop, axis=1).columns
    else:
        num_cols_df = df.select_dtypes(exclude='object').columns
    display(df[num_cols_df].describe())
    for c in num_cols_df:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        df[c].plot(kind='hist', bins=50, ax=axes[0])
        df[c].plot(kind='box', ax=axes[1])
        axes[0].set_title(f'Гистограмма распределения {c}')
        axes[1].set_title(f'Разброс значений {c}')
        plt.show()


def plot_images(df: DataFrame, dish_ids: list[str]):
    fig, axes = plt.subplots(nrows=len(dish_ids) // 4, ncols=4, figsize=(20, 15))
    idx = 0
    axes = axes.flatten()
    for ax in axes:
        image_path = df.loc[df["dish_id"] == dish_ids[idx], "image_path"].item()
        image = Image.open(image_path).convert('RGB')
        ax.imshow(image)
        ax.set_title(dish_ids[idx])
        ax.axis('off')
        idx += 1
        if idx >= len(dish_ids):
            break
    plt.tight_layout()
    plt.show()