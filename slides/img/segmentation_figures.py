"""Figures for the customer segmentation topic deck.

Mirrors the examples in HDBSCAN_Example_DS460.ipynb: moons and blobs for the
shape argument, then digits + UMAP for the dimension-reduction argument.

Run: uv run --with scikit-learn --with umap-learn --with plotly --with kaleido python segmentation_figures.py
"""

import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageChops
from plotly.subplots import make_subplots
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.datasets import load_digits, make_blobs, make_moons

NAVY, ORANGE, TEAL, RIDGE, SPARK = "#022F4A", "#F59625", "#4D93AC", "#426A8A", "#D5773C"
GREY = "#BDC6CC"
PALETTE = [NAVY, ORANGE, TEAL, SPARK, RIDGE, "#1F8A5B", "#C7352B", "#7B5EA7"]
FONT = dict(family="Open Sans, sans-serif", size=14, color="#111111")


def trim(path, pad=18):
    """Crop the white margin plotly leaves around sparse scatter panels."""
    im = Image.open(path).convert("RGB")
    bbox = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        box = (max(bbox[0] - pad, 0), max(bbox[1] - pad, 0),
               min(bbox[2] + pad, im.width), min(bbox[3] + pad, im.height))
        im.crop(box).save(path)


def colors(labels):
    return [GREY if v == -1 else PALETTE[int(v) % len(PALETTE)] for v in labels]


def scatter(fig, row, col, x, y, labels, title):
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers",
            marker=dict(size=5, color=colors(labels), opacity=0.85),
            showlegend=False,
        ),
        row=row, col=col,
    )
    fig.layout.annotations[(row - 1) * 2 + col - 1].update(text=title)


def shape_figure(path="segmentation_shapes.png"):
    """k-means cuts by distance to a centre; HDBSCAN follows density."""
    moons, _ = make_moons(n_samples=1000, noise=0.08, random_state=0)
    centers = [[-0.85, -0.85], [-0.85, 0.85], [3, 3], [3, -3]]
    blobs, _ = make_blobs(n_samples=750, centers=centers,
                          cluster_std=[0.2, 0.35, 1.35, 1.35], random_state=0)

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.13, horizontal_spacing=0.08,
                        subplot_titles=[" "] * 4)
    scatter(fig, 1, 1, moons[:, 0], moons[:, 1], KMeans(2, n_init=10).fit_predict(moons),
            "k-means, 2 clusters")
    scatter(fig, 1, 2, moons[:, 0], moons[:, 1],
            HDBSCAN(min_cluster_size=20).fit_predict(moons), "HDBSCAN")
    scatter(fig, 2, 1, blobs[:, 0], blobs[:, 1], KMeans(4, n_init=10).fit_predict(blobs),
            "k-means, 4 clusters")
    scatter(fig, 2, 2, blobs[:, 0], blobs[:, 1],
            HDBSCAN(min_cluster_size=20).fit_predict(blobs), "HDBSCAN (grey = noise)")

    fig.update_layout(width=1100, height=720, font=FONT, plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(l=30, r=30, t=50, b=30))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    for a in fig.layout.annotations:
        a.update(font=dict(family="Fira Code, monospace", size=15, color=NAVY))
    fig.write_image(path, scale=2)
    trim(path)
    print("wrote", path)


def umap_figure(path="segmentation_umap.png"):
    """64 dimensions of pixels, projected to 2, then clustered."""
    import umap

    digits = load_digits()
    emb = umap.UMAP(n_neighbors=15, min_dist=0.12, n_components=2,
                    random_state=42).fit_transform(digits.data)
    emb5 = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=5,
                     random_state=42).fit_transform(digits.data)
    labels = HDBSCAN(min_cluster_size=25, min_samples=10).fit_predict(emb5)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.07, subplot_titles=[" ", " "])
    scatter(fig, 1, 1, emb[:, 0], emb[:, 1], digits.target, "Coloured by the true digit")
    scatter(fig, 1, 2, emb[:, 0], emb[:, 1], labels, "Coloured by HDBSCAN (grey = noise)")
    fig.update_layout(width=1200, height=560, font=FONT, plot_bgcolor="white",
                      paper_bgcolor="white", margin=dict(l=30, r=30, t=50, b=30))
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    for a in fig.layout.annotations:
        a.update(font=dict(family="Fira Code, monospace", size=15, color=NAVY))
    fig.write_image(path, scale=2)
    trim(path)
    found = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"wrote {path}: {found} clusters, {(labels == -1).mean():.1%} noise")


if __name__ == "__main__":
    shape_figure()
    umap_figure()
