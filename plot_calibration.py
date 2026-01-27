from src.figures import make_figure5

if __name__ == "__main__":
    out_path = "artifacts/figure5.png"
    make_figure5(preds_path="artifacts/preds.csv", out_path=out_path, gamma=1.0, n_bins=10)
    print(f"Saved: {out_path}")

    # show it like killer_plot()
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(out_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Figure 5")
    plt.show()
