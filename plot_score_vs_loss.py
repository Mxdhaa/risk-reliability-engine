from src.figures import make_figure4

if __name__ == "__main__":
    out_path = "artifacts/figure4.png"
    make_figure4(preds_path="artifacts/preds.csv", out_path=out_path, gamma=1.0)
    print(f"Saved: {out_path}")

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(out_path)
    plt.figure(figsize=(12, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Figure 4")
    plt.show()
