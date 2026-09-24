"""
plot_loss.py - turn a training log into a loss chart for the README.

Usage:
  python plot_loss.py models/shakespeare_log.csv
  python plot_loss.py models/shakespeare_log.csv --out assets/shakespeare_loss.png
"""
import argparse, csv, os
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot train/val loss from a tinygpt log.")
    parser.add_argument("log", help="CSV log written during training")
    parser.add_argument("--out", default="assets/loss_curve.png", help="where to save the chart")
    parser.add_argument("--title", default="Training and validation loss")
    args = parser.parse_args()

    steps, train_loss, val_loss = [], [], []
    with open(args.log, newline="") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, train_loss, label="train")
    plt.plot(steps, val_loss, label="validation")
    plt.xlabel("training step")
    plt.ylabel("cross-entropy loss")
    plt.title(args.title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"saved {args.out}  (final val loss {val_loss[-1]:.3f}, best {min(val_loss):.3f})")


if __name__ == "__main__":
    main()
