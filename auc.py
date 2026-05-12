import os
import argparse


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Compute AUC and evaluation metrics")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=False)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


# Load predictions (NO LABEL CONVERSION HERE)
def load_predictions(file_path):
    # Expected format: label, predict, score
    positives = 0
    negatives = 0
    predictions = []

    with open(file_path, "r") as f:
        lines = f.readlines()[1:]  # skip header

    for line in lines:
        label, _, score = line.strip().split("\t")

        label = int(label)      # stays -1 or 1
        score = float(score)

        if label == 1:
            positives += 1
        elif label == -1:
            negatives += 1
        else:
            raise ValueError(f"Unexpected label: {label}")

        predictions.append((label, score))

    return predictions, positives, negatives


# Compute ROC + AUC
def compute_metrics(predictions, positives, negatives, threshold):

    # Sort by score descending
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)

    TP = FP = 0
    TP_fixed = FP_fixed = 0

    auc_area = 0

    roc_points = []
    table = []

    next_tp_target = 1
    next_fpr_target = 1e-5

    for i, (label, score) in enumerate(predictions):

        # Positive = label == 1
        # Negative = label == -1
        if label == 1:
            TP += 1
            if score >= threshold:
                TP_fixed += 1

            auc_area += FP

        elif label == -1:
            FP += 1
            if score >= threshold:
                FP_fixed += 1

        # ROC curve sampling
        if TP >= next_tp_target or i == len(predictions) - 1:
            tpr = TP / positives
            fpr = FP / negatives

            roc_points.append((score, TP, FP, tpr, fpr))
            next_tp_target += 0.05 * positives

        # Log-scale FPR table
        if FP >= next_fpr_target * negatives or i == len(predictions) - 1:
            tpr = TP / positives
            table.append((next_fpr_target, tpr))
            next_fpr_target *= 10

    # AUC calculation with formula
    auc = 1.0 - (auc_area / positives) / negatives

    metrics = {
        "TP": TP_fixed,
        "FN": positives - TP_fixed,
        "FP": FP_fixed,
        "TN": negatives - FP_fixed,
        "AUC": auc,
        "roc": roc_points,
        "table": table,
    }

    return metrics


# Save results
def save_results(output_path, metrics, positives, negatives):

    with open(output_path, "w") as f:

        total = positives + negatives

        f.write(f"Total number of instances: {total}\n")
        f.write(f"P (malicious=1): {positives}\n")
        f.write(f"N (benign=-1): {negatives}\n")

        f.write("-" * 30 + "\n")
        f.write("ROC Curve\n")
        f.write("-" * 30 + "\n")
        f.write("threshold\tTP\tFP\tTPR\tFPR\n")

        for score, TP, FP, TPR, FPR in metrics["roc"]:
            f.write(f"{score}\t{TP}\t{FP}\t{TPR}\t{FPR}\n")

        f.write("-" * 30 + "\n")
        f.write("FPR vs TPR Table\n")
        f.write("-" * 30 + "\n")
        f.write("FPR\tTPR\n")

        for fpr, tpr in metrics["table"]:
            f.write(f"{fpr}\t{tpr}\n")

        f.write("-" * 30 + "\n")
        f.write(f"AUC:\t{metrics['AUC']}\n")

        f.write("Confusion Matrix @ threshold\n")
        f.write(f"TP:\t{metrics['TP']}\n")
        f.write(f"FN:\t{metrics['FN']}\n")
        f.write(f"FP:\t{metrics['FP']}\n")
        f.write(f"TN:\t{metrics['TN']}\n")


def save_roc_csv(output_path, metrics):
    base, _ = os.path.splitext(output_path)  # removes any extension
    csv_path = base + ".csv"

    with open(csv_path, "w") as f:
        f.write("threshold,TP,FP,TPR,FPR\n")
        for score, TP, FP, TPR, FPR in metrics["roc"]:
            f.write(f"{score},{TP},{FP},{TPR},{FPR}\n")

    return csv_path


# Main
def main():
    args = parse_args()

    input_file = os.path.join(args.input_path, args.input_file)
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = os.path.join(
            args.input_path,
            args.input_file.replace(".txt", ".auc")
        )

    predictions, pos, neg = load_predictions(input_file)

    metrics = compute_metrics(predictions, pos, neg, args.threshold)

    save_results(output_file, metrics, pos, neg)
    csv_file = save_roc_csv(output_file, metrics)

    print(f"Saved report to {output_file}")
    print(f"Saved ROC data to {csv_file}")


if __name__ == "__main__":
    main()