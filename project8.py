"""
CST-305 Project 8 - Numerical Integration
Rodrigo Gomez

Packages used: math, csv, pathlib, numpy, matplotlib

This program uses Riemann sums to approximate definite integrals.
In Part 1, it solves the functions required in the assignment.
In Part 2, it applies the same numerical integration idea to estimate
the total amount of downloaded data over a 30-minute interval.
"""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def riemann_sum(func, a, b, n, method="left"):
    # This is the general Riemann sum function used throughout the project.
    dx = (b - a) / n

    if method == "left":
        sample_points = a + np.arange(n) * dx
    elif method == "right":
        sample_points = a + np.arange(1, n + 1) * dx
    elif method == "midpoint":
        sample_points = a + (np.arange(n) + 0.5) * dx
    else:
        raise ValueError("method must be 'left', 'right', or 'midpoint'")

    return np.sum(func(sample_points)) * dx


def plot_riemann_rectangles(func, a, b, n, method, filename, title):
    x = np.linspace(a, b, 1000)
    y = func(x)
    dx = (b - a) / n
    left_edges = a + np.arange(n) * dx

    if method == "left":
        ck = left_edges
    elif method == "right":
        ck = left_edges + dx
    else:
        ck = left_edges + dx / 2

    heights = func(ck)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label="f(x)")
    plt.bar(
        left_edges,
        heights,
        width=dx,
        align="edge",
        alpha=0.35,
        edgecolor="black",
        label=f"{method.capitalize()} rectangles"
    )
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def plot_ln_dense():
    x = np.linspace(1, math.e, 5000)
    y = np.log(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("ln(x)")
    plt.title("y = ln(x) on [1, e]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ln_dense_plot.png", dpi=200)
    plt.close()


def create_flowchart():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis("off")

    boxes = [
        (0.5, 0.90, "Start"),
        (0.5, 0.74, "Choose function or data"),
        (0.5, 0.58, "Set interval [a, b]\nand number of subintervals n"),
        (0.5, 0.42, "Compute Δx and sample points c_k"),
        (0.5, 0.26, "Evaluate Σ f(c_k)Δx"),
        (0.5, 0.10, "Display result and graph"),
    ]

    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black")
        )

    arrows = [(0.86, 0.78), (0.70, 0.62), (0.54, 0.46), (0.38, 0.30), (0.22, 0.14)]
    for start_y, end_y in arrows:
        ax.annotate(
            "",
            xy=(0.5, end_y),
            xytext=(0.5, start_y),
            arrowprops=dict(arrowstyle="->", lw=1.5)
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "algorithm_flowchart.png", dpi=200, bbox_inches="tight")
    plt.close()


def solve_part_1b():
    # Here I write the Riemann sum formula for right endpoints and keep the limit separately.
    formula = (
        "S_n = Σ[(3k/n) + 2(k^2/n^2)](1/n), for k = 1 to n\n"
        "    = (3/n^2)Σk + (2/n^3)Σk^2\n"
        "    = 3(n+1)/(2n) + (n+1)(2n+1)/(3n^2)"
    )
    limit_value = 13 / 6
    return formula, limit_value


def solve_part_1c1(n=100000):
    # For this part I use a very large n so the approximation is very close to the exact value.
    a = 1
    b = math.e
    dx = (b - a) / n
    k = np.arange(1, n + 1)
    ck = a + k * dx
    approximation = np.sum(np.log(ck)) * dx
    exact_value = 1.0

    formula = (
        "Δx = (e - 1)/n\n"
        "c_k = 1 + k(e - 1)/n\n"
        "S_n = Σ ln(1 + k(e - 1)/n) * (e - 1)/n"
    )

    return formula, approximation, exact_value


def solve_part_1c2():
    # This stores the algebraic form of the Riemann sum before taking the limit as n goes to infinity.
    formula = (
        "Δx = 1/n\n"
        "x_k = -1 + k/n\n"
        "S_n = (1/n)Σ[(-1 + k/n)^2 - (-1 + k/n)^3]\n"
        "    = (1/n)Σ[2 - 5k/n + 4k^2/n^2 - k^3/n^3]"
    )
    limit_value = 7 / 12
    return formula, limit_value


def create_sample_csv(csv_path):
    rows = [
        [0, 82.4], [1, 84.1], [2, 79.8], [3, 88.3], [4, 91.5],
        [5, 86.2], [6, 83.7], [7, 89.4], [8, 94.6], [9, 92.8],
        [10, 87.9], [11, 85.3], [12, 90.1], [13, 96.2], [14, 93.5],
        [15, 88.7], [16, 84.9], [17, 86.8], [18, 91.2], [19, 95.1],
        [20, 97.4], [21, 92.3], [22, 89.6], [23, 87.2], [24, 90.8],
        [25, 94.0], [26, 96.5], [27, 93.1], [28, 88.9], [29, 85.7],
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["minute", "rate_MBps"])
        writer.writerows(rows)


def load_download_rates(csv_path):
    if not csv_path.exists():
        create_sample_csv(csv_path)

    minutes = []
    rates = []

    with open(csv_path, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            minutes.append(int(row["minute"]))
            rates.append(float(row["rate_MBps"]))

    return np.array(minutes), np.array(rates)


def build_step_rate_function(rates):
    rates = np.array(rates, dtype=float)

    def rate_function(t):
        t = np.asarray(t, dtype=float)
        indices = np.floor(t).astype(int)
        indices = np.clip(indices, 0, len(rates) - 1)
        return rates[indices]

    return rate_function


def plot_network_rates(minutes, rates):
    x = np.append(minutes, 30)
    y = np.append(rates, rates[-1])

    plt.figure(figsize=(8, 5))
    plt.step(x, y, where="post", label="R(t)")
    plt.scatter(minutes, rates, s=20)
    plt.xlabel("Time (minutes)")
    plt.ylabel("Rate (MB/s)")
    plt.title("Download rate over 30 minutes")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "network_rate_plot.png", dpi=200)
    plt.close()


def analyze_part_2(minutes, rates):
    # In Part 2, the integral gives accumulated data, so I multiply by 60
    # because the rate is in MB/s and time is measured in minutes.
    rate_function = build_step_rate_function(rates)
    integral_value = riemann_sum(rate_function, 0, 30, 30, method="left")

    total_megabytes = integral_value * 60
    total_gibibytes = total_megabytes / 1024
    average_rate = np.mean(rates)

    plot_network_rates(minutes, rates)

    return total_megabytes, total_gibibytes, average_rate


def save_output(text):
    with open(OUTPUT_DIR / "program_output.txt", "w", encoding="utf-8") as file:
        file.write(text)


def main():
    create_flowchart()

    # Part 1(a): compare left, right, and midpoint Riemann sums for the same function.
    f1 = lambda x: np.sin(x) + 1
    a1 = -math.pi
    b1 = math.pi
    n1 = 4

    left_sum = riemann_sum(f1, a1, b1, n1, "left")
    right_sum = riemann_sum(f1, a1, b1, n1, "right")
    midpoint_sum = riemann_sum(f1, a1, b1, n1, "midpoint")
    exact_1a = 2 * math.pi

    plot_riemann_rectangles(
        f1, a1, b1, n1, "left",
        "sin_left.png",
        "f(x) = sin(x) + 1 on [-π, π] using left endpoints"
    )
    plot_riemann_rectangles(
        f1, a1, b1, n1, "right",
        "sin_right.png",
        "f(x) = sin(x) + 1 on [-π, π] using right endpoints"
    )
    plot_riemann_rectangles(
        f1, a1, b1, n1, "midpoint",
        "sin_midpoint.png",
        "f(x) = sin(x) + 1 on [-π, π] using midpoint rule"
    )

    # Part 1(b): formula and limit for the polynomial on [0, 1].
    formula_1b, limit_1b = solve_part_1b()

    # Part 1(c)(1): numerical approximation for integral of ln(x).
    plot_ln_dense()
    formula_1c1, approx_1c1, exact_1c1 = solve_part_1c1()

    # Part 1(c)(2): formula and limit for x^2 - x^3 on [-1, 0].
    formula_1c2, limit_1c2 = solve_part_1c2()

    # Part 2: use the recorded rate values to estimate total downloaded data.
    csv_path = Path("download_rates.csv")
    minutes, rates = load_download_rates(csv_path)
    total_mb, total_gib, avg_rate = analyze_part_2(minutes, rates)

    output_text = f"""
CST-305 Project 8 - Numerical Integration Results

PART 1(a): f(x) = sin(x) + 1 on [-pi, pi], n = 4
Left Riemann sum     = {left_sum:.6f}
Right Riemann sum    = {right_sum:.6f}
Midpoint Riemann sum = {midpoint_sum:.6f}
Exact integral       = {exact_1a:.6f}

PART 1(b): f(x) = 3x + 2x^2 on [0, 1]
{formula_1b}
Limit as n -> infinity = {limit_1b:.6f}

PART 1(c)(1): integral from 1 to e of ln(x) dx
{formula_1c1}
High-granularity approximation (n = 100000) = {approx_1c1:.6f}
Exact value = {exact_1c1:.6f}

PART 1(c)(2): f(x) = x^2 - x^3 on [-1, 0]
{formula_1c2}
Limit as n -> infinity = {limit_1c2:.6f}

PART 2: Downloaded data over 30 minutes
Average transfer rate = {avg_rate:.2f} MB/s
Estimated total downloaded data = {total_mb:.2f} MB
Estimated total downloaded data = {total_gib:.2f} GiB

Interpretation:
The integral of the rate function represents the total amount of data
transferred over the full experiment. For this dataset, the server
downloaded about {total_gib:.2f} GiB in 30 minutes.
""".strip()

    print(output_text)
    save_output(output_text)


if __name__ == "__main__":
    main()