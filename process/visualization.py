from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_optimizer_comparison(
    all_results: dict,
    save_dir: str = "results/comparison"
):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Global metrics
    metrics = [
        ("train_loss", "Training Loss", "Loss"),
        ("epoch_time", "Epoch Execution Time", "Time (s)"),
        ("optimizer_flops", "Optimizer FLOPs", "FLOPs"),
    ]

    for metric, title, ylabel in metrics:

        plt.figure(figsize=(10, 6))

        for optimizer_name, result in all_results.items():

            history = result["history"]

            plt.plot(
                history[metric],
                label=optimizer_name,
                linewidth=2
            )

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)

        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            save_dir / f"{metric}_comparison.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # Gradient norms
    norm_types = [
        ("frobenius", "Gradient Frobenius Norm"),
        ("spectral", "Gradient Spectral Norm"),
        ("nuclear", "Gradient Nuclear Norm"),
    ]

    for norm_type, title in norm_types:

        # Retrieve layer names
        first_result = next(iter(all_results.values()))

        layer_names = list(
            first_result["history"]["gradient_norms"]
            [norm_type][0].keys()
        )

        for layer_name in layer_names:

            plt.figure(figsize=(10, 6))

            for optimizer_name, result in all_results.items():

                values = [
                    epoch_metrics[layer_name]
                    for epoch_metrics in
                    result["history"]["gradient_norms"][norm_type]
                ]

                plt.plot(
                    values,
                    label=optimizer_name,
                    linewidth=2
                )

            plt.xlabel("Epoch")
            plt.ylabel("Norm")
            plt.title(
                f"{title} - {layer_name}"
            )

            plt.legend()
            plt.grid(alpha=0.3)

            plt.tight_layout()

            plt.savefig(
                save_dir /
                f"{norm_type}_{layer_name}_comparison.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

def plot_classification_comparison(results: dict, save_dir: Path):

    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("accuracy", "Test Accuracy", "Accuracy"),
        ("precision", "Test Precision (Macro)", "Precision"),
        ("recall", "Test Recall (Macro)", "Recall"),
        ("f1", "Test F1-Score (Macro)", "F1-Score"),
        ("test_loss", "Test Cross Entropy", "Loss")
    ]

    for key, title, ylabel in metrics:

        names = list(results.keys())

        values = [
            results[name][key]
            for name in names
        ]

        plt.figure(figsize=(9, 5))

        bars = plt.bar(names, values)
        
        plt.title(title)
        plt.xlabel("Optimizer")
        plt.ylabel(ylabel)

        plt.grid(axis="y", alpha=0.3)

        # Annotation des valeurs
        for bar, value in zip(bars, values):

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom"
            )

        plt.savefig(
            save_dir / f"{key}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# --------------------------------------------------
# COLORS
# --------------------------------------------------

OPTIMIZER_COLORS = {
    "sgd": "#3498DB",
    "momentum": "#E67E22",
    "nesterov": "#2ECC71",
    "adam": "#9B59B6",
    "sign": "#E74C3C",
    "muon": "#1ABC9C",
}


# --------------------------------------------------
# CONTOUR GRID
# --------------------------------------------------

def get_contour_grid(benchmark_name, resolution=350):

    if benchmark_name == "Rosenbrock":

        x = np.linspace(-2.5, 2.5, resolution)
        y = np.linspace(-1.5, 3.5, resolution)

        X, Y = np.meshgrid(x, y)

        Z = (
            (1 - X)**2
            + 100 * (Y - X**2)**2
        )

    elif benchmark_name == "Griewank":

        x = np.linspace(-12, 12, resolution)
        y = np.linspace(-12, 12, resolution)

        X, Y = np.meshgrid(x, y)

        Z = (
            1
            + (X**2 + Y**2) / 4000
            - np.cos(X) * np.cos(Y / np.sqrt(2))
        )

    else:

        raise ValueError(
            f"Unknown benchmark: {benchmark_name}"
        )

    return X, Y, Z


# --------------------------------------------------
# BENCHMARK DASHBOARD
# --------------------------------------------------

def plot_benchmark_dashboard(
    benchmark_name,
    results,
    initial_points,
    learning_rates,
    optimum,
    save_dir="results/dashboards"
):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    X, Y, Z = get_contour_grid(benchmark_name)

    # --------------------------------------------------
    # FIGURE
    # --------------------------------------------------

    fig = make_subplots(

        rows=2,
        cols=1,

        row_heights=[0.72, 0.28],

        vertical_spacing=0.12,

        subplot_titles=(
            "Optimization trajectories",
            "Convergence history"
        )
    )

    # --------------------------------------------------
    # CONTOUR
    # --------------------------------------------------

    if benchmark_name == "Rosenbrock":

        # Logarithmic contour levels
        Z_plot = np.log10(Z + 1e-10)

        contour = go.Contour(

            x=X[0, :],
            y=Y[:, 0],
            z=Z_plot,

            colorscale="Viridis",

            contours=dict(
                coloring="lines",
                showlabels=True,
                labelfont=dict(size=10)
            ),

            line=dict(width=1),

            colorbar=dict(
                title="log10(f)",
                len=0.65,
                y=0.72
            ),

            name="Contour",

            showscale=True,

            hovertemplate=(
                "x: %{x:.4f}<br>"
                "y: %{y:.4f}<br>"
                "log10(f): %{z:.4f}"
                "<extra></extra>"
            )
        )

    else:

        contour = go.Contour(

            x=X[0, :],
            y=Y[:, 0],
            z=Z,

            colorscale="Viridis",

            contours=dict(
                coloring="lines",
                showlabels=True
            ),

            line=dict(width=1),

            colorbar=dict(
                title="f(x,y)",
                len=0.65,
                y=0.72
            ),

            name="Contour",

            showscale=True
        )

    fig.add_trace(
        contour,
        row=1,
        col=1
    )

    # --------------------------------------------------
    # OPTIMIZER TRAJECTORIES
    # --------------------------------------------------

    for optimizer_name, color in OPTIMIZER_COLORS.items():

        lr = learning_rates[optimizer_name]

        first_trace = True

        for initial_point in initial_points:

            result = results[initial_point][optimizer_name]

            history = result["history"]

            xs = history["x"]
            ys = history["y"]

            # Trajectory
            fig.add_trace(

                go.Scatter(

                    x=xs,
                    y=ys,

                    mode="lines+markers",

                    line=dict(
                        color=color,
                        width=2
                    ),

                    marker=dict(
                        size=4
                    ),

                    name=f"{optimizer_name.upper()} | LR={lr:g}",

                    legendgroup=optimizer_name,

                    showlegend=first_trace,

                    customdata=np.column_stack([
                        np.arange(len(xs)),
                        history["loss"]
                    ]),

                    hovertemplate=(
                        "Iteration: %{customdata[0]}<br>"
                        "x: %{x:.6f}<br>"
                        "y: %{y:.6f}<br>"
                        "Loss: %{customdata[1]:.6e}"
                        "<extra></extra>"
                    )

                ),

                row=1,
                col=1
            )

            # Starting point
            fig.add_trace(

                go.Scatter(

                    x=[xs[0]],
                    y=[ys[0]],

                    mode="markers",

                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=color,
                        line=dict(
                            color="black",
                            width=1
                        )
                    ),

                    showlegend=False,

                    legendgroup=optimizer_name,

                    hovertemplate=(
                        f"{optimizer_name.upper()} - Initial point<br>"
                        "x: %{x:.5f}<br>"
                        "y: %{y:.5f}"
                        "<extra></extra>"
                    )
                ),

                row=1,
                col=1
            )

            # Final point
            fig.add_trace(

                go.Scatter(

                    x=[xs[-1]],
                    y=[ys[-1]],

                    mode="markers",

                    marker=dict(
                        symbol="star",
                        size=13,
                        color=color,
                        line=dict(
                            color="black",
                            width=1
                        )
                    ),

                    showlegend=False,

                    legendgroup=optimizer_name,

                    hovertemplate=(
                        f"{optimizer_name.upper()} - Final point<br>"
                        "x: %{x:.5f}<br>"
                        "y: %{y:.5f}"
                        "<extra></extra>"
                    )
                ),

                row=1,
                col=1
            )

            # --------------------------------------------------
            # LOSS CURVES
            # --------------------------------------------------

            fig.add_trace(

                go.Scatter(

                    x=np.arange(len(history["loss"])),

                    y=np.maximum(
                        history["loss"],
                        1e-16
                    ),

                    mode="lines",

                    line=dict(
                        color=color,
                        width=1.8
                    ),

                    name=f"{optimizer_name.upper()} | {initial_point}",

                    legendgroup=optimizer_name,

                    showlegend=False,

                    hovertemplate=(
                        "Iteration: %{x}<br>"
                        "Loss: %{y:.6e}"
                        "<extra></extra>"
                    )

                ),

                row=2,
                col=1
            )

            first_trace = False

    # --------------------------------------------------
    # GLOBAL MINIMUM
    # --------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=[optimum[0]],
            y=[optimum[1]],

            mode="markers",

            marker=dict(
                symbol="x",
                size=16,
                color="red",
                line=dict(width=3)
            ),

            name="Global minimum",

            hovertemplate=(
                "Global minimum<br>"
                "x: %{x}<br>"
                "y: %{y}"
                "<extra></extra>"
            )

        ),

        row=1,
        col=1
    )

    # --------------------------------------------------
    # LAYOUT
    # --------------------------------------------------

    fig.update_layout(

        title=dict(
            text=f"{benchmark_name} — Optimizer Benchmark",
            x=0.5,
            xanchor="center"
        ),

        template="plotly_white",

        height=950,

        width=1250,

        hovermode="closest",

        legend=dict(
            title="Optimizers",
            groupclick="togglegroup",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        margin=dict(
            l=70,
            r=100,
            t=130,
            b=70
        )
    )

    # Axes
    fig.update_xaxes(
        title_text="x",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="y",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Iterations",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Loss",
        type="log",
        row=2,
        col=1
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    filename = f"{benchmark_name.lower()}_dashboard.html"

    fig.write_html(
        save_dir / filename,
        include_plotlyjs=True
    )

    fig.show()

    return fig