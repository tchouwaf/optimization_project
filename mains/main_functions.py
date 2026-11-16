import torch

from models.functions import OptimizationModel
from models.functions import rosenbrock, griewank

from process.functions_processor import make_optimizers, train
from process.visualization import plot_benchmark_dashboard


SEED = 88
EPOCHS = 2000

torch.manual_seed(SEED)


# --------------------------------------------------
# BENCHMARKS
# --------------------------------------------------

BENCHMARKS = {

    "Rosenbrock": {
        "function": rosenbrock,
        "optimum": (1.0, 1.0),
    },

    "Griewank": {
        "function": griewank,
        "optimum": (0.0, 0.0),
    }
}


# --------------------------------------------------
# INITIAL POINTS
# --------------------------------------------------

INITIAL_POINTS = {

    "Rosenbrock": [
        # (-2.0, 2.0),
        (1.75, 3),
        # (-1.0, 1.0),
        # (0.0, 2.0),
        # (2.0, 2.0),
        # (-1.5, 1.5),
        # (1.5, -1.0),
    ],

    "Griewank": [
        (-5.0, 5.0),
        # (-5.0, -5.0),
        # (5.0, 5.0),
        # (5.0, -5.0),
        # (10.0, 10.0),
        # (-10.0, 10.0),
        # (2.0, -7.0),
    ]
}


# --------------------------------------------------
# LEARNING RATES
# --------------------------------------------------

LEARNING_RATES = {

    "Rosenbrock": {
        "sgd": 0.001,
        "momentum": 0.0005,
        "nesterov": 0.0001,
        "adam": 0.01,
        "sign": 0.001,
        "muon": 0.01,
    },

    "Griewank": {
        "sgd": 2.0,
        "momentum": 2.0,
        "nesterov": 2.0,
        "adam": 2.0,
        "sign": 3.0,
        "muon": 2.0,
    }
}


OPTIMIZERS_NAMES = [
    "sgd",
    "momentum",
    "nesterov",
    "adam",
    "sign",
    "muon"
]


LR_DECAY = {

    "Rosenbrock": {
        "step_size": 500,
        "gamma": 0.01
    },

    "Griewank": {
        "step_size": 100,
        "gamma": 0.1
    }
}

# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main_functions():

    results = {}

    for benchmark_name, benchmark in BENCHMARKS.items():

        objective = benchmark["function"]
        optimum = benchmark["optimum"]

        learning_rates = LEARNING_RATES[benchmark_name]

        results[benchmark_name] = {}

        print(f"\n========== {benchmark_name} ==========")

        # ----------------------------------
        # TRAINING
        # ----------------------------------

        for initial_point in INITIAL_POINTS[benchmark_name]:

            print(f"\nInitial point: {initial_point}")

            results[benchmark_name][initial_point] = {}

            for optimizer_name in OPTIMIZERS_NAMES:

                model = OptimizationModel(initial_point)

                optimizers = make_optimizers(
                    model,
                    learning_rates
                )

                optimizer = optimizers[optimizer_name]

                lr = learning_rates[optimizer_name]

                result = train(
                    model=model,
                    optimizer=optimizer,
                    objective=objective,
                    optimum=optimum,
                    epochs=EPOCHS,
                    lr_decay=LR_DECAY[benchmark_name]
                )

                results[benchmark_name][initial_point][optimizer_name] = result

                print(
                    f"{optimizer_name:10s} | "
                    f"LR: {lr:.2e} | "
                    f"Loss: {result['final_loss']:.8e} | "
                    f"Distance: {result['final_distance']:.6f} | "
                    f"Iterations: {result['iterations']:5d} | "
                    f"Time: {result['time']:.4f}s"
                )

        # ----------------------------------
        # VISUALIZATION
        # ----------------------------------

        plot_benchmark_dashboard(
            benchmark_name=benchmark_name,
            results=results[benchmark_name],
            initial_points=INITIAL_POINTS[benchmark_name],
            learning_rates=learning_rates,
            optimum=optimum
        )

    return results