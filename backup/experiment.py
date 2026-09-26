import random
import time
import csv
import os
import subprocess
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SIZES = [100, 1_000, 10_000, 100_000]

DATA_TYPES = [
    "random",
    "sorted",
    "reverse",
    "nearly_sorted",
    "multi_run",
    "duplicate_heavy",
    "skewed"
]

ALGORITHMS = [
    "insertion",
    "merge",
    "quick3",
    "heap"
]

SEED = 42

RAW_CSV = "raw_measurements.csv"
ML_CSV = "ml_dataset.csv"


# Number of repetitions.
# Larger inputs need fewer repetitions because they take longer.
REPETITIONS = {
    100: 1000,
    1_000: 100,
    10_000: 10,
    100_000: 1
}


# ============================================================
# SORTING ALGORITHMS
# ============================================================

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr


def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2

    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    result = []

    i = 0
    j = 0

    while i < len(left) and j < len(right):

        if left[i] <= right[j]:
            result.append(left[i])
            i += 1

        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def three_way_quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]

    less = []
    equal = []
    greater = []

    for value in arr:

        if value < pivot:
            less.append(value)

        elif value > pivot:
            greater.append(value)

        else:
            equal.append(value)

    return (
        three_way_quick_sort(less)
        + equal
        + three_way_quick_sort(greater)
    )


def heap_sort(arr):
    import heapq

    heap = arr.copy()

    heapq.heapify(heap)

    result = []

    while heap:
        result.append(heapq.heappop(heap))

    return result


SORT_FUNCTIONS = {
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick3": three_way_quick_sort,
    "heap": heap_sort
}


# ============================================================
# DATA GENERATORS
# ============================================================

def random_data(size):
    return [
        random.randint(0, 1_000_000)
        for _ in range(size)
    ]


def sorted_data(size):
    return sorted(random_data(size))


def reverse_sorted_data(size):
    return sorted(random_data(size), reverse=True)


def nearly_sorted_data(size):
    data = sorted(random_data(size))

    changes = max(1, size // 20)

    for _ in range(changes):

        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)

        data[i], data[j] = data[j], data[i]

    return data


def multi_run_data(size):
    data = random_data(size)

    run_size = max(1, size // 5)

    for start in range(0, size, run_size):

        end = min(start + run_size, size)

        data[start:end] = sorted(
            data[start:end]
        )

    return data


def duplicate_heavy_data(size):

    values = [
        10, 20, 30, 40, 50,
        60, 70, 80, 90, 100
    ]

    return [
        random.choice(values)
        for _ in range(size)
    ]


def skewed_data(size):

    data = []

    for _ in range(size):

        if random.random() < 0.8:

            data.append(
                random.randint(0, 1_000)
            )

        else:

            data.append(
                random.randint(
                    1_000_000,
                    10_000_000
                )
            )

    return data


DATA_GENERATORS = {
    "random": random_data,
    "sorted": sorted_data,
    "reverse": reverse_sorted_data,
    "nearly_sorted": nearly_sorted_data,
    "multi_run": multi_run_data,
    "duplicate_heavy": duplicate_heavy_data,
    "skewed": skewed_data
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def calculate_features(data):

    size = len(data)

    if size <= 1:

        return {
            "runs": 1,
            "longest_run": size,
            "duplicate_ratio": 0.0,
            "value_range": 0
        }


    # --------------------------------------------------------
    # Ascending runs
    # --------------------------------------------------------

    runs = 1
    current_run = 1
    longest_run = 1

    for i in range(1, size):

        if data[i] >= data[i - 1]:

            current_run += 1

        else:

            runs += 1
            current_run = 1

        longest_run = max(
            longest_run,
            current_run
        )


    # --------------------------------------------------------
    # Duplicate ratio
    #
    # 0 = all values unique
    # 1 = all values identical
    # --------------------------------------------------------

    unique_values = len(set(data))

    duplicate_ratio = (
        1 - (unique_values / size)
    )


    # --------------------------------------------------------
    # Value range
    # --------------------------------------------------------

    value_range = max(data) - min(data)


    return {
        "runs": runs,
        "longest_run": longest_run,
        "duplicate_ratio": duplicate_ratio,
        "value_range": value_range
    }


# ============================================================
# INTEL RAPL ENERGY
# ============================================================

RAPL_PATH = (
    "/sys/class/powercap/"
    "intel-rapl:0/energy_uj"
)


def read_rapl():

    try:

        with open(RAPL_PATH, "r") as f:

            return int(
                f.read().strip()
            )

    except PermissionError:

        print(
            "ERROR: RAPL permission denied."
        )

        print(
            "Run the program with sudo."
        )

        raise

    except FileNotFoundError:

        print(
            "ERROR: Intel RAPL was not found."
        )

        raise


# ============================================================
# MEASURE SORT
# ============================================================

def measure_sort(
    algorithm_name,
    data,
    repetitions
):

    sort_function = SORT_FUNCTIONS[
        algorithm_name
    ]


    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    sort_function(data.copy())


    # --------------------------------------------------------
    # Start measurements
    # --------------------------------------------------------

    start_energy = read_rapl()

    start_time = time.perf_counter()


    for _ in range(repetitions):

        # Copying ensures that every repetition
        # receives the same unsorted input.

        test_data = data.copy()

        result = sort_function(test_data)


    end_time = time.perf_counter()

    end_energy = read_rapl()


    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------

    total_time = end_time - start_time

    total_energy_uj = (
        end_energy - start_energy
    )


    # Convert microjoules → joules

    total_energy_j = (
        total_energy_uj / 1_000_000
    )


    # Average per sorting operation

    average_time = (
        total_time / repetitions
    )

    average_energy = (
        total_energy_j / repetitions
    )


    return (
        average_time,
        average_energy
    )


# ============================================================
# CREATE RAW CSV
# ============================================================

def run_experiment():

    random.seed(SEED)

    rows = []

    input_id = 0


    total_experiments = (
        len(DATA_TYPES)
        * len(SIZES)
        * len(ALGORITHMS)
    )

    experiment_number = 0


    for data_type in DATA_TYPES:

        for size in SIZES:

            print()
            print("=" * 60)
            print(
                f"Data type: {data_type}"
            )
            print(
                f"Size: {size}"
            )
            print("=" * 60)


            # Generate ONE input.

            generator = DATA_GENERATORS[
                data_type
            ]

            data = generator(size)


            # Calculate features ONCE.

            features = calculate_features(
                data
            )


            input_id += 1


            repetitions = REPETITIONS[
                size
            ]


            for algorithm in ALGORITHMS:

                experiment_number += 1

                print(
                    f"[{experiment_number}/"
                    f"{total_experiments}] "
                    f"{algorithm}"
                )


                average_time, average_energy = (
                    measure_sort(
                        algorithm,
                        data,
                        repetitions
                    )
                )


                row = {

                    "input_id": input_id,

                    "input_type": data_type,

                    "size": size,

                    "runs": features["runs"],

                    "longest_run":
                        features["longest_run"],

                    "duplicate_ratio":
                        features["duplicate_ratio"],

                    "value_range":
                        features["value_range"],

                    "algorithm":
                        algorithm,

                    "execution_time":
                        average_time,

                    "energy":
                        average_energy
                }


                rows.append(row)


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    df.to_csv(
        RAW_CSV,
        index=False
    )


    print()
    print(
        f"Saved: {RAW_CSV}"
    )

    print(
        f"Rows: {len(df)}"
    )


# ============================================================
# CREATE ML DATASET
# ============================================================

def create_ml_dataset():

    df = pd.read_csv(
        RAW_CSV
    )


    ml_rows = []


    # Process one input at a time.

    for input_id, group in df.groupby(
        "input_id"
    ):

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        first = group.iloc[0]


        row = {

            "input_id":
                input_id,

            "input_type":
                first["input_type"],

            "size":
                first["size"],

            "runs":
                first["runs"],

            "longest_run":
                first["longest_run"],

            "duplicate_ratio":
                first["duplicate_ratio"],

            "value_range":
                first["value_range"]
        }


        # ----------------------------------------------------
        # PERFORMANCE LABEL
        # Lowest execution time
        # ----------------------------------------------------

        fastest_index = (
            group["execution_time"]
            .idxmin()
        )

        fastest_algorithm = (
            group.loc[
                fastest_index,
                "algorithm"
            ]
        )


        row["performance_label"] = (
            fastest_algorithm
        )


        # ----------------------------------------------------
        # ENERGY LABEL
        # Lowest energy
        # ----------------------------------------------------

        lowest_energy_index = (
            group["energy"]
            .idxmin()
        )

        lowest_energy_algorithm = (
            group.loc[
                lowest_energy_index,
                "algorithm"
            ]
        )


        row["energy_label"] = (
            lowest_energy_algorithm
        )


        # ----------------------------------------------------
        # HYBRID LABEL
        #
        # Lowest energy among algorithms
        # within 5% of the fastest algorithm.
        # ----------------------------------------------------

        fastest_time = (
            group["execution_time"]
            .min()
        )


        time_limit = (
            fastest_time * 1.05
        )


        eligible = group[
            group["execution_time"]
            <= time_limit
        ]


        hybrid_index = (
            eligible["energy"]
            .idxmin()
        )


        hybrid_algorithm = (
            eligible.loc[
                hybrid_index,
                "algorithm"
            ]
        )


        row["hybrid_label"] = (
            hybrid_algorithm
        )


        ml_rows.append(row)


    # --------------------------------------------------------
    # Save ML dataset
    # --------------------------------------------------------

    ml_df = pd.DataFrame(
        ml_rows
    )


    ml_df.to_csv(
        ML_CSV,
        index=False
    )


    print()
    print(
        f"Saved: {ML_CSV}"
    )

    print(
        f"Rows: {len(ml_df)}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "SORTING ENERGY EXPERIMENT"
    )

    print()

    run_experiment()

    create_ml_dataset()

    print()
    print(
        "Experiment completed."
    )