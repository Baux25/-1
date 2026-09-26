"""
Sorting algorithm time / energy experiment.

Produces two CSV files:

    raw_measurements.csv
        One row per (input, algorithm) measurement.

    ml_dataset.csv
        One row per input, with the fastest algorithm, the
        lowest-energy algorithm and the hybrid choice attached.

Design notes
============

1. All four sorting algorithms are implemented in pure Python.

   An earlier version used ``heapq`` for Heap Sort. ``heapq`` is a
   C implementation, so it was roughly 9x faster than the other
   three Python implementations. That is not a comparison of
   algorithms, it is a comparison of Python against C, and it made
   Heap Sort win almost every input. Every algorithm here now runs
   in the same language, in the same environment, at the same
   level of abstraction.

2. The input generators have graded parameters.

   Previously every "nearly sorted" input had exactly size / 20
   swaps, and every "duplicate heavy" input drew from the same ten
   values. The model could therefore identify the input type from
   the features and nothing more. Each generator now sweeps a set
   of parameter levels, so one input type produces inputs of
   varying degrees.

3. There are many independent inputs per (type, size).

   7 types x 4 sizes x 30 inputs = 840 inputs, which gives the
   models something to generalise over.

4. The warm-up runs once per algorithm, not once per measurement.

   The warm-up exists to let the interpreter settle. Repeating it
   for all 3,360 measurements doubled the cost of the experiment
   for no benefit. Repetitions are still used to average out RAPL
   noise, but they are counted per input size.
"""

import argparse
import csv
import os
import random
import time

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

SIZES = [
    100,
    1_000,
    10_000,
    100_000
]

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


# Number of independent inputs generated for every
# (input_type, size) combination.

DATASETS_PER_TYPE = 30


# Number of measured repetitions.
#
# Small inputs are repeated many times because a single sort is
# far below the RAPL noise floor. Large inputs are repeated once
# because insertion sort is quadratic and cannot afford it.

REPETITIONS = {
    100: 200,
    1_000: 20,
    10_000: 3,
    100_000: 1
}


# Parameter levels swept by the generators.
#
# The variant index of an input selects a level, so the inputs of
# one type cover every degree of that condition.

SORTED_DISRUPTION = [
    0.0,
    0.0,
    0.0,
    0.001,
    0.002
]

NEARLY_SORTED_DISORDER = [
    0.01,
    0.02,
    0.05,
    0.10,
    0.20
]

MULTI_RUN_COUNTS = [
    2,
    5,
    10,
    20,
    50
]

DUPLICATE_RATIOS = [
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.99
]

SKEW_OUTLIER_SHARES = [
    0.05,
    0.10,
    0.20,
    0.50,
    0.80
]

VALUE_RANGE_CEILINGS = [
    1_000,
    100_000,
    1_000_000,
    10_000_000
]


# ============================================================
# SORTING ALGORITHMS
#
# All four are pure Python and sort in place or on a copy.
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
    """
    Pure Python Heap Sort.

    Replaces the previous version, which delegated to ``heapq``.
    ``heapq`` is implemented in C and was about 9x faster than the
    other three algorithms, which made Heap Sort win regardless of
    the input.
    """

    arr = arr.copy()

    n = len(arr)

    def sift_down(heap_size, root):

        largest = root

        left = 2 * root + 1
        right = 2 * root + 2

        if (
            left < heap_size
            and arr[left] > arr[largest]
        ):
            largest = left

        if (
            right < heap_size
            and arr[right] > arr[largest]
        ):
            largest = right

        if largest != root:

            arr[root], arr[largest] = (
                arr[largest],
                arr[root]
            )

            sift_down(heap_size, largest)

    # Build the max heap.

    for root in range(
        n // 2 - 1,
        -1,
        -1
    ):
        sift_down(n, root)

    # Move the largest remaining element to the end,
    # then restore the heap over the smaller prefix.

    for end in range(
        n - 1,
        0,
        -1
    ):

        arr[0], arr[end] = (
            arr[end],
            arr[0]
        )

        sift_down(end, 0)

    return arr


SORT_FUNCTIONS = {
    "insertion": insertion_sort,
    "merge": merge_sort,
    "quick3": three_way_quick_sort,
    "heap": heap_sort
}


# ============================================================
# DATA GENERATORS
#
# Every generator receives a variant index and returns
# (data, parameters) so the parameters end up in the CSV.
# ============================================================

def random_data(rng, size, variant):

    ceiling = VALUE_RANGE_CEILINGS[
        variant % len(VALUE_RANGE_CEILINGS)
    ]

    data = [
        rng.randint(0, ceiling)
        for _ in range(size)
    ]

    return data, {
        "value_ceiling": ceiling
    }


def sorted_data(rng, size, variant):
    """
    Ascending input.

    The run structure of a perfectly sorted input is the
    definition of this input type, so it is kept exact. Only a
    very small number of inputs receive a few random swaps, which
    stops the type from being perfectly degenerate.
    """

    ceiling = VALUE_RANGE_CEILINGS[
        variant % len(VALUE_RANGE_CEILINGS)
    ]

    disruption = SORTED_DISRUPTION[
        variant % len(SORTED_DISRUPTION)
    ]

    data = sorted(
        rng.randint(0, ceiling)
        for _ in range(size)
    )

    swaps = int(size * disruption)

    for _ in range(swaps):

        i = rng.randrange(size)
        j = rng.randrange(size)

        data[i], data[j] = data[j], data[i]

    return data, {
        "value_ceiling": ceiling,
        "disruption": disruption
    }


def reverse_sorted_data(rng, size, variant):
    """
    Descending input.

    As with the sorted input, the run structure is the definition
    of the type and is kept exact.
    """

    ceiling = VALUE_RANGE_CEILINGS[
        variant % len(VALUE_RANGE_CEILINGS)
    ]

    disruption = SORTED_DISRUPTION[
        variant % len(SORTED_DISRUPTION)
    ]

    data = sorted(
        (
            rng.randint(0, ceiling)
            for _ in range(size)
        ),
        reverse=True
    )

    swaps = int(size * disruption)

    for _ in range(swaps):

        i = rng.randrange(size)
        j = rng.randrange(size)

        data[i], data[j] = data[j], data[i]

    return data, {
        "value_ceiling": ceiling,
        "disruption": disruption
    }


def nearly_sorted_data(rng, size, variant):
    """
    Ascending input with a graded fraction of
    randomly swapped positions.
    """

    ceiling = VALUE_RANGE_CEILINGS[
        variant % len(VALUE_RANGE_CEILINGS)
    ]

    disorder = NEARLY_SORTED_DISORDER[
        variant % len(NEARLY_SORTED_DISORDER)
    ]

    data = sorted(
        rng.randint(0, ceiling)
        for _ in range(size)
    )

    swaps = max(1, int(size * disorder))

    for _ in range(swaps):

        i = rng.randrange(size)
        j = rng.randrange(size)

        data[i], data[j] = data[j], data[i]

    return data, {
        "value_ceiling": ceiling,
        "disorder": disorder
    }


def multi_run_data(rng, size, variant):
    """
    Input divided into a graded number of
    independently ascending runs.
    """

    ceiling = VALUE_RANGE_CEILINGS[
        variant % len(VALUE_RANGE_CEILINGS)
    ]

    run_count = MULTI_RUN_COUNTS[
        variant % len(MULTI_RUN_COUNTS)
    ]

    data = [
        rng.randint(0, ceiling)
        for _ in range(size)
    ]

    run_count = max(1, min(run_count, size))

    run_size = max(1, size // run_count)

    for start in range(0, size, run_size):

        end = min(start + run_size, size)

        data[start:end] = sorted(
            data[start:end]
        )

    return data, {
        "value_ceiling": ceiling,
        "run_count": run_count
    }


def duplicate_heavy_data(rng, size, variant):
    """
    Input drawn from a small pool of distinct values,
    so the duplicate ratio follows a graded target.
    """

    target_ratio = DUPLICATE_RATIOS[
        variant % len(DUPLICATE_RATIOS)
    ]

    distinct_count = max(
        1,
        int(round(size * (1.0 - target_ratio)))
    )

    values = rng.sample(
        range(10_000_000),
        distinct_count
    )

    data = [
        rng.choice(values)
        for _ in range(size)
    ]

    return data, {
        "target_duplicate_ratio": target_ratio,
        "distinct_values": distinct_count
    }


def skewed_data(rng, size, variant):
    """
    Input where a graded share of the elements
    sits far above the bulk of the distribution.
    """

    outlier_share = SKEW_OUTLIER_SHARES[
        variant % len(SKEW_OUTLIER_SHARES)
    ]

    data = []

    for _ in range(size):

        if rng.random() < outlier_share:

            data.append(
                rng.randint(
                    1_000_000,
                    10_000_000
                )
            )

        else:

            data.append(
                rng.randint(0, 1_000)
            )

    return data, {
        "outlier_share": outlier_share
    }


DATA_GENERATORS = {
    "random": random_data,
    "sorted": sorted_data,
    "reverse": reverse_sorted_data,
    "nearly_sorted": nearly_sorted_data,
    "multi_run": multi_run_data,
    "duplicate_heavy": duplicate_heavy_data,
    "skewed": skewed_data
}


# Columns that record the generator parameters.
# They are stored for inspection but are not used as
# machine learning features.

PARAMETER_COLUMNS = [
    "value_ceiling",
    "disruption",
    "disorder",
    "run_count",
    "target_duplicate_ratio",
    "distinct_values",
    "outlier_share"
]


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


def read_rapl(path):

    try:

        with open(path, "r") as f:

            return int(
                f.read().strip()
            )

    except PermissionError:

        print()
        print(
            "ERROR: RAPL permission denied."
        )

        print(
            "Reading the RAPL energy counter "
            "requires root privileges."
        )

        print(
            "Run the program with sudo."
        )

        raise SystemExit(1)

    except FileNotFoundError:

        print()
        print(
            f"ERROR: {path} was not found."
        )

        print(
            "Intel RAPL is not available on "
            "this machine, or the path is wrong."
        )

        print(
            "Pass a different path with "
            "--rapl-path if needed."
        )

        raise SystemExit(1)

    except OSError as error:

        print()
        print(
            f"ERROR: could not read {path}."
        )

        print(error)

        raise SystemExit(1)


# ============================================================
# MEASURE SORT
# ============================================================

def warm_up(rapl_path):

    """
    Run each algorithm once before any measurement.

    Done once for the whole experiment rather than once per
    measurement.
    """

    sample = list(range(2_000))

    for name in ALGORITHMS:

        SORT_FUNCTIONS[name](
            sample.copy()
        )

    read_rapl(rapl_path)


def measure_sort(
    algorithm_name,
    data,
    repetitions,
    rapl_path
):

    sort_function = SORT_FUNCTIONS[
        algorithm_name
    ]


    # --------------------------------------------------------
    # Start measurements
    # --------------------------------------------------------

    start_energy = read_rapl(
        rapl_path
    )

    start_time = time.perf_counter()


    for _ in range(repetitions):

        # Copying ensures that every repetition
        # receives the same unsorted input.

        test_data = data.copy()

        result = sort_function(test_data)

        # The sorted result is never used, but keeping the
        # reference alive until here stops the call from
        # being optimised away and makes the intent clear.

        del result


    end_time = time.perf_counter()

    end_energy = read_rapl(
        rapl_path
    )


    # --------------------------------------------------------
    # Calculate results
    # --------------------------------------------------------

    total_time = end_time - start_time

    total_energy_uj = (
        end_energy - start_energy
    )

    if total_energy_uj < 0:
        raise RuntimeError(
            "RAPL counter went backwards. "
            "The counter may have wrapped around."
        )


    # Convert microjoules to joules.

    total_energy_j = (
        total_energy_uj / 1_000_000
    )


    # Average per sorting operation.

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

def run_experiment(
    sizes,
    data_types,
    datasets_per_type,
    repetitions,
    rapl_path,
    raw_csv
):

    random.seed(SEED)

    rows = []

    input_id = 0

    total_inputs = (
        len(data_types)
        * len(sizes)
        * datasets_per_type
    )

    total_experiments = (
        total_inputs * len(ALGORITHMS)
    )

    print()
    print(
        f"Inputs:  {total_inputs}"
    )

    print(
        f"Rows:    {total_experiments}"
    )


    warm_up(rapl_path)


    for data_type in data_types:

        for size in sizes:

            print()
            print("=" * 60)

            print(
                f"Data type: {data_type}"
            )

            print(
                f"Size: {size}"
            )

            print("=" * 60)

            generator = DATA_GENERATORS[
                data_type
            ]

            repeat_count = repetitions.get(
                size,
                1
            )

            for variant in range(
                datasets_per_type
            ):

                # Each input gets its own generator so the
                # inputs are independent but still reproducible.

                input_seed = (
                    SEED * 1_000_000
                    + input_id
                )

                rng = random.Random(
                    input_seed
                )

                data, parameters = generator(
                    rng,
                    size,
                    variant
                )


                # Features are calculated once per input,
                # not once per algorithm.

                features = calculate_features(
                    data
                )

                input_id += 1

                for algorithm in ALGORITHMS:

                    print(
                        f"[{len(rows) + 1}/"
                        f"{total_experiments}] "
                        f"{data_type} "
                        f"n={size} "
                        f"v={variant} "
                        f"{algorithm}"
                    )

                    average_time, average_energy = (
                        measure_sort(
                            algorithm,
                            data,
                            repeat_count,
                            rapl_path
                        )
                    )

                    row = {

                        "input_id": input_id,

                        "input_type": data_type,

                        "variant": variant,

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

                        "repetitions":
                            repeat_count,

                        "execution_time":
                            average_time,

                        "energy":
                            average_energy
                    }

                    for column in PARAMETER_COLUMNS:

                        row[column] = parameters.get(
                            column
                        )

                    rows.append(row)


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df = pd.DataFrame(rows)

    df.to_csv(
        raw_csv,
        index=False
    )


    print()
    print(
        f"Saved: {raw_csv}"
    )

    print(
        f"Rows: {len(df)}"
    )


# ============================================================
# CREATE ML DATASET
# ============================================================

def create_ml_dataset(
    raw_csv,
    ml_csv,
    time_tolerance=0.05
):

    df = pd.read_csv(
        raw_csv
    )

    ml_rows = []

    print()
    print(
        "Building the ML dataset"
    )

    print(
        f"Tolerance band: "
        f"{time_tolerance:.0%}"
    )


    # Process one input at a time.

    for input_id, group in df.groupby(
        "input_id"
    ):

        first = group.iloc[0]

        row = {

            "input_id":
                input_id,

            "input_type":
                first["input_type"],

            "variant":
                first["variant"],

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

        row["performance_label"] = (
            group.loc[
                fastest_index,
                "algorithm"
            ]
        )


        # ----------------------------------------------------
        # ENERGY LABEL
        # Lowest energy
        # ----------------------------------------------------

        lowest_energy_index = (
            group["energy"]
            .idxmin()
        )

        row["energy_label"] = (
            group.loc[
                lowest_energy_index,
                "algorithm"
            ]
        )


        # ----------------------------------------------------
        # HYBRID LABEL
        #
        # Lowest energy among the algorithms that
        # are within the tolerance band of the
        # fastest algorithm.
        # ----------------------------------------------------

        fastest_time = (
            group["execution_time"]
            .min()
        )

        time_limit = (
            fastest_time
            * (1.0 + time_tolerance)
        )

        eligible = group[
            group["execution_time"]
            <= time_limit
        ]

        hybrid_index = (
            eligible["energy"]
            .idxmin()
        )

        row["hybrid_label"] = (
            eligible.loc[
                hybrid_index,
                "algorithm"
            ]
        )


        # ----------------------------------------------------
        # MARGINS
        #
        # Recorded so that near ties can be identified.
        # A tiny margin means the label is decided by
        # measurement noise rather than by a real
        # difference in the algorithms.
        # ----------------------------------------------------

        fastest_time = (
            group["execution_time"]
            .min()
        )

        slowest_time = (
            group["execution_time"]
            .max()
        )

        row["time_margin"] = (
            1.0 - (fastest_time / slowest_time)
            if slowest_time > 0
            else 0.0
        )

        lowest_energy = (
            group["energy"]
            .min()
        )

        highest_energy = (
            group["energy"]
            .max()
        )

        row["energy_margin"] = (
            1.0 - (lowest_energy / highest_energy)
            if highest_energy > 0
            else 0.0
        )


        ml_rows.append(row)


    # --------------------------------------------------------
    # Save ML dataset
    # --------------------------------------------------------

    ml_df = pd.DataFrame(
        ml_rows
    )

    ml_df.to_csv(
        ml_csv,
        index=False
    )


    print()
    print(
        f"Saved: {ml_csv}"
    )

    print(
        f"Rows: {len(ml_df)}"
    )


    # --------------------------------------------------------
    # Report how distinct the three labels actually are.
    # --------------------------------------------------------

    print()
    print("Label agreement")

    print(
        f"  performance == energy:  "
        f"{(ml_df['performance_label'] == ml_df['energy_label']).mean():.1%}"
    )

    print(
        f"  performance == hybrid:  "
        f"{(ml_df['performance_label'] == ml_df['hybrid_label']).mean():.1%}"
    )

    print(
        f"  energy == hybrid:       "
        f"{(ml_df['energy_label'] == ml_df['hybrid_label']).mean():.1%}"
    )

    print()
    print("Label distribution")

    for column in [
        "performance_label",
        "energy_label",
        "hybrid_label"
    ]:

        counts = ml_df[
            column
        ].value_counts()

        summary = ", ".join(
            f"{name}={count}"
            for name, count in counts.items()
        )

        print(
            f"  {column:20s} {summary}"
        )


# ============================================================
# MAIN
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Measure the time and energy of four "
            "pure Python sorting algorithms."
        )
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Small run for checking the pipeline. "
            "Writes to separate CSV files so the "
            "real results are not overwritten."
        )
    )

    parser.add_argument(
        "--datasets",
        type=int,
        default=DATASETS_PER_TYPE,
        help=(
            "Independent inputs per "
            "(input_type, size). Default: "
            f"{DATASETS_PER_TYPE}"
        )
    )

    parser.add_argument(
        "--rapl-path",
        default=RAPL_PATH,
        help=(
            "Path to the RAPL energy counter. "
            f"Default: {RAPL_PATH}"
        )
    )

    parser.add_argument(
        "--raw-csv",
        default="raw_measurements.csv",
        help=(
            "Output path for the raw "
            "measurements."
        )
    )

    parser.add_argument(
        "--ml-csv",
        default="ml_dataset.csv",
        help=(
            "Output path for the ML dataset."
        )
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    sizes = list(SIZES)
    data_types = list(DATA_TYPES)
    datasets_per_type = args.datasets
    repetitions = dict(REPETITIONS)

    raw_csv = args.raw_csv
    ml_csv = args.ml_csv


    if args.quick:

        sizes = [100, 1_000]
        datasets_per_type = 3

        repetitions = {
            100: 20,
            1_000: 5
        }

        raw_csv = "raw_measurements_quick.csv"
        ml_csv = "ml_dataset_quick.csv"

        print()
        print(
            "QUICK MODE: sizes "
            f"{sizes}, "
            f"{datasets_per_type} inputs per type"
        )

        print(
            f"Writing to {raw_csv} "
            f"and {ml_csv}"
        )


    print()
    print(
        "SORTING TIME AND ENERGY EXPERIMENT"
    )

    print()
    print(
        "All four algorithms are pure Python."
    )


    run_experiment(
        sizes=sizes,
        data_types=data_types,
        datasets_per_type=datasets_per_type,
        repetitions=repetitions,
        rapl_path=args.rapl_path,
        raw_csv=raw_csv
    )

    create_ml_dataset(
        raw_csv=raw_csv,
        ml_csv=ml_csv
    )

    print()
    print(
        "Experiment completed."
    )


if __name__ == "__main__":

    main()
