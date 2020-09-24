import argparse
import os
import os.path as osp
import time
import timeit
import tqdm

import numpy as np
import pandas as pd
from sort import bubblesort, quicksort, timsort

from algo_utils.config import system_config


def const_(vector):
    return 1


def sum_(vector):
    return np.sum(vector)


def prod_(vector):
    return np.prod(vector)


def direct_poly(vector, x_val=1.5):
    x_values = []
    poly_output = 0
    for degree in range(len(vector)):
        if degree == 0:
            x_values.append(1)
        else:
            x_values.append(x_values[-1] * x_val)
        poly_output += x_values[-1] * vector[degree]
    return poly_output


def horners_poly(vector, x_val=1.5):
    poly_output = 0
    for degree in list(range(len(vector)))[::-1]:
        if degree == len(vector) - 1:
            poly_output += vector[degree]
        else:
            poly_output += x_val * poly_output + vector[degree]
    return poly_output


def _filter_times(times):
    times = np.asarray(times)
    times = times[(times != times.max()) & (times != times.min())]
    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather data for task 1")
    parser.add_argument("--output_file", default=osp.join(system_config.data_dir, "task1.csv"), help="Output file")
    parser.add_argument("--random_state", type=int, default=24, help="Random state for random generator")
    args = parser.parse_args()

    np.random.seed(args.random_state)
    os.makedirs(osp.dirname(args.output_file), exist_ok=True)

    data = []
    for n in tqdm.tqdm(range(1, 2001), desc="Iterate over n"):
        vectors = np.split(np.random.uniform(low=0, high=100, size=n * 7), 7)
        current_sample = {"n": n}
        for operation in [
            "const_",
            "sum_",
            "prod_",
            "direct_poly",
            "horners_poly",
            "bubblesort.sort",
            "quicksort.sort",
            "timsort.sort",
        ]:
            times = []
            for vector in vectors:
                vector = vector.tolist()
                times.append(timeit.timeit(stmt=f"{operation}(vector)", globals=globals(), number=1))
            current_sample[operation] = _filter_times(times).mean()

        matrices = np.split(np.random.uniform(low=0, high=100, size=(2 * 7, n, n)), 7, axis=0)
        times = []
        for matrix in matrices:
            times.append(timeit.timeit(stmt="matrix[0].dot(matrix[1])", globals=globals(), number=1))
        current_sample["matrix_mult"] = _filter_times(times).mean()

        data.append(current_sample)
        if (n % 100) == 0:
            df = pd.DataFrame(data)
            df.to_csv(args.output_file, index=False)

    df = pd.DataFrame(data)
    df.to_csv(args.output_file, index=False)
