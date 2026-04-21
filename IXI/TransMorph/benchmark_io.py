import argparse
import glob
import os
import pickle
import random
import tempfile
import time
from statistics import mean


def format_mb_per_s(num_bytes, seconds):
    if seconds <= 0:
        return float("inf")
    return num_bytes / (1024 * 1024) / seconds


def percentile(values, ratio):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(int(round((len(sorted_values) - 1) * ratio)), len(sorted_values) - 1)
    return sorted_values[index]


def benchmark_raw_io(tmp_dir, file_size_mb, chunk_size_mb, keep_file):
    total_bytes = file_size_mb * 1024 * 1024
    chunk_bytes = chunk_size_mb * 1024 * 1024
    payload = os.urandom(chunk_bytes)

    with tempfile.NamedTemporaryFile(prefix="io_bench_", suffix=".bin", dir=tmp_dir, delete=False) as handle:
        target_path = handle.name

    write_start = time.perf_counter()
    with open(target_path, "wb", buffering=chunk_bytes) as handle:
        written = 0
        while written < total_bytes:
            current = min(chunk_bytes, total_bytes - written)
            handle.write(payload[:current])
            written += current
        handle.flush()
        os.fsync(handle.fileno())
    write_seconds = time.perf_counter() - write_start

    read_start = time.perf_counter()
    with open(target_path, "rb", buffering=chunk_bytes) as handle:
        while handle.read(chunk_bytes):
            pass
    read_seconds = time.perf_counter() - read_start

    result = {
        "path": target_path,
        "file_size_mb": file_size_mb,
        "chunk_size_mb": chunk_size_mb,
        "write_seconds": write_seconds,
        "write_mb_per_s": format_mb_per_s(total_bytes, write_seconds),
        "read_seconds": read_seconds,
        "read_mb_per_s": format_mb_per_s(total_bytes, read_seconds),
    }

    if not keep_file:
        os.remove(target_path)
    return result


def pkload(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def timed_pkload(path, repeats):
    latencies = []
    payload_sizes = []
    for _ in range(repeats):
        start = time.perf_counter()
        payload = pkload(path)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        try:
            payload_sizes.append(os.path.getsize(path))
        except OSError:
            pass
        del payload

    return {
        "path": path,
        "repeats": repeats,
        "mean_seconds": mean(latencies),
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "max_seconds": max(latencies),
        "file_size_mb": (payload_sizes[0] / (1024 * 1024)) if payload_sizes else 0.0,
    }


def benchmark_dataset_io(atlas_path, data_glob, sample_count, repeats):
    case_paths = sorted(glob.glob(data_glob))
    if not case_paths:
        raise FileNotFoundError(f"No files matched data_glob: {data_glob}")

    if sample_count < len(case_paths):
        rng = random.Random(0)
        case_paths = rng.sample(case_paths, sample_count)

    atlas_stats = timed_pkload(atlas_path, repeats)

    case_stats = []
    pair_latencies = []
    for case_path in case_paths:
        case_result = timed_pkload(case_path, repeats)
        case_stats.append(case_result)

        for _ in range(repeats):
            start = time.perf_counter()
            atlas_payload = pkload(atlas_path)
            case_payload = pkload(case_path)
            elapsed = time.perf_counter() - start
            pair_latencies.append(elapsed)
            del atlas_payload
            del case_payload

    return {
        "atlas": atlas_stats,
        "cases": case_stats,
        "pair_read_mean_seconds": mean(pair_latencies),
        "pair_read_p50_seconds": percentile(pair_latencies, 0.50),
        "pair_read_p95_seconds": percentile(pair_latencies, 0.95),
        "pair_read_max_seconds": max(pair_latencies),
        "sampled_case_count": len(case_paths),
        "case_mean_seconds": mean([entry["mean_seconds"] for entry in case_stats]),
        "case_p95_seconds": percentile([entry["mean_seconds"] for entry in case_stats], 0.95),
    }


def print_raw_io_summary(result):
    print("== Raw Sequential IO ==")
    print(f"temp file: {result['path']}")
    print(f"write: {result['write_seconds']:.3f}s, {result['write_mb_per_s']:.2f} MB/s")
    print(f"read : {result['read_seconds']:.3f}s, {result['read_mb_per_s']:.2f} MB/s")


def print_dataset_summary(result):
    print("== Pickle/Dataset IO ==")
    atlas = result["atlas"]
    print(
        "atlas: mean={:.4f}s p50={:.4f}s p95={:.4f}s size={:.2f} MB".format(
            atlas["mean_seconds"],
            atlas["p50_seconds"],
            atlas["p95_seconds"],
            atlas["file_size_mb"],
        )
    )
    print(
        "case : mean={:.4f}s p95={:.4f}s across {} sampled cases".format(
            result["case_mean_seconds"],
            result["case_p95_seconds"],
            result["sampled_case_count"],
        )
    )
    print(
        "atlas+case per sample: mean={:.4f}s p50={:.4f}s p95={:.4f}s max={:.4f}s".format(
            result["pair_read_mean_seconds"],
            result["pair_read_p50_seconds"],
            result["pair_read_p95_seconds"],
            result["pair_read_max_seconds"],
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark raw disk IO and IXI pickle loading speed.")
    parser.add_argument("--tmp-dir", default=".", help="Directory for the temporary raw IO test file.")
    parser.add_argument("--file-size-mb", type=int, default=1024, help="Temp file size for raw IO benchmark.")
    parser.add_argument("--chunk-size-mb", type=int, default=64, help="Read/write chunk size for raw IO benchmark.")
    parser.add_argument("--keep-temp-file", action="store_true", help="Keep the temporary raw IO file after benchmarking.")
    parser.add_argument("--atlas", default=None, help="Path to atlas.pkl for dataset IO benchmark.")
    parser.add_argument("--data-glob", default=None, help="Glob for case .pkl files, for example '/data/.../Train/*.pkl'.")
    parser.add_argument("--samples", type=int, default=8, help="Number of case files to sample from data_glob.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats per file.")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_result = benchmark_raw_io(
        tmp_dir=args.tmp_dir,
        file_size_mb=args.file_size_mb,
        chunk_size_mb=args.chunk_size_mb,
        keep_file=args.keep_temp_file,
    )
    print_raw_io_summary(raw_result)

    if args.atlas and args.data_glob:
        dataset_result = benchmark_dataset_io(
            atlas_path=args.atlas,
            data_glob=args.data_glob,
            sample_count=args.samples,
            repeats=args.repeats,
        )
        print_dataset_summary(dataset_result)
    else:
        print("dataset benchmark skipped: provide both --atlas and --data-glob")


if __name__ == "__main__":
    main()
