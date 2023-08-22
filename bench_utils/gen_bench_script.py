import os


def get_header(component_name):
    return f"""
OUTPUT_DIR="artifacts/benchmark_{component_name}_`date +%Y%m%d%H%M`"
mkdir -p $OUTPUT_DIR
"""


def gen_m_n_k(min, max):
    for m in range(min, max + 1):
        for n in range(m - 1, m + 2):
            for k in range(m - 1, m + 2):
                yield (2**m, 2**n, 2**k)


workloads = {
    "gemm": {
        "path": "cuBLAS/Level-3/bench_gemm/build/cublas_bench_gemm_example",
        "extra_flags": [""],
        "mnk": [*gen_m_n_k(5, 16)],
    },
    "cublasLt_spmm": {
        "path": "cuSPARSELt/matmul_bench/matmul_bench",
        "extra_flags": ["--tune", ""],
        "mnk": [*gen_m_n_k(5, 16)],
    },
}


# cuBLAS/Level-3/bench_gemm/build/cublas_bench_gemm_example
# cuSPARSELt/matmul_bench/build/cusparseLt_matmul_bench

if __name__ == "__main__":
    print([*gen_m_n_k(5, 18)])
    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")
    for workload in workloads:
        workload_info = workloads[workload]
        with open(
            "artifacts/generated_bench{workload}.sh".format(
                workload=workload), "w"
        ) as fd:
            fd.write(get_header(workload))
            for m, n, k in workload_info["mnk"]:
                for extra_flag in workload_info["extra_flags"]:
                    fd.write(
                        f"{workload_info['path']} --m={m} --n={n} --k={k} {extra_flag} > $OUTPUT_DIR/{workload}_{m}_{n}_{k}_{extra_flag}.txt\n"
                    )
