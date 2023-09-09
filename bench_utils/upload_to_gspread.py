from .nsight_utils.upload_benchmark_results import (
    open_worksheet,
    update_gspread,
)
import os


def find_latest_subdirectory(root, prefix):
    candidates = []
    for subdir in os.listdir(root):
        if subdir.startswith(prefix):
            candidates.append(subdir)
    return os.path.join(root, max(candidates))


def extract_info_from(filename):
    # artifacts/benchmark_cusparseLt_spmm_202306160108/cusparseLt_spmm_32_16_32_--tune.txt
    info = filename[:-4].split("_")
    # skip the first few name components and start from the m, n, k
    for idx, entry in enumerate(info):
        if entry.isdigit():
            return info[idx:]


def extract_gemm_result(txt_file):
    with open(txt_file, "r") as f:
        "cublas<X>gemm time (ms): 0.252928"
        lines = f.readlines()
        for line in lines:
            if line.startswith("cublas"):
                return [line.split(":")[1].strip().split()[0]]
    # there is a runtime error
    return ["RuntimeError"]


def extract_cusparselt_result(txt_file):
    "first four lines are about m, n, k, tune_flag"
    " the time breakdown includes the following lines"
    """
    cusparseLtMatmul time breakdown: 
    handle creation: 50.964481 ms pruning: 0.029696 ms verification: 0.031872 ms compression: 0.063360 ms tuning: 29.173759 ms execution: 0.048128 ms destruction: 0.004096 ms total: 80.315392 ms
    """
    "a status line mentioning the benchmark passed, failed (runtime error), or wrong result"
    "the number of all the other lines could reflect amount of wrong results"
    # if it is empty then it suggests a segmentation fault
    with open(txt_file) as fd:
        (
            handle_creation_time,
            prune_time,
            verification_time,
            compression_time,
            tuning_time,
            workspace_time,
            execution_time,
            destruction_time,
            total_time,
        ) = (None, None, None, None, None, None, None, None, None)
        status = "RuntimeError"
        num_line = 0
        for line in fd:
            num_line += 1
            if line.find("handle creation:") != -1:
                handle_creation_time = line.split(":")[1].strip().split()[0]
            elif line.find("pruning:") != -1:
                prune_time = line.split(":")[1].strip().split()[0]
            elif line.find("verification:") != -1:
                verification_time = line.split(":")[1].strip().split()[0]
            elif line.find("compression:") != -1:
                compression_time = line.split(":")[1].strip().split()[0]
            elif line.find("tuning:") != -1:
                tuning_time = line.split(":")[1].strip().split()[0]
            elif line.find("workspace allocation:") != -1:
                workspace_time = line.split(":")[1].strip().split()[0]
            elif line.find("execution:") != -1:
                execution_time = line.split(":")[1].strip().split()[0]
            elif line.find("destruction:") != -1:
                destruction_time = line.split(":")[1].strip().split()[0]
            elif line.find("total:") != -1:
                total_time = line.split(":")[1].strip().split()[0]
            elif "FAILED" in line:
                status = line.strip()
            elif "PASSED" in line:
                status = "PASSED"
        if status != "PASSED":
            status += " total lines: {}".format(num_line)
        return [
            handle_creation_time,
            prune_time,
            verification_time,
            compression_time,
            tuning_time,
            workspace_time,
            execution_time,
            destruction_time,
            total_time,
            status,
        ]


def extract_results_from_folder(path, file_extraction_func):
    all_names_and_info = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            name_info = extract_info_from(filename)
            result_info = file_extraction_func(os.path.join(path, filename))
            all_names_and_info.append(name_info + result_info)
    return all_names_and_info


# TODO: remove the following code as they are now provided by nsight_utils

# def open_worksheet(target_sheet_url: str, target_gid: str):
#     if target_gid != "0":
#         raise NotImplementedError("To avoid data loss, only gid=0 is supported for now")
#     gc = gspread.service_account()
#     sh = gc.open_by_url(target_sheet_url)
#     sheet_data = sh.fetch_sheet_metadata()

#     try:
#         item = finditem(
#             lambda x: str(x["properties"]["sheetId"]) == target_gid,
#             sheet_data["sheets"],
#         )
#         ws = Worksheet(sh, item["properties"])
#     except (StopIteration, KeyError):
#         raise WorksheetNotFound(target_gid)
#     return ws


# def create_worksheet(target_sheet_url: str, title: str, retry=False) -> Worksheet:
#     gc = gspread.service_account()
#     sh = gc.open_by_url(target_sheet_url)
#     title_suffix = ""
#     # when retry is True, we will ask user to specify a suffix if the title already exists
#     if retry:
#         while True:
#             if (title + title_suffix)[:100] in [ws.title for ws in sh.worksheets()]:
#                 # ask user to specify a suffix
#                 title_suffix = input("title already exists, please specify a suffix:")
#             else:
#                 break

#     return sh.add_worksheet(title=title + title_suffix, rows=100, cols=20)


# def update_gspread(entries, ws: Worksheet, cell_range=None):
#     if cell_range is None:
#         # start from A1
#         cell_range = "A1:"
#         num_rows = len(entries)
#         num_cols = max([len(row) for row in entries])
#         cell_range += gspread.utils.rowcol_to_a1(num_rows, num_cols)

#     ws.update(cell_range, entries)


if __name__ == "__main__":
    with open(
        "/home/" + os.getlogin() + "/.config/gspread/gpu_microbenchmark.url"
    ) as fd:
        url = fd.readlines()[0].strip()
    print(url)

    # Upload cuspoarseLt results to google sheet
    ws = open_worksheet(url, "300213523")
    update_gspread(
        [
            [
                "m",
                "n",
                "k",
                "tune_flag",
                "create handle(ms)",
                "prune(ms)",
                "verify(ms)",
                "compress(ms)",
                "tune(ms)",
                "workspace alloc(ms)",
                "execute(ms)",
                "destroy handle(ms)",
                "total time(ms)",
                "status",
            ]
        ]
        + extract_results_from_folder(
            find_latest_subdirectory(
                "./artifacts", "benchmark_cusparseLt_spmm"
            ),
            extract_cusparselt_result,
        ),
        ws,
    )

    # Upload cublas results to google sheet
    ws2 = open_worksheet(url, "1193553658")
    update_gspread(
        [["m", "n", "k", "NO_OTHER_FLAG", "total time(ms)"]]
        + extract_results_from_folder(
            find_latest_subdirectory("./artifacts", "benchmark_gemm"),
            extract_gemm_result,
        ),
        ws2,
    )
