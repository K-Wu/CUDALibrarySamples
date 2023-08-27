import gspread
import os
from upload_to_gspread import update_gspread, open_worksheet

if __name__ == "__main__":
    with open(
        "/home/" + os.getlogin() + "/.config/gspread/gpu_microbenchmark.url"
    ) as fd:
        url = fd.readlines()[0].strip()
    print(url)
    ws = open_worksheet(url, "0")
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
        ],
        ws,
    )
