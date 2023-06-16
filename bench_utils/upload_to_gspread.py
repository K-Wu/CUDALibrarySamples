import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem

def update_gspread(entries, target_sheet_url, target_gid, cell_range=None):
    gc = gspread.service_account()
    sh = gc.open_by_url(target_sheet_url)
    sheet_data = sh.fetch_sheet_metadata()

    try:
        item = finditem(
            lambda x: str(x["properties"]["sheetId"]) == target_gid,
            sheet_data["sheets"],
        )
        ws = Worksheet(sh, item["properties"])
    except (StopIteration, KeyError):
        raise WorksheetNotFound(target_gid)

    if cell_range is None:
        # start from A1
        cell_range = "A1:"
        num_rows = len(entries)
        num_cols = max([len(row) for row in entries])
        cell_range += gspread.utils.rowcol_to_a1(num_rows, num_cols)

    ws.update(cell_range, entries)

if __name__ == "__main__":
    with open("/home/kunww/.config/gspread/gpu_microbenchmark.url") as fd:
        url = fd.readlines()[0].strip()
    print(url)
    update_gspread(
        [[1,2,3],[4,5,6],[7,8,9]],
        url, "0",
    )
