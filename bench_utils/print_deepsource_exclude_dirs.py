# please run this script in the root directory of the project
import os

if __name__ == "__main__":
    for item in set([x[0] for x in os.walk(".")]):
        # skip 3rdparty, .git, and .vscode
        if (
            item.startswith("./3rdparty")
            or item.startswith("./.git/")
            or item.startswith("./.vscode")
        ):
            continue

        # skip if it is not the innermost directory
        if set([x[0] for x in os.walk(item) if not x[0].endswith("build")]) != {item}:
            continue

        # skip my code
        if item.startswith("./bench_utils") or "/bench_" in item:
            continue

        # remove the leading "." and add "**" to the end
        print(item[1:] + "/**")
