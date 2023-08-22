# please run this script in the root directory of the project
import os

if __name__ == "__main__":
    for item in set([x[0] for x in os.walk(".")]):
        # skip 3rdparty, .git, and .vscode
        if (
            item.startswith("./3rdparty")
            or item.startswith("./.git")
            or item.startswith("./.vscode")
        ):
            continue

        # skip /build/
        if "/build/" in item:
            continue

        # skip my code
        if (
            item.startswith("./bench_utils")
            or "/bench_" in item
            or "cuSPARSELt/matmul_bench" in item
            or "cuSPARSE/spgemm_reuse" in item
        ):
            continue

        # no adding this directory if it is not the innermost directory
        if (
            sum(
                set(
                    map(
                        lambda x: not os.path.isfile(os.path.join(item, x)),
                        set(os.listdir(item)),
                    )
                ).difference("build")
            )
            > 0
        ):
            # add the source files at this level
            for file_or_dir in os.listdir(item):
                if os.path.isfile(os.path.join(item, file_or_dir)):
                    # remove the leading "."
                    print(os.path.join(item, file_or_dir)[1:])
            continue

        # remove the leading "." and add "**" to the end
        print(item[1:] + "/**")
