
import sys
import os
import ast
import inspect

def called_from_main_function() -> bool:
    """This function checks if the main simulation file is ran from if __name__ == '__main__':

    Returns:
        bool: If the script is run as main function
    """
    # 1) Locate the __main__ module’s file
    main_mod = sys.modules.get("__main__")
    main_file = getattr(main_mod, "__file__", None)
    if not main_file:
        return False
    main_file = os.path.realpath(main_file)

    # 2) Parse its AST to find the `main` function and its line range
    try:
        with open(main_file, "r") as f:
            src = f.read()
    except OSError:
        return False

    try:
        tree = ast.parse(src, filename=main_file)
    except SyntaxError:
        return False

    main_ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            start = node.lineno
            # end_lineno is available in Python 3.8+; otherwise compute it
            end = getattr(node, "end_lineno",
                          max(getattr(n, "lineno", start) for n in ast.walk(node)))
            main_ranges.append((start, end))

    if not main_ranges:
        return False

    # 3) Inspect the call stack for any frame inside main_file within those line ranges
    for frame_info in inspect.stack():
        fn = os.path.realpath(frame_info.filename)
        if fn != main_file:
            continue
        lineno = frame_info.lineno
        for start, end in main_ranges:
            if start <= lineno <= end:
                return True

    return False
