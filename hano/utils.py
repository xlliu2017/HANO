import pickle


class Colors:
    """Defining color codes for terminal text."""

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


def save_pickle(var, save_path):
    with open(save_path, "wb") as file_obj:
        pickle.dump(var, file_obj)
