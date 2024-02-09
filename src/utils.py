KEEP_CHARACTERS = set([".", "_", " "])
REPLACE_CHARACTERS = {" ": "_"}


def safe_str(unsafe: str) -> str:
    if not isinstance(unsafe, str):
        raise ValueError(f"{unsafe} ({type(unsafe)}) not a string")
    return "".join(
        REPLACE_CHARACTERS.get(c, c)
        for c in unsafe
        if c.isalnum() or c in KEEP_CHARACTERS
    ).rstrip()
