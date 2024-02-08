KEEPCHARACTERS = set([".", "_"])


def safe_str(unsafe: str) -> str:
    if not isinstance(unsafe, str):
        raise ValueError(f"{unsafe} ({type(unsafe)}) not a string")

    return "".join(c for c in unsafe if c.isalnum() or c in KEEPCHARACTERS).rstrip()
