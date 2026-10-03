__version__ = "0.9.32"


def banner(script: str) -> None:
    import platform
    print(f"MLEM-FONF v{__version__} | {script} | py {platform.python_version()}")
