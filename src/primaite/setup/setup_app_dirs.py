# Crown Owned Copyright (C) Dstl 2023. DEFCON 703. Shared in confidence.
from logging import Logger

from primaite import _USER_DIRS, getLogger, LOG_DIR, NOTEBOOKS_DIR

_LOGGER: Logger = getLogger(__name__)


def run() -> None:
    """
    Handles creation of application directories and user directories.

    Uses `platformdirs.PlatformDirs` and `pathlib.Path` to create the required
    app directories in the correct locations based on the users OS.
    """
    app_dirs = [
        _USER_DIRS,
        NOTEBOOKS_DIR,
        LOG_DIR,
    ]

    for app_dir in app_dirs:
        if not app_dir.is_dir():
            app_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(f"Created directory: {app_dir}")


if __name__ == "__main__":
    run()
