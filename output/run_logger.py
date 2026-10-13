"""
Automatic console + traceback logging for GeoPyFEM.

The logger mirrors stdout and stderr to both:
    1. the user's terminal, and
    2. a timestamped .log file.

The log is opened near the beginning of main.py and registered with atexit.
Therefore an unhandled Python traceback is also written to the log before the
interpreter shuts down.

Default log location
--------------------
The logger inspects the problem XML only to determine the output location.

Priority:
    <Output><Log><Directory> ...       (if supplied)
    <Output><VTK><Directory> ...       (otherwise)
    <xml-directory>/results            (fallback)

Default filename:
    <VTK BaseName>_YYYYMMDD_HHMMSS.log

If there is no VTK BaseName, the Problem name or XML filename stem is used.

Optional XML control
--------------------
Logging is ON by default.  It may optionally be customized:

<Output>
  <Log enabled="true">
    <Directory>results</Directory>
    <BaseName>my_case</BaseName>
    <Timestamp>true</Timestamp>
  </Log>
  ...
</Output>

Setting enabled="false" disables automatic logging.
"""

from __future__ import annotations

import atexit
import re
import sys
import xml.etree.ElementTree as ET

from datetime import datetime
from pathlib import Path


def _safe_name(text):
    text = str(text or "").strip()

    text = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        text,
    ).strip("._")

    return text or "geopyfem"


def _bool_text(text, default):
    if text is None:
        return bool(default)

    value = str(text).strip().lower()

    if value in (
        "true",
        "1",
        "yes",
        "on",
    ):
        return True

    if value in (
        "false",
        "0",
        "no",
        "off",
    ):
        return False

    return bool(default)


def _child_text(parent, tag):
    if parent is None:
        return None

    child = parent.find(
        tag
    )

    if (
        child is None
        or child.text is None
    ):
        return None

    text = child.text.strip()

    return text or None


def _resolve_directory(
    text,
    xml_directory,
):
    if not text:
        return (
            xml_directory
            / "results"
        )

    path = Path(
        text
    ).expanduser()

    if not path.is_absolute():
        path = (
            xml_directory
            / path
        )

    return path


def _read_log_settings(
    problem_file,
):
    """
    Read only the few XML fields required to decide the log path.

    This deliberately does NOT replace problem_reader.py.  The normal problem
    parser still performs the authoritative input validation later.
    """
    xml_path = Path(
        problem_file
    ).expanduser().resolve()

    xml_directory = (
        xml_path.parent
    )

    settings = {
        "enabled": True,
        "directory": (
            xml_directory
            / "results"
        ),
        "base_name": (
            xml_path.stem
        ),
        "timestamp": True,
    }

    try:
        root = ET.parse(
            xml_path
        ).getroot()

    except Exception:
        # Even malformed XML should still get a log.  The normal XML parser
        # will raise the real error later and that traceback will be captured.
        return settings

    problem_name = (
        root.get(
            "name"
        )
        or xml_path.stem
    )

    settings[
        "base_name"
    ] = problem_name

    output = root.find(
        "Output"
    )

    if output is None:
        return settings

    vtk = output.find(
        "VTK"
    )

    if vtk is not None:
        vtk_directory = _child_text(
            vtk,
            "Directory",
        )

        if vtk_directory:
            settings[
                "directory"
            ] = _resolve_directory(
                vtk_directory,
                xml_directory,
            )

        vtk_base_name = _child_text(
            vtk,
            "BaseName",
        )

        if vtk_base_name:
            settings[
                "base_name"
            ] = vtk_base_name

    log = output.find(
        "Log"
    )

    if log is None:
        return settings

    settings[
        "enabled"
    ] = _bool_text(
        log.get(
            "enabled"
        ),
        True,
    )

    log_directory = _child_text(
        log,
        "Directory",
    )

    if log_directory:
        settings[
            "directory"
        ] = _resolve_directory(
            log_directory,
            xml_directory,
        )

    log_base_name = _child_text(
        log,
        "BaseName",
    )

    if log_base_name:
        settings[
            "base_name"
        ] = log_base_name

    settings[
        "timestamp"
    ] = _bool_text(
        _child_text(
            log,
            "Timestamp",
        ),
        True,
    )

    return settings


def _unique_log_path(
    directory,
    base_name,
    use_timestamp,
):
    directory = Path(
        directory
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_name = _safe_name(
        base_name
    )

    if use_timestamp:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        stem = (
            f"{base_name}_{stamp}"
        )

    else:
        stem = base_name

    path = (
        directory
        / f"{stem}.log"
    )

    if not path.exists():
        return path

    # Avoid accidental overwrite when two analyses begin in the same second
    # or Timestamp=false is requested.
    index = 1

    while True:
        candidate = (
            directory
            / f"{stem}_{index:02d}.log"
        )

        if not candidate.exists():
            return candidate

        index += 1


class _TeeStream:
    """
    File-like stream that writes to both the original terminal and a log file.
    """

    def __init__(
        self,
        terminal_stream,
        log_file,
    ):
        self._terminal = (
            terminal_stream
        )

        self._log = log_file

    def write(
        self,
        text,
    ):
        if text is None:
            return 0

        self._terminal.write(
            text
        )

        self._log.write(
            text
        )

        return len(
            text
        )

    def flush(
        self,
    ):
        self._terminal.flush()
        self._log.flush()

    def isatty(
        self,
    ):
        try:
            return (
                self._terminal.isatty()
            )

        except Exception:
            return False

    def fileno(
        self,
    ):
        return (
            self._terminal.fileno()
        )

    @property
    def encoding(
        self,
    ):
        return getattr(
            self._terminal,
            "encoding",
            "utf-8",
        )


class RunLogger:
    """
    Manage one GeoPyFEM run log.

    stdout/stderr remain visible in the terminal; they are simply mirrored.
    """

    def __init__(
        self,
        path,
        problem_file,
    ):
        self.path = Path(
            path
        ).resolve()

        self.problem_file = Path(
            problem_file
        ).expanduser().resolve()

        self.started_at = datetime.now()

        self._original_stdout = (
            sys.stdout
        )

        self._original_stderr = (
            sys.stderr
        )

        self._file = open(
            self.path,
            "w",
            encoding="utf-8",
            buffering=1,
        )

        self._stdout_tee = _TeeStream(
            self._original_stdout,
            self._file,
        )

        self._stderr_tee = _TeeStream(
            self._original_stderr,
            self._file,
        )

        self._success = False
        self._closed = False

        sys.stdout = (
            self._stdout_tee
        )

        sys.stderr = (
            self._stderr_tee
        )

        self._write_header()

        atexit.register(
            self.close
        )

    def _write_header(
        self,
    ):
        print("=" * 88)
        print("GEOPYFEM RUN LOG")
        print("=" * 88)

        print(
            "Started      :",
            self.started_at.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        print(
            "Problem file :",
            self.problem_file,
        )

        print(
            "Command      :",
            " ".join(
                sys.argv
            ),
        )

        print(
            "Log file     :",
            self.path,
        )

        print("=" * 88)
        print()

    def mark_success(
        self,
    ):
        self._success = True

    def close(
        self,
    ):
        if self._closed:
            return

        self._closed = True

        finished_at = datetime.now()

        # Keep stdout/stderr mirrored while writing the footer.
        try:
            print()
            print("=" * 88)
            print("GEOPYFEM RUN LOG END")
            print("=" * 88)

            print(
                "Status       :",
                (
                    "COMPLETED"
                    if self._success
                    else "TERMINATED / ERROR"
                ),
            )

            print(
                "Finished     :",
                finished_at.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )

            print(
                "Log file     :",
                self.path,
            )

            print("=" * 88)

            self._stdout_tee.flush()
            self._stderr_tee.flush()

        finally:
            if (
                sys.stdout
                is self._stdout_tee
            ):
                sys.stdout = (
                    self._original_stdout
                )

            if (
                sys.stderr
                is self._stderr_tee
            ):
                sys.stderr = (
                    self._original_stderr
                )

            try:
                self._file.close()

            except Exception:
                pass


def start_run_logger(
    problem_file,
):
    """
    Start automatic GeoPyFEM logging.

    Returns
    -------
    RunLogger or None
        None only when logging is explicitly disabled in the XML.
    """
    settings = _read_log_settings(
        problem_file
    )

    if not settings[
        "enabled"
    ]:
        return None

    try:
        path = _unique_log_path(
            directory=settings[
                "directory"
            ],
            base_name=settings[
                "base_name"
            ],
            use_timestamp=settings[
                "timestamp"
            ],
        )

    except Exception as error:
        # A logging-path problem should not make the numerical analysis
        # impossible.  Fall back to a local "results" directory next to XML.
        xml_directory = Path(
            problem_file
        ).expanduser().resolve().parent

        fallback_directory = (
            xml_directory
            / "results"
        )

        try:
            path = _unique_log_path(
                directory=fallback_directory,
                base_name=Path(
                    problem_file
                ).stem,
                use_timestamp=True,
            )

            print(
                "Warning: requested log path could not be created; "
                f"using fallback {path}. Original error: {error}",
                file=sys.stderr,
            )

        except Exception as fallback_error:
            print(
                "Warning: automatic run logging disabled because no "
                "writable log path could be created. "
                f"Original error: {error}; fallback error: {fallback_error}",
                file=sys.stderr,
            )

            return None

    return RunLogger(
        path=path,
        problem_file=problem_file,
    )
