"""
watcher.py
----------
Watches the 'input/' folder for new PDF files and automatically runs
the full processing pipeline on each one.

Usage:
    python watcher.py          # start watching
    Ctrl-C                     # stop

Drop any PDF into:  d:/tm/ocr-machine/input/

For every PDF detected you will get:
    images/<name>/page_001.png  ...   (one PNG per page)
    output/<name>/<name>.json
    output/<name>/<name>.csv
    output/<name>/<name>.xlsx
"""

import sys
import time

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

from processor import process_pdf

INPUT_DIR = Path(__file__).parent / "input"


class _PDFHandler(FileSystemEventHandler):
    def __init__(self):
        self._seen = set()

    def on_created(self, event):
        if not isinstance(event, FileCreatedEvent):
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return
        if path in self._seen:
            return
        self._seen.add(path)

        print(f"\n[Watcher] New file detected: {path.name}")
        # Small delay so the file is fully written before we open it
        time.sleep(1)
        try:
            process_pdf(path)
        except Exception as exc:
            print(f"[ERROR] Failed to process {path.name}: {exc}")
        finally:
            self._seen.discard(path)


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    banner = (
        "\n" + "=" * 60 + "\n"
        "  OCR Machine -- Folder Watcher\n"
        + "=" * 60 + "\n"
        f"  Watching : {INPUT_DIR}\n"
        "  Drop a PDF into that folder to start processing.\n"
        "  Press Ctrl-C to stop.\n"
        + "=" * 60
    )
    print(banner)

    handler  = _PDFHandler()
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    observer.stop()
    observer.join()
    print("\n[Watcher] Stopped.")


if __name__ == "__main__":
    main()
