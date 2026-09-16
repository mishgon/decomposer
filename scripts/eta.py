#!/usr/bin/env python3
"""Print the local finish time: python3 scripts/eta.py '15h 34m'."""

import argparse
from datetime import datetime
import re


def finish_time(duration, now=None):
    match = re.fullmatch(r"\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*", duration, re.IGNORECASE)
    if not match or all(value is None for value in match.groups()):
        raise ValueError("Use a duration like '15h 34m', '2h', or '45m'.")
    hours, minutes = (int(value or 0) for value in match.groups())
    now = datetime.now().astimezone() if now is None else now
    # Add elapsed seconds, including when a local daylight-saving transition occurs.
    end = datetime.fromtimestamp(now.timestamp() + hours * 3600 + minutes * 60).astimezone()
    days = (end.date() - now.date()).days
    day = {0: "today", 1: "tomorrow"}.get(days, end.strftime("%Y-%m-%d"))
    return f"{day} {end.hour}:{end.minute:02d}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("duration", nargs="+", help="For example: 15h 34m")
    args = parser.parse_args()
    try:
        print(finish_time(" ".join(args.duration)))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
