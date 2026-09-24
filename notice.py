# EDUCATIONAL USE ONLY. See NOTICE.md.

NOTICE = """\
Educational use only. This runs Qwen3, an open-weight model from Alibaba Cloud. Please don't
use this model or this code for any company work, or with any company code or data."""


def print_notice(color=True):
    start, end = ("\033[1;33m", "\033[0m") if color else ("", "")
    print(f"{start}⚠  {NOTICE}{end}\n", flush=True)
