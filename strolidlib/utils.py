"""General-purpose utilities shared across Conserver components."""


def seconds_to_ydhms(seconds: float) -> str:
    """Convert seconds into a human-friendly combination of years, days, hours, minutes, seconds."""
    seconds = int(seconds)
    years, rem = divmod(seconds, 31_536_000)
    days, rem = divmod(rem, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)

    parts = []
    if years:
        parts.append(f"{years}y")
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def are_we_parallel(opts):
    return opts.get("parallel", False)


def opts_have_changed(cur_opts, prev_opts) -> bool:
    return cur_opts != prev_opts


def include_default_opts(opts, default_options):
    for key, value in default_options.items():
        if key not in opts:
            opts[key] = value
    return opts