import math


def annuityfactor(rate, years):
    """Net present value factor for fixed payments per year at fixed rate"""
    if rate == 0:
        annuity = years
    else:
        annuity = (1 - 1 / ((1 + rate) ** years)) / rate
    return annuity


def round_to_base(x, base=1, method="round"):
    """Round to nearest multiple of base"""
    if method == "round":
        return int(base * round(float(x) / base))
    elif method == "floor":
        return int(base * math.floor(float(x) / base))
    elif method == "ceil":
        return int(base * math.ceil(float(x) / base))
    else:
        raise ValueError("method must be 'round', 'floor', or 'ceil'")
