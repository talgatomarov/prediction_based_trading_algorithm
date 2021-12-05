def percentile_rank(arr, x):
    assert len(arr) > 0

    result = (arr <= x).sum() / len(arr)
    return result