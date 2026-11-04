#!/usr/bin/env python3
"""Check recorded merge-interval and retry generation against independent cases.

Use with the retained sampled coding/prose request reports. This checks actual
generated code, outside hosted CI; token equality alone is not a task-quality
check. Run only reviewed responses, on the model development machine.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import re
import sys
import types
import unittest


def source_blocks(text):
    blocks = re.findall(r"```(?:python)?[ \t]*\n(.*?)\n```", text, re.S)
    return blocks or [text]


def load_source(source):
    module = types.ModuleType("generated")
    exec(compile(source, "<generated>", "exec"), module.__dict__)
    return module


def intervals(result):
    assert result["finish"] == "stop", "interval generation is incomplete"
    module = load_source(source_blocks(result["text"])[0])
    merge = module.merge_intervals
    rng = random.Random(74831)
    for _ in range(300):
        values = [sorted((rng.randrange(-12, 13), rng.randrange(-12, 13)))
                  for _ in range(rng.randrange(12))]
        original = copy.deepcopy(values)
        actual = merge(values)
        assert values == original, "input was mutated"
        assert all(isinstance(pair, tuple) and len(pair) == 2 and pair[0] <= pair[1]
                   for pair in actual), actual
        assert all(a[1] < b[0] for a, b in zip(actual, actual[1:])), actual
        # Coverage on half-integer points is independent of a sorting/merging
        # implementation and detects gaps between adjacent integer endpoints.
        expected_points = {point for lo, hi in values for point in range(2 * lo, 2 * hi + 1)}
        actual_points = {point for lo, hi in actual for point in range(2 * lo, 2 * hi + 1)}
        assert actual_points == expected_points, (values, actual)
    for values in ([(5, 3)], [(0, 1), (5, 3)], [(-2, -3), (1, 4)]):
        try:
            merge(values)
        except ValueError:
            pass
        else:
            raise AssertionError(("invalid interval accepted", values))
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    assert suite.countTestCases() >= 6, "requested generated tests are missing"
    assert unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()


def retry(result):
    assert result["finish"] == "stop", "retry generation is incomplete"
    blocks = source_blocks(result["text"])
    source = next(block for block in blocks if re.search(r"^def retry\(", block, re.M))
    module = load_source(source)
    function = module.retry
    previous = sys.modules.get("retry_helper")
    sys.modules["retry_helper"] = module
    try:
        suite = unittest.TestSuite()
        for block in blocks:
            if "unittest.TestCase" in block:
                suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(load_source(block)))
        assert suite.countTestCases() > 0, "requested retry tests are missing"
        assert unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
    finally:
        if previous is None:
            sys.modules.pop("retry_helper", None)
        else:
            sys.modules["retry_helper"] = previous
    calls, sleeps = [], []
    errors = [OSError(str(i)) for i in range(4)]

    def fail():
        error = errors[len(calls)]
        calls.append(error)
        raise error

    try:
        function(fail, max_attempts=4, base_delay=3, max_delay=4, sleep=sleeps.append)
    except OSError as error:
        assert error is errors[-1], "last error was not re-raised"
    else:
        raise AssertionError("exhausted retries succeeded")
    assert len(calls) == 4 and sleeps == [3, 4, 4], (calls, sleeps)
    calls.clear()
    sleeps.clear()
    sentinel = object()

    def succeed():
        calls.append(None)
        if len(calls) < 3:
            raise OSError("transient")
        return sentinel

    assert function(succeed, max_attempts=3, base_delay=1, max_delay=10,
                    sleep=sleeps.append) is sentinel
    assert len(calls) == 3 and sleeps == [1, 2], (calls, sleeps)

    def wrong_error():
        calls.append(None)
        raise ValueError("not retryable")

    calls.clear()
    sleeps.clear()
    try:
        function(wrong_error, sleep=sleeps.append)
    except ValueError:
        pass
    else:
        raise AssertionError("nonretryable error was swallowed")
    assert len(calls) == 1 and sleeps == [], (calls, sleeps)
    for parameters in ({"max_attempts": 0}, {"base_delay": -1}, {"max_delay": -1}):
        try:
            function(lambda: sentinel, sleep=sleeps.append, **parameters)
        except ValueError:
            pass
        else:
            raise AssertionError(("invalid retry parameters accepted", parameters))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="recorded results.json")
    parser.add_argument("--thinking-results", type=Path,
                        help="separate code-think run with enough output budget to finish")
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    intervals(results["code"])
    thinking = json.loads(args.thinking_results.read_text()) if args.thinking_results else results
    retry(thinking["code-think"])
    print("PASS: generated interval tests, 300 independent unions and retry behavior")


if __name__ == "__main__":
    main()
