#!/usr/bin/env python3
"""
TDD Tests for MACA notification gating in agent.

Run with: python tests/test_agent_notifications.py
"""

import sys
import os

# Add project directory to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent import should_send_maca_summary


def test_should_send_maca_summary_true():
    print("\n[TEST] test_should_send_maca_summary_true")
    final_decision = {"proceed_to_execution": True}
    assert should_send_maca_summary(final_decision) is True
    print("  PASSED: proceed=True -> send summary")
    return True


def test_should_send_maca_summary_false():
    print("\n[TEST] test_should_send_maca_summary_false")
    final_decision = {"proceed_to_execution": False}
    assert should_send_maca_summary(final_decision) is False
    print("  PASSED: proceed=False -> skip summary")
    return True


def test_should_send_maca_summary_missing():
    print("\n[TEST] test_should_send_maca_summary_missing")
    assert should_send_maca_summary({}) is False
    assert should_send_maca_summary(None) is False
    print("  PASSED: missing decision -> skip summary")
    return True


def run_all_tests():
    print("=" * 60)
    print("AGENT NOTIFICATION TESTS - TDD PHASE")
    print("=" * 60)

    tests = [
        test_should_send_maca_summary_true,
        test_should_send_maca_summary_false,
        test_should_send_maca_summary_missing,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n[TEST] {test.__name__}")
            print(f"  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nSome tests failed - check implementation")
    else:
        print("\nAll tests passing!")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
