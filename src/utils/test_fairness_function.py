from .fairness_function import create_mask
import numpy as np

def test_mask_simple_condition_example_1():
    a = np.asarray([[1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    condition = [1,1]
    expected_mask = [True, True, False, False, True, False, False, False, False, False]
    received_mask = create_mask(a, condition)
    assert np.array_equal(expected_mask, received_mask)


def test_mask_simple_condition_example_2():
    a = np.asarray([[1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    condition = [1,0]
    expected_mask = [False, False, True, True, False, True, False, False, False, False]
    received_mask = create_mask(a, condition)
    assert np.array_equal(expected_mask, received_mask)


def test_mask_single_condtion():
    a = np.asarray([[1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    condition = [1,'X']
    expected_mask = [True, True, True, True, True, True, False, False, False, False]
    received_mask = create_mask(a, condition)
    assert np.array_equal(expected_mask, received_mask)


def test_mask_no_condtion():
    a = np.asarray([[1, 1], [1, 1], [1, 0], [1, 0], [1, 1], [1, 0], [0, 1], [0, 1], [0, 0], [0, 0]])
    condition = ['X','X']
    expected_mask = [True, True, True, True, True, True, True, True, True, True]
    received_mask = create_mask(a, condition)
    assert np.array_equal(expected_mask, received_mask)