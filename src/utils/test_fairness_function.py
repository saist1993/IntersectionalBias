import collections
import numpy as np
from .fairness_function import create_mask, \
    create_all_possible_groups, get_gerrymandering_groups, get_independent_groups, get_intersectional_groups

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


def test_all_create_all_possible_groups_simple():
    number_of_attributes = 2
    expected_results = [(1,1), (1,0), (0,1), (0,0), ('x',1), (1,'x'), ('x','x'), (0,'x'), ('x',0)]
    received_all_possible_groups = create_all_possible_groups(number_of_attributes=number_of_attributes)
    assert collections.Counter(expected_results) == collections.Counter(received_all_possible_groups)

def test_get_gerrymandering_groups():
    groups = [(1,1,1), (2,2,2), ('x','x','x')]
    assert ('x','x','x') not in get_gerrymandering_groups(groups)


def test_get_independent_groups():
    groups = [(1,1,1), (2,2,2), ('x','x','x'), ('x','x',1), ('x',1,'x')]
    expected_results = [('x','x',1), ('x',1,'x')]
    received_groups = get_independent_groups(groups)
    assert collections.Counter(expected_results) == collections.Counter(received_groups)


def test_get_intersectional_groups():
    groups = [(1,1,1), (2,2,2), ('x','x','x'), ('x','x',1), ('x',1,'x')]
    expected_results = [(1,1,1), (2,2,2)]
    received_groups = get_intersectional_groups(groups)
    assert collections.Counter(expected_results) == collections.Counter(received_groups)
