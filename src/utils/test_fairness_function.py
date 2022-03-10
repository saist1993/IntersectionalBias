import collections
import numpy as np
from .fairness_function import create_mask, \
    create_all_possible_groups, get_gerrymandering_groups, get_independent_groups, get_intersectional_groups,\
    calculate_accuracy_classification, accuracy_parity

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
    output, index = get_gerrymandering_groups(groups=groups)
    print(index, groups)
    assert ('x','x','x') not in output
    assert len(groups) == len(index) + 1


def test_get_independent_groups():
    groups = [(1,1,1), (2,2,2), ('x','x','x'), ('x','x',1), ('x',1,'x')]
    expected_results = [('x','x',1), ('x',1,'x')]
    received_groups, indexes = get_independent_groups(groups)
    assert collections.Counter(expected_results) == collections.Counter(received_groups)
    assert [3,4] == indexes

def test_get_intersectional_groups():
    groups = [(1,1,1), (2,2,2), ('x','x','x'), ('x','x',1), ('x',1,'x')]
    expected_results = [(1,1,1), (2,2,2)]
    received_groups, indexes = get_intersectional_groups(groups)
    assert collections.Counter(expected_results) == collections.Counter(received_groups)
    assert [0,1] == indexes


def test_calculate_accuracy_classification():
    predictions = np.asarray([1,1,0,0])
    labels = np.asarray([1,1,1,0])
    expected_accuracy = 3.0/4.0
    received_accuracy = calculate_accuracy_classification(predictions, labels)
    assert expected_accuracy == received_accuracy


def test_accuracy_parity():
    predictions = np.asarray([1,1,0,0])
    labels = np.asarray([1,1,1,0])
    mask = [True, True, False, True]
    expected_accuracy = 1.0
    received_accuracy = accuracy_parity(predictions, labels, mask)
    assert expected_accuracy == received_accuracy


lista = [10, 20, 30]
def marks(lista):
    lista.append([11, 12, 13, 14, 15])
    print("Value inside the function: ", lista)
    return lista


marks(lista)
print("Value outside the function: ", lista)