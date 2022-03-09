import numpy as np


def create_mask(data, condition):
    """
    :param data: np.array
    :param condition: a row of that numpy array
    :return:
    """
    # Step1: Find all occurances of x.
    dont_care_indices = [i for i, x in enumerate(condition) if str(x).lower() == "x"]

    # Step2: replace all occurances of x with 0. Here 0 is just arbitary as we don't care about these indices
    # However, it is necessary as creating mask requires only 0,1.
    updated_condition = []
    for index, c in enumerate(condition):
        if index in dont_care_indices:
            updated_condition.append(0)
        else:
            updated_condition.append(c)

    # Step3: Create the mask
    _mask = data == updated_condition

    # Step3: Iterate over the column of the mask and ignore all those columns which had x initially.
    mask = []
    for index, i in enumerate(range(data.shape[1])):
        if index not in dont_care_indices:
            mask.append(_mask[:, i])

    # Step4: if the mask is empty, it mean all columns are ignore. In that case, create a dummy mask with all True
    if not mask:
        mask = [np.full(data.shape[0], True, dtype=bool)]
    # Step4: reduce the mask.
    mask = np.logical_and.reduce(mask)
    return mask

