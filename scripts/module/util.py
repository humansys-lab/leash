import numpy as np
import pandas as pd
import pandas.api.types

from typing import Union
from sklearn.metrics import average_precision_score as APS

class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    ''' Many metrics can raise more errors than can be handled manually. This function attempts
    to identify errors that can be treated as ParticipantVisibleError without leaking any competition data.

    If the solution is purely numeric, and there are no numbers in the error message,
    then the error message is sufficiently unlikely to leak usable data and can be shown to participants.

    We expect this filter to reject many safe messages. It's intended only to reduce the number of errors we need to manage manually.
    '''
    # This check treats bools as numeric
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pandas.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pandas.api.types.is_bool_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution_is_all_numeric = pandas.api.types.is_numeric_dtype(solution)
        solution_has_bools = pandas.api.types.is_bool_dtype(solution)

    if not solution_is_all_numeric:
        return False

    for char in error_message:
        if char.isnumeric():
            return False
    if solution_has_bools:
        if 'true' in error_message.lower() or 'false' in error_message.lower():
            return False
    return True


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    '''
    Call score. If that raises an error and that already been specifically handled, just raise it.
    Otherwise make a conservative attempt to identify potential participant visible errors.
    '''
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == 'ParticipantVisibleError':
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == 'HostVisibleError':
            raise HostVisibleError(error_message)
        else:
            if treat_as_participant_error(error_message, solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err
    return score_result


def verify_valid_probabilities(df: pd.DataFrame, df_name: str):
    """ Verify that the dataframe contains valid probabilities.

    The dataframe must be limited to the target columns; do not pass in any ID columns.
    """
    if not pandas.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f'All target values in {df_name} must be numeric')

    if df.min().min() < 0:
        raise ParticipantVisibleError(f'All target values in {df_name} must be at least zero')

    if df.max().max() > 1:
        raise ParticipantVisibleError(f'All target values in {df_name} must be no greater than one')

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f'Target values in {df_name} do not add to one within all rows')
    
    



def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Calculates the average of mAP score across each protein-group combination.
    This metric only applies to Leash Bio - Predict New Medicines with BELKA
    https://www.kaggle.com/competitions/leash-BELKA
    Pseudocode:
    1. Calculate mAP at each protein-group combination
    2. Return average of Step 1
    '''

    target = 'binds'

    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All target values must be numeric')

    if submission.min().min() < 0:
        raise ParticipantVisibleError('All target values must be at least zero')

    if submission.max().max() > 1:
        raise ParticipantVisibleError('All target values must be no greater than one')

    protein_names = solution['protein_name'].unique().tolist()
    split_groups = solution['split_group'].unique().tolist()

    scores = []

    for protein_name in protein_names:
        for split_group in split_groups:
            select = (solution['protein_name'] == protein_name) & (solution['split_group'] == split_group)
            if len(solution.loc[select]) > 0: # not all combinations are present in the Public split
                score = safe_call_score(
                    sklearn.metrics.average_precision_score,
                    solution.loc[select, target].values,
                    submission.loc[select].values)
                scores.append(score)

    return np.mean(scores)


def get_score(target, preds):
    aps0 = APS(target[:,0], preds[:,0], average="micro")
    aps1 = APS(target[:,1], preds[:,1], average="micro")
    aps2 = APS(target[:,2], preds[:,2], average="micro")
    return (aps0 + aps1 + aps2) / 3