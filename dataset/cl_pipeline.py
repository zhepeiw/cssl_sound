import pandas as pd
import numpy as np
import pdb


def prepare_task_csv_from_replay(
    input_csv,
    buffer,
    num_keep
):
    '''
        prepare csv for each task with a rehearsal buffer

        args:
            input_csv: str, path to the input csv
            output_csv: str, path to the output csv
            buffer: list of dicts
    '''
    rng = np.random.RandomState(1234)
    df = pd.read_csv(input_csv, index_col=None)
    curr_data = df.to_dict('records')
    curr_buffer = rng.choice(
        curr_data,
        min(num_keep, len(curr_data)),
        replace=False
    ).tolist()
    df_agg = pd.DataFrame(curr_data + buffer)
    df_agg['ID'] = np.arange(len(df_agg))
    df_agg.to_csv(input_csv.replace('raw', 'replay'), index=False)
    return curr_buffer
