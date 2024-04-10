"""
This code reads BCI2000 EDF files, applies ICA and down-sampling (160Hz -> 10Hz), concatenates three sessions of
performing Task 2 (i.e. 4, 8, and 12), and finally stores the results to a CSV file for each participant. The original
data and the paper for the BCI2000 dataset can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/ and
https://pubmed.ncbi.nlm.nih.gov/15188875/, respectively.
"""

import mne
import os
import pandas as pd


def main():
    data_path = 'raw'
    down_sample = True
    new_frequency = 10
    do_ica = True
    no_components = 0.99

    for root, directories, files in os.walk(data_path, topdown=False):
        final_data = pd.DataFrame()
        for file in files:
            if file.endswith('04.edf') | file.endswith('08.edf') | file.endswith('12.edf'):

                # Reading data
                input_file = os.path.join(root, file)
                data = mne.io.read_raw_edf(input_file)
                data.pick_types(eeg=True).load_data()

                # ICA
                if do_ica:
                    ica = mne.preprocessing.ICA(n_components=no_components, method='fastica').fit(data)
                    ica.apply(data)

                # Down-sampling
                if down_sample:
                    data = data.resample(sfreq=new_frequency)

                # Extracting events
                events = mne.events_from_annotations(data)

                # Epoching data
                epochs = mne.Epochs(data, events[0])
                data = epochs.to_data_frame()

                data = data.reset_index(drop=True)
                data.insert(data.shape[1], 'label', data['condition'])
                data = data.drop(['time'], axis=1)
                data = data.drop(['epoch'], axis=1)
                data = data.drop(['condition'], axis=1)
                final_data = pd.concat([final_data, data], axis=1)

        # Storing data
        if not os.path.exists('eeg'):
            os.mkdir('eeg')

        final_data.to_csv(os.path.join('eeg', file[:4] + '.csv'), index=False, header=False)


if __name__ == "__main__":
    main()
