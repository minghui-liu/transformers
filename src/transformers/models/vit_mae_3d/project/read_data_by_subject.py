import os

def read_data_by_subject(data_dir='/media/minghui/Data/Datasets/NMSS Yuxin/agg_normalized/'):
    files = os.listdir(data_dir)
    # filter out files that are not nifti files
    files = [f for f in files if f.endswith('.nii.gz')]

    # group files by subject id
    subject_files = {}
    for f in files:
        splits = f.split('_')
        subject_long_id = '_'.join(splits[:3])
        subject_id = splits[2].split('-')[1]
        scan_time = splits[3]

        if subject_id not in subject_files.keys():
            subject_files[subject_id] = {}
        subject_files[subject_id][scan_time] = f

    # sort by subject id
    subject_files = dict(sorted(subject_files.items()))

