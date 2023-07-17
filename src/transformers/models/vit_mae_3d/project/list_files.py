# make a list of all the files in the directory

import os
import random

data_dir = '/media/minghui/Data/Datasets/NMSS Yuxin/agg_normalized/'

files = os.listdir(data_dir)
# filter out files that are not nifti files
files = [f for f in files if f.endswith('.nii.gz')]

# # sort the files
# files = sorted(files)

# # write file names to a text file
# with open('files.txt', 'w') as f:
#     for file in files:
#         f.write(file + '\n')

# print('Done! # of files: ', len(files))

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


# retrive the label of each subject from the file name
label_file = '/media/minghui/Data/Datasets/NMSS Yuxin/For_Cornelia_Clinical-and-Disability-Data/Progression_SPRINT_2023-01-10.csv'
# read the label file
with open(label_file, 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        splits = line.split(',')
        subject_id = splits[0]
        label = splits[5].strip()
        if subject_id in subject_files.keys():
            subject_files[subject_id]['label'] = label
    
progress = 0
not_progress = 0

# split subject_files into train and val set
# randomly shuffle the files
sids = list(subject_files.keys())
random.shuffle(sids)
train_set = sids[:int(len(sids)*0.8)]
val_set = sids[int(len(sids)*0.8):]

# write train and val set to csv files
with open('train_set.csv', 'w') as f:
    f.write('subject_id,baseline,w24,w48,label\n')
    for subject_id in train_set:
        baseline = subject_files[subject_id]['baseline'] if 'baseline' in subject_files[subject_id].keys() else ''
        w24 = subject_files[subject_id]['w24'] if 'w24' in subject_files[subject_id].keys() else ''
        w48 = subject_files[subject_id]['w48'] if 'w48' in subject_files[subject_id].keys() else ''
        label = subject_files[subject_id]['label'] if 'label' in subject_files[subject_id].keys() else ''
        if label == '1':
            progress += 1
        else:
            not_progress += 1
        f.write(subject_id + ',' + baseline + ',' + w24 + ',' + w48 + ',' + label + '\n')

with open('val_set.csv', 'w') as f:
    f.write('subject_id,baseline,w24,w48,label\n')
    for subject_id in val_set:
        baseline = subject_files[subject_id]['baseline'] if 'baseline' in subject_files[subject_id].keys() else ''
        w24 = subject_files[subject_id]['w24'] if 'w24' in subject_files[subject_id].keys() else ''
        w48 = subject_files[subject_id]['w48'] if 'w48' in subject_files[subject_id].keys() else ''
        label = subject_files[subject_id]['label'] if 'label' in subject_files[subject_id].keys() else ''
        if label == '1':
            progress += 1
        else:
            not_progress += 1
        f.write(subject_id + ',' + baseline + ',' + w24 + ',' + w48 + ',' + label + '\n')
    

# # make a csv of subject id, baseline, w24, w48
# with open('subject_list.csv', 'w') as f:
#     f.write('subject_id,baseline,w24,w48,label\n')
#     for subject_id in subject_files.keys():
#         baseline = subject_files[subject_id]['baseline'] if 'baseline' in subject_files[subject_id].keys() else ''
#         w24 = subject_files[subject_id]['w24'] if 'w24' in subject_files[subject_id].keys() else ''
#         w48 = subject_files[subject_id]['w48'] if 'w48' in subject_files[subject_id].keys() else ''
#         label = subject_files[subject_id]['label'] if 'label' in subject_files[subject_id].keys() else ''
#         if label == '1':
#             progress += 1
#         else:
#             not_progress += 1
#         f.write(subject_id + ',' + baseline + ',' + w24 + ',' + w48 + ',' + label + '\n')

print('Done! # of subjects: ', len(subject_files.keys()))
print(f'Progress: {progress}, Not progress: {not_progress}')