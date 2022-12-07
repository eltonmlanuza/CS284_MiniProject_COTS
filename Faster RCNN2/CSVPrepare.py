import os
import cv2
import subprocess
from tqdm.auto import tqdm
import pandas as pd
from IPython.display import Video, display, HTML
import warnings; warnings.simplefilter("ignore")

BASE_PATH = './tensorflow-great-barrier-reef/train_images/'
BASE_PATH_2 = './tensorflow-great-barrier-reef/'

df = pd.read_csv("./tensorflow-great-barrier-reef/train.csv")
df['annotations'] = df['annotations'].apply(eval)
df['n_annotations'] = df['annotations'].str.len()
df['has_annotations'] = df['annotations'].str.len() > 0
df['has_2_or_more_annotations'] = df['annotations'].str.len() >= 2
df['doesnt_have_annotations'] = df['annotations'].str.len() == 0
df['image_path'] = BASE_PATH + "video_" + df['video_id'].astype(str) + "/" + df['video_frame'].astype(str) + ".jpg"

df['sequence'].unique()

df['sequence'].nunique()

df.groupby("sequence")['video_id'].nunique()

# Videos 0 and 1 have 8 sequences, while video 2 has 4
df.groupby("video_id")['sequence'].nunique()

df_agg = df.groupby(["video_id", 'sequence']).agg({'sequence_frame': 'count', 'has_annotations': 'sum', 'doesnt_have_annotations': 'sum'})\
           .rename(columns={'sequence_frame': 'Total Frames', 'has_annotations': 'Frames with at least 1 object', 'doesnt_have_annotations': "Frames with no object"})

df_agg.sort_values("Total Frames")

df_agg.sort_values("Frames with at least 1 object")

# image_id is a unique identifier for a row
df['image_id'].nunique() == len(df)

df_agg.loc[[(0, 40258)]]

pd.set_option("display.max_rows", 500)
df[df['sequence'] == 40258]

df['start_cut_here'] = df['has_annotations'] & df['doesnt_have_annotations'].shift(1)  & df['doesnt_have_annotations'].shift(2)
df['end_cut_here'] = df['doesnt_have_annotations'] & df['has_annotations'].shift(1)  & df['has_annotations'].shift(2)
df['sequence_change'] = df['sequence'] != df['sequence'].shift(1)
df['last_row'] =  df.index == len(df)-1
df['cut_here'] = df['start_cut_here'] | df['end_cut_here'] | df['sequence_change'] | df['last_row']

start_idx = 0
for subsequence_id, end_idx in enumerate(df[df['cut_here']].index):
    df.loc[start_idx:end_idx, 'subsequence_id'] = subsequence_id
    start_idx = end_idx

df['subsequence_id'] = df['subsequence_id'].astype(int)

df['subsequence_id'].nunique()

drop_cols = ['start_cut_here', 'end_cut_here', 'sequence_change', 'last_row', 'cut_here', 'has_2_or_more_annotations', 'doesnt_have_annotations']
df = df.drop(drop_cols, axis=1)
df.head()

df.groupby("subsequence_id")['has_annotations'].mean().round(2).sort_values().value_counts()

df_subseq_agg = df.groupby("subsequence_id")['has_annotations'].mean()
df_subseq_agg[~df_subseq_agg.isin([0, 1])]

df[df['subsequence_id'] == 52]
df[df['subsequence_id'] == 53]
df[df['subsequence_id'] == 54]

def load_image(img_path):
    assert os.path.exists(img_path), f'{img_path} does not exist.'
    img = cv2.imread(img_path)
    return img


def load_image_with_annotations(img_path, annotations):
    img = load_image(img_path)
    if len(annotations) > 0:
        for ann in annotations:
            cv2.rectangle(img, (ann['x'], ann['y']),
                          (ann['x'] + ann['width'], ann['y'] + ann['height']),
                          (255, 255, 0), thickness=2, )
    return img


def make_video(df, part_id, is_subsequence=False):
    """
    Args:
        - part_id: either a sequence or a subsequence id
    """

    if is_subsequence:
        part_str = "subsequence_id"
    else:
        part_str = "sequence"

    print(f"Creating video for part={part_id}, is_subsequence={is_subsequence} (querying by {part_str})")
    # partly borrowed from https://github.com/RobMulla/helmet-assignment/blob/main/helmet_assignment/video.py
    fps = 15  # don't know exact value
    width = 1280
    height = 720
    save_path = BASE_PATH_2 + f'videos/video_{part_str}_{part_id}.mp4'
    tmp_path = BASE_PATH_2 + f'videos/tmp_video_{part_str}_{part_id}.mp4'

    output_video = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (width, height))

    df_part = df.query(f'{part_str} == @part_id')
    for _, row in tqdm(df_part.iterrows(), total=len(df_part)):
        img = load_image_with_annotations(row.image_path, row.annotations)
        output_video.write(img)

    output_video.release()
    # Not all browsers support the codec, we will re-load the file at tmp_output_path
    # and convert to a codec that is more broadly readable using ffmpeg
    if os.path.exists(save_path):
        os.remove(save_path)
    subprocess.run(
        ["ffmpeg", "-i", tmp_path, "-crf", "18", "-preset", "veryfast", "-vcodec", "libx264", save_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    os.remove(tmp_path)
    print(f"Finished creating video for {part_id}... saved as {save_path}")
    return save_path

video_path = make_video(df, 40258)
Video(video_path, width= 1280/2, height= 720/2)
subsequences = df.loc[df['sequence'] == 40258, 'subsequence_id'].unique()

for subsequence in subsequences:
    video_path = make_video(df, subsequence, is_subsequence=True)
    display(HTML(f"<h2>Subsequence ID: {subsequence}</h2>"))
    display(Video(video_path, width= 1280/2, height= 720/2))

from sklearn.model_selection import train_test_split, StratifiedKFold
df.head()

df_split  = df.groupby("subsequence_id").agg({'has_annotations': 'max', 'video_frame': 'count'}).astype(int).reset_index()
df_split.head()


def analyze_split(df_train, df_val, df):
    # Analize results
    print(f"   Train images                 : {len(df_train) / len(df):.3f}")
    print(f"   Val   images                 : {len(df_val) / len(df):.3f}")
    print()
    print(
        f"   Train images with annotations: {len(df_train[df_train['has_annotations']]) / len(df[df['has_annotations']]):.3f}")
    print(
        f"   Val   images with annotations: {len(df_val[df_val['has_annotations']]) / len(df[df['has_annotations']]):.3f}")
    print()
    print(
        f"   Train images w/no annotations: {len(df_train[~df_train['has_annotations']]) / len(df[~df['has_annotations']]):.3f}")
    print(
        f"   Val   images w/no annotations: {len(df_val[~df_val['has_annotations']]) / len(df[~df['has_annotations']]):.3f}")
    print()
    print(f"   Train mean annotations       : {df_train['n_annotations'].mean():.3f}")
    print(f"   Val   mean annotations       : {df_val['n_annotations'].mean():.3f}")
    print()

df_train_idx_init, df_test_idx = train_test_split(df_split['subsequence_id'], stratify=df_split["has_annotations"], test_size=0.2, random_state=42)
print("df_train_idx_init: ", df_train_idx_init.shape)
print("df_test_idx_init: ", df_test_idx.shape)
df['is_train'] = df['subsequence_id'].isin(df_train_idx_init)
df_train_init, df_test = df[df['is_train']], df[~df['is_train']]

# Print some statistics
analyze_split(df_train_init, df_test, df)

# Save to file
f_name_init = BASE_PATH_2 + f"train-validation-split/train-test.csv"
print(f"Saving file to {f_name_init}")
df.to_csv(f_name_init, index=False)
print()

for test_size in [0.01, 0.05, 0.1, 0.2]:
    print(f"Generating train-validation split with {test_size * 100}% validation")
    # df_train_idx, df_val_idx = train_test_split(df_split['subsequence_id'], stratify=df_split["has_annotations"], test_size=test_size, random_state=42)
    df_train_idx, df_val_idx = train_test_split(df_train_idx_init, test_size=test_size, random_state=42)
    print("df_train_idx_init: ", df_train_idx.shape)
    print("df_test_idx_init: ", df_val_idx.shape)
    df['is_train'] = df['subsequence_id'].isin(df_train_idx)
    df_train, df_val = df[df['is_train']], df[~df['is_train']]

    # Print some statistics
    analyze_split(df_train, df_val, df)

    # Save to file
    f_name = BASE_PATH_2 + f"train-validation-split/train-{test_size}.csv"
    print(f"Saving file to {f_name}")
    df.to_csv(f_name, index=False)
    print()

df = df.drop("is_train", axis=1)
n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
for fold_id, (_, val_idx) in enumerate(kf.split(df_split['subsequence_id'], y=df_split["has_annotations"])):
    subseq_val_idx = df_split['subsequence_id'].iloc[val_idx]
    df.loc[df['subsequence_id'].isin(subseq_val_idx), 'fold'] = fold_id

df['fold'] = df['fold'].astype(int)
df['fold'].value_counts(dropna=False)

for fold_id in df['fold'].sort_values().unique():
    print("=============================")
    print(f"Analyzing fold {fold_id}")
    df_train, df_val = df[df['fold'] != fold_id], df[df['fold'] == fold_id]
    analyze_split(df_train, df_val, df)
    print()

df.to_csv(BASE_PATH_2 + "cross-validation/train-5folds.csv", index=False)

n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2021)
for fold_id, (_, val_idx) in enumerate(kf.split(df_split['subsequence_id'], y=df_split["has_annotations"])):
    subseq_val_idx = df_split['subsequence_id'].iloc[val_idx]
    df.loc[df['subsequence_id'].isin(subseq_val_idx), 'fold'] = fold_id

df['fold'] = df['fold'].astype(int)
df['fold'].value_counts(dropna=False)

for fold_id in df['fold'].sort_values().unique():
    print("=============================")
    print(f"Analyzing fold {fold_id}")
    df_train, df_val = df[df['fold'] != fold_id], df[df['fold'] == fold_id]
    analyze_split(df_train, df_val, df)
    print()

df.to_csv(BASE_PATH_2 + "/cross-validation/train-10folds.csv", index=False)
