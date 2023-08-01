import cv2
import glob
import os
import youtube_dl
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from argparse import ArgumentParser

def extract_clip_cropped_face(clip_meta_path, vid_path, save_vid_root, save_cropped_face_root, verbose=True):
    compression_level = 6
    # read the basic info
    clip_meta_file = open(clip_meta_path, 'r')
    clip_name = os.path.splitext(os.path.basename(clip_meta_path))[0]
    for line in clip_meta_file:
        if line.startswith('FPS'):
            clip_fps = float(line.strip().split(' ')[-1])
        # get the coordinates of face
        if line.startswith('CROP'):
            clip_crop_bbox = line.strip().split(' ')[-4:]
            x0 = int(clip_crop_bbox[0])
            y0 = int(clip_crop_bbox[1])
            x1 = int(clip_crop_bbox[2])
            y1 = int(clip_crop_bbox[3])

    if verbose:
        print(f'FPS: {clip_fps}')

    _, _, pid, clip_idx, frame_rlt = clip_name.split('+')
    pid = int(pid.split('P')[1])
    clip_idx = int(clip_idx.split('C')[1])
    frame_start, frame_end = frame_rlt.replace('F', '').split('-')
    # NOTE
    frame_start, frame_end = int(frame_start) + 1, int(frame_end) - 1

    start_t = round(frame_start / float(clip_fps), 5)
    end_t = round(frame_end / float(clip_fps), 5)
    duration_t = end_t - start_t

    if verbose:
        print(f'\t{frame_start} - {frame_end}: {frame_end-frame_start}. ' f'{start_t} - {end_t}: {duration_t}')

    save_clip_root = os.path.join(save_vid_root, clip_name)
    os.makedirs(save_clip_root, exist_ok=True)

    save_cropped_face_clip_root = os.path.join(save_cropped_face_root, clip_name)
    os.makedirs(save_cropped_face_clip_root, exist_ok=True)

    ffmpeg_cmd = (
        f'ffmpeg -loglevel error -i {vid_path} -an -vf "select=between(n\,{frame_start}\,{frame_end}),setpts=PTS-STARTPTS" '  # noqa E501
        f'-qscale:v 1 -qmin 1 -qmax 1 -vsync 0 '
        f'{save_clip_root}/%08d.jpg')
    if verbose:
        print(ffmpeg_cmd)
    os.system(ffmpeg_cmd)

    # crop the HQ frames
    hq_frame_list = sorted(glob.glob(os.path.join(save_clip_root, '*')))
    for frame_path in hq_frame_list:
        basename = os.path.splitext(os.path.basename(frame_path))[0]
        frame = cv2.imread(frame_path)
        cropped_face = frame[y0:y1, x0:x1]
        save_cropped_face_path = os.path.join(save_cropped_face_clip_root, f'{basename}.png')
        cv2.imwrite(save_cropped_face_path, cropped_face, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])


def download_video(videoid):
    # download youtube video
    ydl_opts = {
        'format': 'bestvideo/best',
        'outtmpl': 'datasets/train/download_youtube/%(id)s.%(ext)s',
        'ignore-errors': True
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={videoid}'])

    except Exception as error:
        print('Error: ', error)

def crop_clip_meta(clip_meta_path):
    clip_name = os.path.basename(clip_meta_path)
    _, videoid, pid, clip_idx, frame_rlt = clip_name.split('+')

    # save extracted hq frames
    save_extracted_hq_results = f'datasets/train/extracted_hq_results/{videoid}'
    # save cropped hq faces
    save_cropped_face_root = f'datasets/train/extracted_cropped_face_results/{videoid}'
    vid_path = os.path.join('datasets/train/download_youtube', f'{videoid}.mp4')
    if not os.path.exists(vid_path):
        vid_path = os.path.join('datasets/train/download_youtube', f'{videoid}.webm')
    os.makedirs(save_extracted_hq_results, exist_ok=True)
    os.makedirs(save_cropped_face_root, exist_ok=True)
    extract_clip_cropped_face(clip_meta_path, vid_path, save_extracted_hq_results, save_cropped_face_root, verbose=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--metadata", default='meta_info', help='Path to metadata')
    parser.add_argument("--workers", default=4, type=int, help='Number of workers')
    args = parser.parse_args()
    meta_info = sorted(list(Path(args.metadata).glob("*.txt")))
    meta_info = [str(file) for file in meta_info]
    video_info = set()

    for clip_meta_path in meta_info:
        clip_name = os.path.basename(clip_meta_path)
        _, videoid, pid, clip_idx, frame_rlt = clip_name.split('+')
        video_info.add(videoid)
    
    print("Number of videos :", len(video_info))

    pool = Pool(processes=args.workers)
    for chunks_data in tqdm(pool.imap(download_video, video_info)):
        pass

    for chunks_data in tqdm(pool.imap(crop_clip_meta, meta_info)):
        pass
