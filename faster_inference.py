from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

def ensure_dir(directory):
    """ディレクトリが存在することを確認し、存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description='Inference code to lip-sync images using Wav2Lip models with BF16 optimization')

parser.add_argument('--checkpoint_path', type=str, 
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
                    help='Filepath of image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, 
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
                                default='results/result_voice.mp4')

parser.add_argument('--fps', type=float, help='FPS for output video (default: 25)', 
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int, 
                    help='Batch size for face detection', default=32)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=256)

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

# 引数を解析
args = parser.parse_args()
args.img_size = 96  # モデルの入力画像サイズを96x96に固定

# 一時ディレクトリの作成
if not os.path.exists('temp'):
    os.makedirs('temp')

# 入力顔画像が適切な形式かチェック
if not args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
    raise ValueError('--face argument must be an image file (jpg, png, or jpeg)')

# 出力ディレクトリの作成
ensure_dir(os.path.dirname(args.outfile))

def get_smoothened_boxes(boxes, T):
    """
    顔検出のバウンディングボックスを時間的に平滑化する関数
    
    引数:
        boxes: 顔検出のバウンディングボックス配列
        T: 平滑化するウィンドウサイズ
        
    戻り値:
        平滑化されたバウンディングボックス配列
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    """
    画像から顔を検出する関数
    
    引数:
        images: 顔検出を行う画像のリスト
        
    戻り値:
        検出された顔の切り抜き画像とその座標のリスト
    """
    # 顔検出器の初期化
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    
    # バッチサイズを調整しながら顔検出を実行
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            # GPUメモリ不足の場合、バッチサイズを半分にして再試行
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            # 顔が検出されなかった場合のエラー
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the image contains a face.')

        # 検出された顔の周囲にパディングを追加
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    # 時間的な平滑化（--nosmooth フラグが設定されていない場合）
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    # 検出器のメモリを解放
    del detector
    return results 

def datagen(frame, mels):
    """
    Wav2Lipモデルへの入力データを生成するジェネレータ関数
    
    引数:
        frame: 入力顔画像
        mels: メルスペクトログラムのチャンク
        
    戻り値:
        バッチ処理用のデータジェネレータ
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # 顔の検出または指定されたバウンディングボックスを使用
    if args.box[0] == -1:
        face_det_results = face_detect([frame])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[frame[y1: y2, x1:x2], (y1, y2, x1, x2)]]

    # 各メルスペクトログラムチャンクに対して処理
    for m in mels:
        frame_to_save = frame.copy()
        face, coords = face_det_results[0].copy()

        # 顔画像をモデル入力サイズにリサイズ
        face = cv2.resize(face, (args.img_size, args.img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        # バッチサイズに達したらデータを変換してイールド
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            # 画像の右半分をマスク（モデルの入力要件）
            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            # データの形状変換とスケーリング
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    # 残りのデータがあればそれも処理
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

# メルスペクトログラムのステップサイズ
mel_step_size = 16
# 使用するデバイス（CUDAが利用可能ならGPU、そうでなければCPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    """
    モデルのチェックポイントを読み込む関数
    
    引数:
        checkpoint_path: チェックポイントファイルのパス
        
    戻り値:
        ロードされたチェックポイント
    """
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    """
    Wav2Lipモデルを初期化して重みを読み込む関数
    
    引数:
        path: モデルのチェックポイントファイルのパス
        
    戻り値:
        評価モードに設定されたモデル
    """
    model = Wav2Lip()
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    # DataParallelラッパーを削除（'module.'プレフィックスを削除）
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    # モデルをデバイスに移動
    model = model.to(device)
    # GPUの場合はBFloat16形式に変換して効率化
    if device == 'cuda':
        model = model.bfloat16()
    return model.eval()

def main():
    """
    メイン処理関数
    Wav2Lipを使用して顔画像と音声から口の動きを同期させた動画を生成
    """
    # 入力顔画像の存在確認
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to an image file')

    # 顔画像の読み込み
    full_frame = cv2.imread(args.face)
    if full_frame is None:
        raise ValueError(f'Failed to load image: {args.face}')
        
    fps = args.fps

    # 入力オーディオがWAV形式でない場合、FFMPEGを使用して変換
    if not args.audio.endswith('.wav'):
        subprocess.call('ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav'), shell=True)
        args.audio = 'temp/temp.wav'

    # オーディオの読み込みとメルスペクトログラムの計算
    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    # メルスペクトログラムのNaN値チェック
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    # メルスペクトログラムをフレームレートに合わせてチャンク分割
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    # モデルのロードとデータジェネレータの初期化
    model = load_model(args.checkpoint_path)
    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frame, mel_chunks)

    # 出力ビデオの設定
    frame_h, frame_w = full_frame.shape[:-1]
    temp_video_path = 'temp/result.avi'
    ensure_dir(os.path.dirname(temp_video_path))
    
    try:
        # 出力ビデオライターの作成
        out = cv2.VideoWriter(temp_video_path, 
                            cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
        if not out.isOpened():
            raise ValueError(f"Failed to create VideoWriter at {temp_video_path}")

        # ジェネレータからデータを取得し、モデル推論を実行
        for img_batch, mel_batch, frames, coords in tqdm(gen, 
                                                total=int(np.ceil(float(len(mel_chunks))/batch_size))):
            # テンソル変換とデバイス移動
            img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device).bfloat16()
            mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device).bfloat16()

            # 勾配計算なしでモデル推論
            with torch.no_grad():
                pred = model(mel_batch, img_batch)

            # 予測結果の後処理
            pred = pred.cpu().float().numpy().transpose(0, 2, 3, 1) * 255.
            
            # 各フレームに予測された口の動きを合成
            for p, f, c in zip(pred, frames, coords):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
        
        # ビデオライターを解放
        out.release()

    finally:
        # 例外発生時もビデオライターを確実に解放
        if 'out' in locals() and out is not None:
            out.release()

    # 出力ディレクトリの確保
    ensure_dir(os.path.dirname(args.outfile))
    # 最終的な出力ビデオを生成（NVIDIAエンコーダを使用）
    output_command = [
        'ffmpeg', '-y',
        '-i', args.audio,
        '-i', temp_video_path,
        '-c:v', 'h264_nvenc',  # NVIDIAのハードウェアエンコーダを使用
        '-preset', 'p4',
        '-b:v', '2M',
        '-maxrate', '2.5M',
        '-bufsize', '2M',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-strict', '-2',
        args.outfile
    ]

    # FFMPEGコマンドを実行して音声と映像を結合
    subprocess.call(output_command)

if __name__ == '__main__':
    main()