import asyncio
import websockets
import pyaudio
import base64
import json
import os
import time
import threading
import tkinter as tk
from datetime import datetime
import re
from openai import OpenAI
from gui_player import GUIVideoPlayer

class ConversationBuffer:
    def __init__(self):
        self.current_buffer = []
        self.current_role = None
        self.current_timestamp = None

    def append(self, text, role, timestamp):
        """バッファにテキストを追加"""
        if self.current_role and role != self.current_role:
            return self.flush()
        
        self.current_role = role
        self.current_timestamp = timestamp
        self.current_buffer.append(text)
        return None

    def flush(self):
        """現在のバッファを文章として結合""",
        if not self.current_buffer:
            return None

        complete_text = ''.join(self.current_buffer)
        result = {
            'timestamp': self.current_timestamp,
            'role': self.current_role,
            'content': complete_text.strip()
        }
        
        self.current_buffer = []
        self.current_role = None
        self.current_timestamp = None
        
        return result

class EnhancedChat:
    def __init__(self, face_image_path="face.jpg"):
        # 基本設定
        self.api_key = os.environ.get('OPENAI_API_KEY')
        self.ws_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        
        # OpenAI clientの初期化
        self.openai_client = OpenAI()
        
        # 音声設定
        self.chunk = 2400
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 24000
        
        # PyAudioの初期化
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        
        # 会話管理
        self.max_history_length = 10
        self.last_voice_timestamp = time.time()
        self.last_audio_output_timestamp = time.time()
        self.silence_threshold = 30
        self.last_prompt_time = 0
        
        # ログ関連の設定
        self.log_dir = "conversation_logs"
        self.ensure_log_directory()
        self.conversation_buffer = ConversationBuffer()
        self.current_log_file = self.initialize_log_file()
        
        # 音声保存用のディレクトリ設定
        self.audio_dir = "saved_audios"
        self.ensure_audio_directory()
        
        # リップシンク関連の設定
        self.temp_dir = "temp"
        self.ensure_temp_dir()
        self.face_image_path = face_image_path
        self.current_video_path = None
        
        # 簡素化したバッファ
        self.response_text = ""
        self.audio_buffer = bytearray()
        
        # 応答完了フラグ
        self.response_completed = False
        
        # 顔画像の存在確認
        if not os.path.exists(self.face_image_path):
            raise ValueError(f"顔画像が見つかりません: {self.face_image_path}")
        
        # アバター応答を蓄積する配列
        self.user_transcripts = []

    def ensure_log_directory(self):
        """ログディレクトリの作成"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
    def ensure_audio_directory(self):
        """音声保存ディレクトリの作成"""
        if not os.path.exists(self.audio_dir):
            os.makedirs(self.audio_dir)

    def initialize_log_file(self):
        """新しいログファイルの初期化"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"conversation_{timestamp}.txt")
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"会話ログ - 開始時刻: {timestamp}\n")
            f.write("-" * 50 + "\n\n")
        
        return log_file

    def update_conversation_history(self, role, content):
        """会話履歴をファイルに保存"""
        if content.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            complete_message = self.conversation_buffer.append(content, role, timestamp)
            
            if complete_message:
                try:
                    with open(self.current_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{complete_message['timestamp']}] {complete_message['role']}: {complete_message['content']}\n\n")
                except Exception as e:
                    print(f"メッセージ書き込みエラー: {e}")

    def get_recent_messages(self):
        """最近のメッセージを取得"""
        messages = []
        try:
            with open(self.current_log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            content_lines = [l for l in lines[3:] if l.strip() and l.startswith("[")]
            recent_lines = content_lines[-self.max_history_length:]
            
            for line in recent_lines:
                try:
                    match = re.match(r'\[(.*?)\] (.*?): (.*)', line)
                    if match:
                        timestamp, role, content = match.groups()
                        messages.append({
                            "role": role,
                            "content": content.strip()
                        })
                except Exception as e:
                    print(f"行解析エラー: {e}")
                    
        except Exception as e:
            print(f"メッセージ読み取りエラー: {e}")
            
        return messages

    async def generate_next_prompt(self):
        """GPT-4o-miniを使用して次のプロンプトを生成"""
        try:
            messages = [
                {"role": "developer", "content": """
                あなたは会話を自然に展開するアシスタントです。
                会話が途切れたため、これまでの会話履歴を分析して次の自然な話題を提案してください。
                
                以下の指示に従い、必ず指定された形式で出力してください：

                出力形式：
                [CONTEXT]
                - 現在の話題と状況を簡潔に述べよ
                
                [NEXT_TOPIC]
                - 文脈から導かれる次の自然な話題を30文字以内で述べよ

                ---出力例---
                 
                [CONTEXT]
                ラーメンの具材について話していた
                [NEXT TOPIC]
                どの具材が好きかに関する話題
                ------------
                
                """
                }
            ]
            
            recent_messages = self.get_recent_messages()
            messages.extend(recent_messages)
            
            completion = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"プロンプト生成エラー: {e}")
            return "他に気になることはありますか？"

    def setup_audio(self):
        """音声ストリームのセットアップ"""
        self.input_stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

    def ensure_temp_dir(self):
        """一時ディレクトリの確保"""
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def clean_temp_files(self):
        """一時ファイルの削除"""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path) and file_path != self.current_video_path:
                    try:
                        os.unlink(file_path)
                    except Exception as e:
                        print(f"ファイル削除エラー {file_path}: {e}")
        except Exception as e:
            print(f"一時ファイルクリーンアップエラー: {e}")

    async def generate_lipsync(self, audio_path):
        """Wav2Lipを使用してリップシンク動画を生成"""
        output_path = os.path.join(self.temp_dir, f"result_voice_{int(time.time())}.mp4")
        
        command = [
            "python", "faster_inference.py",
            "--checkpoint_path", "checkpoints/wav2lip.pth",
            "--face", self.face_image_path,
            "--audio", audio_path,
            "--outfile", output_path
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                self.current_video_path = output_path
                return output_path
            else:
                print(f"リップシンク生成エラー: {stderr.decode()}")
                return None
        except Exception as e:
            print(f"リップシンク実行エラー: {e}")
            return None

    def save_audio_buffer(self, audio_buffer, filename):
        """音声バッファをWAVファイルとして保存"""
        import wave
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.p.get_sample_size(self.format))
            wav_file.setframerate(self.rate)
            wav_file.writeframes(audio_buffer)

    async def process_received_audio(self, audio_data, subtitle_text):
        """受信した音声データを処理してリップシンク動画を生成"""
        temp_audio = None
        try:
            # 一時ファイルのパス
            temp_audio = os.path.join(self.temp_dir, "temp_response.wav")
            
            # 永続保存用の音声ファイルのパス
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            permanent_audio = os.path.join(self.audio_dir, f"response_{timestamp}.wav")
            
            # 音声データの保存
            self.save_audio_buffer(audio_data, temp_audio)
            self.save_audio_buffer(audio_data, permanent_audio)
            print(f"音声を保存しました: {permanent_audio}")
            
            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                print("エラー: 音声ファイルが保存できないか空です")
                return
            
            # リップシンク動画の生成
            video_path = await self.generate_lipsync(temp_audio)
            if video_path and os.path.exists(video_path):
                try:
                    print(f"動画再生開始: {video_path}")
                    print(f"字幕: {subtitle_text}")
                    
                    # 別プロセスとしてgui_player.pyを実行
                    import subprocess
                    cmd = [
                        "python", 
                        "gui_player.py", 
                        video_path,
                        "--subtitle", 
                        subtitle_text
                    ]
                    
                    # 非同期でプロセス起動し、終了を待機
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE
                    )
                    
                    # プロセスの終了を待機
                    await process.wait()
                    
                    # 現在の動画パスを保存
                    self.current_video_path = video_path
                except Exception as e:
                    print(f"動画再生エラー: {e}")
            else:
                print("エラー: リップシンク動画の生成に失敗しました")
        except Exception as e:
            print(f"process_received_audioでのエラー: {str(e)}")
        finally:
            if temp_audio and os.path.exists(temp_audio):
                try:
                    os.unlink(temp_audio)
                except Exception as e:
                    print(f"一時音声ファイル削除エラー: {e}")

    async def send_text_message(self, websocket, text):
        """テキストメッセージの送信"""
        try:
            await websocket.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": text
                    }]
                }
            }))
            
            await websocket.send(json.dumps({
                "type": "response.create"
            }))
            
            self.last_prompt_time = time.time()
            print(f"テキストプロンプト送信:\n{text}\n")
            
        except Exception as e:
            print(f"テキストメッセージ送信エラー: {e}")

    async def check_silence(self, websocket):
        """無音期間の監視とプロンプト送信"""
        prompt_cooldown = 60
        
        while True:
            current_time = time.time()
            last_activity = max(self.last_voice_timestamp, self.last_audio_output_timestamp)
            silence_duration = current_time - last_activity
            time_since_last_prompt = current_time - self.last_prompt_time

            if silence_duration > self.silence_threshold and time_since_last_prompt > prompt_cooldown:
                next_prompt = await self.generate_next_prompt()
                await self.send_text_message(websocket, next_prompt)

            await asyncio.sleep(5)

    async def handle_messages(self, websocket):
        """WebSocketメッセージの処理"""
        accumulated_audio = bytearray()
        current_transcript = []  # 音声認識の途中経過を保持
        
        # 変数を初期化
        self.response_text = ""
        self.audio_buffer = bytearray()
        self.response_completed = False
        self.user_transcripts = []
        
        print("\n=== 字幕デバッグモード有効（アバター応答の表示、スペース区切り）===")
        
        async for message in websocket:
            response = json.loads(message)
            msg_type = response.get("type", "")
            
            # 新しい応答の開始を検知
            if msg_type == "response.text.start":
                print("\n[デバッグ] 新しい応答開始")
                self.response_text = ""
            
            elif msg_type == "input_audio_buffer.processed":
                self.last_voice_timestamp = time.time()
            
            elif msg_type == "response.audio_transcript.delta":
                # 音声認識結果の処理
                transcript = response.get("delta", "")
                print(transcript, end="", flush=True)
                if transcript.strip():
                    current_transcript.append(transcript)
                    if any(transcript.endswith(end) for end in ["。", "！", "？", "\n"]):
                        full_transcript = "".join(current_transcript)
                        full_transcript = full_transcript.replace("\n", " ").strip()
                        self.update_conversation_history("user", full_transcript)
                        
                        # 音声認識結果を蓄積
                        self.user_transcripts.append(full_transcript)
                        print(f"\n[デバッグ] 音声認識結果: {full_transcript}")
                        
                        current_transcript = []
            
            elif msg_type == "response.text.delta":
                # テキスト応答の処理
                content = response.get("delta", "")
                if content.strip():
                    self.update_conversation_history("assistant", content)
                    self.response_text += content
            
            elif msg_type == "response.audio.delta":
                # 音声データの蓄積
                if response.get("delta"):
                    audio_data = base64.b64decode(response["delta"])
                    accumulated_audio.extend(audio_data)
            
            elif msg_type == "response.done":
                print("\n[デバッグ] 応答完了")
                
                # 未保存の認識結果があれば保存
                if current_transcript:
                    full_transcript = "".join(current_transcript)
                    full_transcript = full_transcript.replace("\n", " ").strip()
                    self.update_conversation_history("user", full_transcript)
                    current_transcript = []
                
                # バッファに残っているメッセージを確実に保存
                complete_message = self.conversation_buffer.flush()
                if complete_message:
                    with open(self.current_log_file, "a", encoding="utf-8") as f:
                        f.write(f"[{complete_message['timestamp']}] {complete_message['role']}: {complete_message['content']}\n\n")
                
                # 応答完了時刻を記録
                self.last_audio_output_timestamp = time.time()
                self.response_completed = True
                
                # 音声データが蓄積された場合のみ処理
                if accumulated_audio:
                    # アバター応答をスペース区切りで結合して字幕として使用
                    user_display_text = " ".join(self.user_transcripts)
                    
                    # 音声データを処理
                    self.audio_buffer = accumulated_audio.copy()
                    
                    try:
                        # リップシンク動画生成と再生
                        await self.process_received_audio(self.audio_buffer, user_display_text)
                        
                        # 動画再生完了後、アバター応答配列をリセット
                        self.user_transcripts = []
                        
                    except Exception as e:
                        print(f"process_received_audioでエラー発生: {e}")
                    
                    # 各バッファをリセット
                    accumulated_audio = bytearray()
                    self.response_text = ""
                    self.audio_buffer = bytearray()
                    self.response_completed = False


    async def send_audio(self, websocket):
        """音声データの送信"""
        while True:
            try:
                audio_data = self.input_stream.read(self.chunk, exception_on_overflow=False)
                base64_audio = base64.b64encode(audio_data).decode("utf-8")
                
                await websocket.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64_audio
                }))
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"音声送信エラー: {e}")
                await asyncio.sleep(1)

    async def start(self):
        """メイン実行関数"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "OpenAI-Beta": "realtime=v1"
        }
        
        try:
            self.setup_audio()
            
            async with websockets.connect(self.ws_url, additional_headers=headers) as websocket:
                print("WebSocketに接続しました")
                
                # セッション設定
                await websocket.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "modalities": ["audio", "text"],
                        "instructions": "日本語で短く簡潔に回答してください。",
                        "voice": "alloy",
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5
                        }
                    }
                }))
                
                # 初期メッセージ送信
                await self.send_text_message(websocket, "やあ！会話を始めましょう！")
                
                # 並行処理の開始
                await asyncio.gather(
                    self.send_audio(websocket),
                    self.handle_messages(websocket),
                    self.check_silence(websocket)
                )
                
        except Exception as e:
            print(f"エラー: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """リソースの解放"""
        print("クリーンアップ開始...")
        try:
            # 音声関連リソースの解放
            if hasattr(self, 'input_stream') and self.input_stream:
                try:
                    print("音声ストリームを閉じています...")
                    self.input_stream.stop_stream()
                    self.input_stream.close()
                except Exception as e:
                    print(f"音声ストリーム停止エラー: {e}")
            
            if hasattr(self, 'p') and self.p:
                try:
                    print("PyAudioを終了しています...")
                    self.p.terminate()
                except Exception as e:
                    print(f"PyAudio終了エラー: {e}")
            
            # 一時ファイルのクリーンアップ
            try:
                print("一時ファイルを削除しています...")
                self.clean_temp_files()
            except Exception as e:
                print(f"一時ファイル削除エラー: {e}")
            
            print("クリーンアップ完了")
        except Exception as e:
            print(f"クリーンアップ中に重大なエラーが発生: {str(e)}")

if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='リップシンク機能付きチャットシステム')
    parser.add_argument('--face', type=str, default='face.jpg',
                        help='顔画像ファイルのパス')
    args = parser.parse_args()
    
    try:
        chat = EnhancedChat(face_image_path=args.face)
        asyncio.run(chat.start())
    except ValueError as e:
        print(f"エラー: {e}")
        print("顔画像ファイルが存在し、正しく指定されていることを確認してください。")
        sys.exit(1)