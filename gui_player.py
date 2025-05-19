#!/usr/bin/env python3
"""
player.pyをバックエンドとして使用した動画プレーヤー（Linux専用）
VLCとTkinterを使用して動画再生と字幕表示を行う
AI会話システム用に拡張した字幕表示機能付き
"""

import os
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import vlc
import threading
import time

class EnhancedVideoPlayer:
    """
    player.pyの機能拡張版。Tkinterウィンドウに対応。
    """
    def __init__(self):
        """VideoPlayerクラスの初期化"""
        os.environ['LIBVA_MESSAGING_LEVEL'] = '0'
        self.instance = vlc.Instance('--quiet', '--no-xlib')
        self.player = None
        self.lock = threading.Lock()
        self.window_id = None
        self._playing = False
        self._stop_requested = False

    def load_video(self, video_path: str):
        """動画の読み込み"""
        if not os.path.exists(video_path):
            return False, f"エラー: ファイルが見つかりません: {video_path}"

        try:
            with self.lock:
                # 既存のプレーヤーを停止・解放
                if self.player is not None:
                    try:
                        self.player.stop()
                        time.sleep(0.1)  # 停止を確実にするために少し待機
                        self.player.release()
                    except Exception as e:
                        print(f"プレーヤー停止エラー: {e}")
                        # エラーが発生しても続行
                
                print(f"新しい動画プレーヤーを作成: {video_path}")
                self.player = self.instance.media_player_new()
                media = self.instance.media_new(video_path)
                self.player.set_media(media)
                
                # Tkinterウィンドウに埋め込み表示する場合
                if self.window_id:
                    self.player.set_xwindow(self.window_id)
                    print(f"ウィンドウID {self.window_id} を設定")
                
                return True, "読み込み成功"
                
        except Exception as e:
            print(f"詳細な読み込みエラー: {str(e)}")
            return False, f"読み込みエラー: {str(e)}"

    def play(self):
        """動画の再生"""
        if self.player:
            result = self.player.play()
            if result == -1:
                print("再生開始に失敗しました")
                return False
                
            # 再生開始を確認するために少し待つ
            time.sleep(0.1)
            self._playing = True
            print("再生を開始しました")
            return True
        return False
            
    def pause(self):
        """動画の一時停止"""
        if self.player:
            self.player.pause()
            self._playing = False

    def toggle_play_pause(self):
        """再生/一時停止の切り替え"""
        if not self.player:
            return False
            
        if self.is_playing():
            self.pause()
        else:
            self.play()
        return True

    def is_playing(self):
        """再生状態を確認"""
        if self.player:
            return self.player.is_playing()
        return False

    def set_xwindow(self, window_id):
        """Tkinterのキャンバスに埋め込む"""
        self.window_id = window_id
        if self.player:
            self.player.set_xwindow(window_id)

    def cleanup(self):
        """リソースの解放"""
        try:
            self._stop_requested = True
            if self.player is not None:
                self.player.stop()
                self.player.release()
                self.player = None
            if self.instance is not None:
                self.instance.release()
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

    def __del__(self):
        self.cleanup()


class GUIVideoPlayer:
    def __init__(self, root, video_path=None):
        """
        GUIVideoPlayerクラスの初期化
        
        Args:
            root: Tkinterのルートウィンドウ
            video_path: 再生する動画ファイルのパス
        """
        self.root = root
        self.root.title("AI会話システム")
        self.root.geometry("2800x2300")
        self.root.configure(bg="white")
        
        # レイアウト設定
        self._setup_ui()
        
        # 動画のパスを設定
        if video_path is None:
            self.video_path = None
        else:
            self.video_path = Path(video_path)
        
        # EnhancedVideoPlayerインスタンスの作成
        self.video_player = EnhancedVideoPlayer()
        
        # UI の描画後に動画を読み込むために少し遅延させる
        self.root.update()
        self.root.after(100, self.initialize_player)
        
        # 終了時の処理を登録
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def _setup_ui(self):
        """UIコンポーネントの初期化と配置"""
        # スタイル設定 - フォントサイズを2.5倍に
        style = ttk.Style()
        style.configure('TButton', font=('song ti', 35, 'bold'))
        
        # メインフレーム
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=25)
        
        # 動画キャンバス
        self.video_frame = ttk.Frame(self.main_frame)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_canvas = tk.Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 字幕エリア - フォントサイズを2.5倍に
        self.subtitle_var = tk.StringVar()
        self.subtitle_var.set("")
        
        self.subtitle_label = ttk.Label(
            self.main_frame, 
            textvariable=self.subtitle_var,
            font=('song ti', 45, "bold"),  # フォントサイズを2.5倍に
            background="white",
            foreground="black",
            wraplength=2700  # テキストの折り返し幅を設定
        )
        self.subtitle_label.pack(pady=12, fill=tk.X)
        
        # コントロールエリア
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=12)
        
        # 再生/一時停止ボタン - サイズを大きく
        self.play_pause_button = ttk.Button(
            self.control_frame, 
            text="再生/一時停止", 
            command=self.toggle_play_pause,
            width=20  # ボタン幅を大きく
        )
        self.play_pause_button.pack(side=tk.LEFT, padx=25, pady=25)
        
        # 終了ボタン - サイズを大きく
        self.quit_button = ttk.Button(
            self.control_frame, 
            text="終了", 
            command=self.on_closing,
            width=20  # ボタン幅を大きく
        )
        self.quit_button.pack(side=tk.RIGHT, padx=25, pady=25)
    
    def initialize_player(self):
        """動画プレーヤーの初期化と動画の読み込み"""
        # キャンバスのウィンドウIDを取得しプレーヤーに設定
        win_id = self.video_canvas.winfo_id()
        if not win_id:
            self.subtitle_var.set("エラー: キャンバスのウィンドウIDが無効です")
            return
            
        self.video_player.set_xwindow(win_id)
        
        # 動画ファイルが指定されている場合は読み込む
        if self.video_path:
            self.load_and_play(str(self.video_path))
            
            # 動画長を取得し、それに基づいて自動終了時間を設定
            # VLCプレーヤーから動画長を取得（ミリ秒単位）
            def check_duration_and_set_timer():
                if self.video_player.player:
                    duration = self.video_player.player.get_length()
                    if duration > 0:
                        # 動画長 + 少しのバッファ7500ms後に終了
                        self.root.after(duration + 7500, self.on_closing)
                    else:
                        # 長さを取得できない場合は再試行
                        self.root.after(100, check_duration_and_set_timer)
            
            # 動画の長さ情報が利用可能になるまで少し待ってから確認
            self.root.after(500, check_duration_and_set_timer)
    
    def load_and_play(self, video_path):
        """新しい動画を読み込んで再生する"""
        if not video_path:
            print("Error: No video path provided")
            return False
            
        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}")
            return False
            
        print(f"Loading video: {video_path}")
        self.video_path = Path(video_path)
        
        # 既存の再生を停止
        if self.video_player.player:
            self.video_player.player.stop()
        
        # 新しい動画をロード
        success, message = self.video_player.load_video(str(self.video_path))
        
        if not success:
            print(f"Failed to load video: {message}")
            self.subtitle_var.set(message)
            return False
            
        # ファイル名をタイトルに設定
        self.root.title(f"AI会話システム - {self.video_path.name}")
        
        # 確実に再生開始
        self.video_player.play()
        print(f"Video playback started for {video_path}")
        
        # 再生状態に合わせてボタンテキストを更新
        self.update_play_pause_button()
        return True
    
    def update_subtitle(self, text):
        """字幕テキストを更新する"""
        current_text = self.subtitle_var.get()
        # 単純に置き換えるか、テキストを追加するかはアプリケーションの要件に応じて調整
        self.subtitle_var.set(text)
        
    def append_subtitle(self, text):
        """字幕テキストを追加する（ストリーミング対応）"""
        current_text = self.subtitle_var.get()
        # 最後の文章のみ表示するか、一定長さを保つなど要件に応じて調整
        if len(current_text) > 500:  # 文字数制限
            current_text = current_text[-500:]
        self.subtitle_var.set(current_text + text)
    
    def toggle_play_pause(self):
        """再生/一時停止の切り替え"""
        if self.video_player.toggle_play_pause():
            self.update_play_pause_button()
    
    def update_play_pause_button(self):
        """再生状態に応じてボタンのテキストを更新"""
        if self.video_player.is_playing():
            self.play_pause_button.configure(text="一時停止")
        else:
            self.play_pause_button.configure(text="再生")
    
    def on_closing(self):
        """ウィンドウ終了時の処理"""
        try:
            if self.video_player:
                self.video_player.cleanup()
        finally:
            self.root.destroy()
    
    def get_root(self):
        """Tkinterのルートウィンドウを取得する"""
        return self.root


def main():
    """メイン関数"""
    import argparse
    
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='動画と字幕を表示するシンプルプレーヤー')
    parser.add_argument('video_path', nargs='?', help='表示する動画ファイルのパス')
    parser.add_argument('--subtitle', help='表示する字幕テキスト', default='')
    args = parser.parse_args()
    
    root = tk.Tk()
    
    # スタイル設定
    style = ttk.Style()
    style.theme_use('clam')
    
    # インスタンス化
    player = GUIVideoPlayer(root, args.video_path)
    
    # 字幕が指定されていれば設定
    if args.subtitle:
        player.update_subtitle(args.subtitle)
        print(f"字幕テキストを設定: {args.subtitle[:30]}...")
    
    # メインループ開始
    root.mainloop()


if __name__ == "__main__":
    main()
