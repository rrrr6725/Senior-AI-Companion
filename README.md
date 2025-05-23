# 高齢者のための AI 対話システム

卒業研究で開発した、高齢者の孤独感を軽減するための対話システムです。

## 概要
![Image](https://github.com/user-attachments/assets/9cd17569-bd45-4c60-8996-034a42fd2eac)

本システムは、高齢化社会における高齢者の孤独感軽減を目的として開発されました。  ChatGPTのRealtime APIを活用した自然かつ高速な対話生成と、Wav2Lipによるリップシンク動画生成を組み合わせることで、より親しみやすい対話環境を実現しています。  

単なる応答だけでなく、システム側からの能動的な話題提案機能も実装されており、自然な会話の継続を支援します。  会話が途切れると、システムが自動的に新しい話題を提案します。

## 主な特徴

- OpenAI Realtime APIによる文脈を考慮した自然な対話生成
- Wav2Lipによる音声に同期したアバターの口の動き生成
- 無音検出による能動的な話題提案機能（GPT-4o-mini使用）
- BFloat16形式採用による処理の高速化（約23%の処理時間削減）
- Tkinterベースの字幕付きビデオプレーヤー

## デモ動画

part1：雑談  
https://www.youtube.com/watch?v=_3XfP1Rm_UE    

part2：なぞなぞ  
https://www.youtube.com/watch?v=JQ3DcG7iMcw  


