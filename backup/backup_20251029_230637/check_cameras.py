#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利用可能なカメラデバイスを確認するスクリプト
"""

import cv2

def check_available_cameras():
    """利用可能なカメラを確認"""
    print("=" * 50)
    print("利用可能なカメラデバイスを確認中...")
    print("=" * 50)
    
    available_cameras = []
    
    # カメラ0-9をテスト
    for camera_id in range(10):
        print(f"カメラ {camera_id} をテスト中...", end=" ")
        
        cap = cv2.VideoCapture(camera_id)
        if cap.isOpened():
            # フレームを取得してみる
            ret, frame = cap.read()
            if ret and frame is not None:
                print("[OK] 利用可能")
                available_cameras.append(camera_id)
                
                # カメラ情報を取得
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                print(f"   解像度: {int(width)}x{int(height)}")
                print(f"   FPS: {fps}")
            else:
                print("[NG] フレーム取得失敗")
        else:
            print("[NG] カメラを開けません")
        
        cap.release()
    
    print("\n" + "=" * 50)
    print("結果")
    print("=" * 50)
    
    if available_cameras:
        print(f"利用可能なカメラ: {available_cameras}")
        print(f"推奨カメラID: {available_cameras[0]}")
        print(f"\n起動コマンド:")
        print(f"python realtime_inspection.py --camera {available_cameras[0]}")
    else:
        print("[NG] 利用可能なカメラが見つかりませんでした")
        print("\n対処法:")
        print("1. カメラが接続されているか確認")
        print("2. 他のアプリケーションでカメラを使用していないか確認")
        print("3. カメラのドライバーが正しくインストールされているか確認")

if __name__ == "__main__":
    check_available_cameras()