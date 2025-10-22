#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
モデル読み込みテスト
"""

import os
import tensorflow as tf
from tensorflow import keras

def test_model_loading():
    """モデル読み込みテスト"""
    print("=" * 60)
    print("モデル読み込みテスト")
    print("=" * 60)
    
    model_path = "resin_washer_model/resin_washer_model.h5"
    
    # ファイル存在確認
    print(f"モデルパス: {model_path}")
    print(f"ファイル存在: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        print(f"ファイルサイズ: {os.path.getsize(model_path)} bytes")
        
        try:
            print("モデル読み込み中...")
            model = keras.models.load_model(model_path)
            print("モデル読み込み成功！")
            print(f"モデル入力形状: {model.input_shape}")
            print(f"モデル出力形状: {model.output_shape}")
            return True
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            return False
    else:
        print("エラー: モデルファイルが見つかりません")
        return False

if __name__ == "__main__":
    test_model_loading()






