#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
現在の学習精度を確認
"""

import sys
import os
from pathlib import Path
import csv
import json

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_latest_accuracy_from_log(csv_path):
    """CSVログから最新の精度を取得"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                return {
                    'epoch': int(last_row['epoch']),
                    'accuracy': float(last_row['accuracy']),
                    'val_accuracy': float(last_row['val_accuracy']),
                    'loss': float(last_row['loss']),
                    'val_loss': float(last_row['val_loss'])
                }
    except Exception as e:
        return None
    return None

def main():
    print("=" * 60)
    print("現在の学習精度確認")
    print("=" * 60)
    print()
    
    # 学習ステータスを確認
    status_path = Path('logs/training_status.json')
    if status_path.exists():
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                status = json.load(f)
                print("学習ステータス:")
                print(f"  状態: {status.get('stage', 'N/A')}")
                print(f"  進捗: {status.get('overall_progress_percent', 0):.1f}%")
                print(f"  経過時間: {status.get('overall_elapsed_human', 'N/A')}")
                if 'final_accuracy' in status:
                    print(f"  最終精度: {status['final_accuracy']:.4f} ({status['final_accuracy']*100:.2f}%)")
                print()
        except Exception:
            pass
    
    # 最新の学習ログを確認
    sparse_logs = [
        ('logs/training/sparse/sparse_training_log_4class_efficientnetb0.csv', 'EfficientNetB0'),
        ('logs/training/sparse/sparse_training_log_4class_efficientnetb1.csv', 'EfficientNetB1'),
        ('logs/training/sparse/sparse_training_log_4class_efficientnetb2.csv', 'EfficientNetB2'),
    ]
    
    print("各モデルの最新エポック精度:")
    print("-" * 60)
    
    for log_path, model_name in sparse_logs:
        log_file = Path(log_path)
        if log_file.exists():
            result = get_latest_accuracy_from_log(log_file)
            if result:
                print(f"{model_name}:")
                print(f"  エポック: {result['epoch']}")
                print(f"  訓練精度: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                print(f"  バリデーション精度: {result['val_accuracy']:.4f} ({result['val_accuracy']*100:.2f}%)")
                print(f"  損失: {result['loss']:.4f}")
                print(f"  バリデーション損失: {result['val_loss']:.4f}")
                print()
    
    print("=" * 60)
    print("注意:")
    print("  - 上記はバリデーション精度です")
    print("  - 最終的なテスト精度（アンサンブル精度）を確認するには、")
    print("    モデルを評価する必要があります")
    print("=" * 60)

if __name__ == '__main__':
    main()




















