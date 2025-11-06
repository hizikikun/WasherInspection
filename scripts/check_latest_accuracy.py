#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最新の学習精度を確認（Clear Sparseモデル）
"""

import sys
import csv
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def get_latest_accuracy(csv_path):
    """CSVログから最新の精度を取得"""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row.get('epoch')]
            if rows:
                last_row = rows[-1]
                return {
                    'epoch': int(last_row.get('epoch', 0)),
                    'accuracy': float(last_row.get('accuracy', 0)),
                    'val_accuracy': float(last_row.get('val_accuracy', 0)),
                    'loss': float(last_row.get('loss', 0)),
                    'val_loss': float(last_row.get('val_loss', 0))
                }
    except Exception as e:
        print(f"エラー: {csv_path} - {e}")
        return None
    return None

def main():
    print("=" * 70)
    print("最新の学習精度確認（Clear Sparse 4-Class Ensemble）")
    print("=" * 70)
    print()
    
    # Clear Sparseモデルのログ
    logs = [
        ('logs/training/sparse/clear_sparse_training_log_4class_efficientnetb0.csv', 'EfficientNetB0'),
        ('logs/training/sparse/clear_sparse_training_log_4class_efficientnetb1.csv', 'EfficientNetB1'),
        ('logs/training/sparse/clear_sparse_training_log_4class_efficientnetb2.csv', 'EfficientNetB2'),
    ]
    
    print("📊 各モデルの最新エポック精度:")
    print("-" * 70)
    
    all_results = []
    for log_path, model_name in logs:
        log_file = Path(log_path)
        if log_file.exists():
            result = get_latest_accuracy(log_file)
            if result:
                all_results.append((model_name, result))
                print(f"\n{model_name}:")
                print(f"  📈 最終エポック: {result['epoch']}")
                print(f"  🎯 訓練精度: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                print(f"  ✅ バリデーション精度: {result['val_accuracy']:.4f} ({result['val_accuracy']*100:.2f}%)")
                print(f"  📉 損失: {result['loss']:.4f}")
                print(f"  📉 バリデーション損失: {result['val_loss']:.4f}")
        else:
            print(f"\n{model_name}: ❌ ログファイルが見つかりません ({log_path})")
    
    if all_results:
        print("\n" + "=" * 70)
        print("📊 精度サマリー:")
        print("-" * 70)
        avg_val_acc = sum(r['val_accuracy'] for _, r in all_results) / len(all_results)
        max_val_acc = max(r['val_accuracy'] for _, r in all_results)
        min_val_acc = min(r['val_accuracy'] for _, r in all_results)
        
        print(f"  平均バリデーション精度: {avg_val_acc:.4f} ({avg_val_acc*100:.2f}%)")
        print(f"  最高バリデーション精度: {max_val_acc:.4f} ({max_val_acc*100:.2f}%)")
        print(f"  最低バリデーション精度: {min_val_acc:.4f} ({min_val_acc*100:.2f}%)")
        
        print("\n" + "=" * 70)
        print("⚠️  注意:")
        print("  - 上記はバリデーション精度（検証データでの精度）です")
        print("  - アンサンブル精度は通常、個別モデルより高くなります")
        print("  - 最終的なテスト精度（アンサンブル精度）を確認するには、")
        print("    モデル評価スクリプトを実行してください")
        print("=" * 70)
    else:
        print("\n❌ 精度データが見つかりませんでした")

if __name__ == '__main__':
    main()













