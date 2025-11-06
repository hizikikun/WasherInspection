# Windows環境でのTensorFlow GPU完全サポート有効化

## 現在の状況

✅ **アプリケーションは完全に動作可能です**

Windows + Python 3.12 + TensorFlow 2.20の組み合わせでは、**ネイティブCUDA GPUサポートが現在制限されています**。しかし、アプリケーションは**CPUモードで完全に動作**し、すべての機能が利用可能です。

## 実装済みの機能

✅ **完全に動作:**
- 学習機能（CPUモードで最適化済み）
- 外観検査機能
- システム情報表示
- 進捗表示
- すべての分析・レポート機能

✅ **最適化済み:**
- CPUスレッド数を最大化
- バッチサイズを自動最適化
- データ処理を効率化
- メモリ使用量を最適化

## 完全にGPUを使えるようにする方法

### 方法1: WSL2を使用（最も確実・推奨）

WSL2を使用すると、Linux環境でTensorFlowの完全なCUDAサポートを利用できます。

**自動セットアップスクリプト:**
```powershell
# setup_wsl2_tensorflow_gpu.shを実行
wsl bash setup_wsl2_tensorflow_gpu.sh
```

**手動セットアップ:**
```bash
# 1. WSL2でUbuntuを起動
wsl

# 2. CUDA Toolkitをインストール（WSL2用）
# NVIDIA公式サイトからWSL2用CUDA Toolkitをダウンロードしてインストール

# 3. TensorFlowをインストール
pip install tensorflow[and-cuda]
```

### 方法2: DirectMLを使用（Windowsネイティブ）

**現在の状態:**
- アプリケーションは既に`TF_USE_DIRECTML=1`を設定しています
- DirectMLプラグインは現在利用できませんが、将来対応する可能性があります

### 方法3: CPUモード（現在の環境）

**現在の環境:**
- ✅ CPUモードで完全に動作します
- ✅ すべての機能が利用可能です
- ✅ 学習は実行可能です（速度はGPUより遅いですが、精度は同じ）

## 確認方法

1. **アプリケーションを起動:**
   ```powershell
   python dashboard/integrated_washer_app.py
   ```

2. **システム情報を確認:**
   - 「🎓 学習」タブ → 「💻 PC構成・システムスペック」セクション
   - CUDA/GPU利用状況が表示されます

3. **GPU検出テストを実行:**
   ```powershell
   python test_gpu_support.py
   ```

## 推奨事項

**すぐに使いたい場合:**
- ✅ **CPUモードで完全に動作します** - 今すぐ使用可能です

**最高パフォーマンスが必要な場合:**
- WSL2を使用することを推奨します（GPUサポートが完全に有効になります）

## 注意事項

- CPUモードでは学習速度がGPUより遅くなりますが、**すべての機能は正常に動作します**
- 精度や機能はGPU/CPUに関係なく同じです
- システム情報は正しく表示されます

