#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習完了後のリモートPCへの自動転送機能
ノートPC対応: 学習が完了したら自動的に別のPCに転送
"""

import os
import sys
import json
import shutil
import zipfile
from pathlib import Path
from datetime import datetime
import subprocess

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


class RemoteTransfer:
    """リモートPCへの転送クラス"""
    
    def __init__(self, config_path=None):
        """初期化"""
        if config_path is None:
            # プロジェクトルートから設定ファイルを検索
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / 'config' / 'remote_transfer_config.json'
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.project_root = Path(__file__).resolve().parents[2]
    
    def load_config(self):
        """設定ファイルを読み込み"""
        if not self.config_path.exists():
            return self.get_default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"[WARN] 設定ファイルの読み込みエラー: {e}")
            return self.get_default_config()
    
    def get_default_config(self):
        """デフォルト設定を返す"""
        return {
            "enabled": False,
            "transfer_method": "network_share",
            "remote_pc": {
                "hostname": "192.168.1.100",
                "username": "",
                "password": "",
                "share_path": "\\\\192.168.1.100\\WasherInspection",
                "target_folder": "remote_models"
            },
            "transfer_items": {
                "models": True,
                "checkpoints": True,
                "logs": True,
                "config": True,
                "training_history": True
            },
            "auto_transfer_on_complete": True,
            "compress_before_transfer": True
        }
    
    def transfer_on_complete(self):
        """学習完了時に自動転送"""
        if not self.config.get('enabled', False):
            print("[INFO] リモート転送は無効になっています")
            return False
        
        if not self.config.get('auto_transfer_on_complete', False):
            print("[INFO] 自動転送は無効になっています")
            return False
        
        print("\n" + "=" * 60)
        print("学習完了: リモートPCへ転送を開始")
        print("=" * 60)
        
        try:
            # 転送するファイルを収集
            files_to_transfer = self.collect_files()
            
            if not files_to_transfer:
                print("[WARN] 転送するファイルが見つかりませんでした")
                return False
            
            # 転送を実行
            success = self.transfer_files(files_to_transfer)
            
            if success:
                print("\n" + "=" * 60)
                print("リモート転送完了！")
                print("=" * 60)
            else:
                print("\n" + "=" * 60)
                print("リモート転送に失敗しました")
                print("=" * 60)
            
            return success
            
        except Exception as e:
            print(f"[ERROR] 転送エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def collect_files(self):
        """転送するファイルを収集"""
        files_to_transfer = []
        transfer_items = self.config.get('transfer_items', {})
        
        # モデルファイル
        if transfer_items.get('models', True):
            model_patterns = [
                'clear_sparse_best_4class_*.h5',
                'clear_sparse_ensemble_4class_model_*.h5',
                'retrain_sparse_ensemble_4class_model_*.h5',
                '*.h5'
            ]
            for pattern in model_patterns:
                for file in self.project_root.glob(pattern):
                    if file.is_file():
                        files_to_transfer.append(file)
                        print(f"[INFO] 転送対象: {file.name}")
        
        # チェックポイント
        if transfer_items.get('checkpoints', True):
            checkpoint_dir = self.project_root / 'checkpoints'
            if checkpoint_dir.exists():
                for checkpoint_file in checkpoint_dir.rglob('*.h5'):
                    files_to_transfer.append(checkpoint_file)
                    print(f"[INFO] 転送対象: checkpoints/{checkpoint_file.relative_to(checkpoint_dir)}")
        
        # ログファイル
        if transfer_items.get('logs', True):
            logs_dir = self.project_root / 'logs'
            if logs_dir.exists():
                for log_file in logs_dir.glob('*.json'):
                    files_to_transfer.append(log_file)
                for log_file in logs_dir.glob('*.csv'):
                    files_to_transfer.append(log_file)
        
        # 学習履歴
        if transfer_items.get('training_history', True):
            for csv_file in self.project_root.glob('clear_sparse_training_log_*.csv'):
                files_to_transfer.append(csv_file)
        
        # 設定ファイル
        if transfer_items.get('config', True):
            config_dir = self.project_root / 'config'
            if config_dir.exists():
                for config_file in config_dir.glob('*.json'):
                    files_to_transfer.append(config_file)
        
        # アンサンブル情報ファイル
        for info_file in self.project_root.glob('*_info.json'):
            files_to_transfer.append(info_file)
        
        return files_to_transfer
    
    def transfer_files(self, files_to_transfer):
        """ファイルを転送"""
        method = self.config.get('transfer_method', 'network_share')
        
        if method == 'network_share':
            return self.transfer_via_network_share(files_to_transfer)
        elif method == 'scp':
            return self.transfer_via_scp(files_to_transfer)
        elif method == 'ftp':
            return self.transfer_via_ftp(files_to_transfer)
        else:
            print(f"[ERROR] 不明な転送方法: {method}")
            return False
    
    def transfer_via_network_share(self, files_to_transfer):
        """ネットワーク共有経由で転送"""
        remote_config = self.config.get('remote_pc', {})
        share_path = remote_config.get('share_path', '')
        target_folder = remote_config.get('target_folder', 'remote_models')
        
        if not share_path:
            print("[ERROR] ネットワーク共有パスが設定されていません")
            return False
        
        # 転送先ディレクトリを作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        target_dir = Path(share_path) / target_folder / timestamp
        
        try:
            # ネットワーク共有にアクセス
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] 転送先: {target_dir}")
            
            # 圧縮する場合
            if self.config.get('compress_before_transfer', True):
                print("[INFO] ファイルを圧縮中...")
                zip_path = target_dir / f'training_results_{timestamp}.zip'
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in files_to_transfer:
                        if file.exists():
                            # プロジェクトルートからの相対パスで保存
                            arcname = file.relative_to(self.project_root)
                            zipf.write(file, arcname)
                            print(f"[INFO] 圧縮: {arcname}")
                
                print(f"[INFO] 圧縮完了: {zip_path.name}")
                return True
            else:
                # 個別にコピー
                copied_count = 0
                for file in files_to_transfer:
                    if file.exists():
                        # ディレクトリ構造を保持してコピー
                        relative_path = file.relative_to(self.project_root)
                        target_file = target_dir / relative_path
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file, target_file)
                        copied_count += 1
                        print(f"[INFO] 転送: {relative_path}")
                
                print(f"[INFO] {copied_count} ファイルを転送しました")
                return True
                
        except Exception as e:
            print(f"[ERROR] ネットワーク共有への転送エラー: {e}")
            print("[INFO] ネットワーク接続と共有設定を確認してください")
            return False
    
    def transfer_via_scp(self, files_to_transfer):
        """SCP経由で転送"""
        remote_config = self.config.get('remote_pc', {})
        hostname = remote_config.get('hostname', '')
        username = remote_config.get('username', '')
        target_path = remote_config.get('target_folder', '~/WasherInspection')
        
        if not hostname:
            print("[ERROR] ホスト名が設定されていません")
            return False
        
        print("[INFO] SCP転送は実装中です")
        return False
    
    def transfer_via_ftp(self, files_to_transfer):
        """FTP経由で転送"""
        print("[INFO] FTP転送は実装中です")
        return False


def main():
    """メイン関数"""
    transfer = RemoteTransfer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # テストモード
        print("転送テストモード")
        files = transfer.collect_files()
        print(f"\n転送対象ファイル数: {len(files)}")
        for f in files[:10]:  # 最初の10個を表示
            print(f"  - {f.name}")
        if len(files) > 10:
            print(f"  ... 他 {len(files) - 10} ファイル")
    else:
        # 実際の転送
        transfer.transfer_on_complete()


if __name__ == '__main__':
    main()





