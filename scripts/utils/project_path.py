"""
プロジェクトパスの自動検出ユーティリティ
ノートPC対応のため、ハードコードされたパスの代わりに使用
"""

import os
from pathlib import Path
import json


def get_project_root():
    """プロジェクトルートディレクトリを自動検出"""
    # このファイルの場所からプロジェクトルートを推定
    current_file = Path(__file__).resolve()
    # scripts/utils/project_path.py からプロジェクトルートへ
    project_root = current_file.parent.parent.parent
    
    # 設定ファイルから読み込む（もしあれば）
    config_file = project_root / 'config' / 'project_path.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'project_root' in config:
                    configured_path = Path(config['project_root'])
                    if configured_path.exists():
                        return configured_path
        except Exception:
            pass
    
    return project_root


def get_project_root_wsl():
    """WSL2環境用のプロジェクトルートパスを取得"""
    project_root = get_project_root()
    
    # 設定ファイルから読み込む（もしあれば）
    config_file = project_root / 'config' / 'project_path.json'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if 'project_root_wsl' in config:
                    return config['project_root_wsl']
        except Exception:
            pass
    
    # WindowsパスをWSLパスに変換
    project_str = str(project_root)
    # ドライブレターを検出
    if len(project_str) >= 2 and project_str[1] == ':':
        drive_letter = project_str[0].lower()
        wsl_path = project_str.replace('\\', '/').replace(f'{project_str[0]}:', f'/mnt/{drive_letter}')
        return wsl_path
    
    # 既にWSLパスの場合
    return str(project_root).replace('\\', '/')


def get_venv_wsl2_path():
    """WSL2仮想環境のパスを取得"""
    project_root_wsl = get_project_root_wsl()
    return f"{project_root_wsl}/venv_wsl2"


# プロジェクトルートを取得（グローバル変数として）
PROJECT_ROOT = get_project_root()
PROJECT_ROOT_WSL = get_project_root_wsl()
VENV_WSL2_PATH = get_venv_wsl2_path()





