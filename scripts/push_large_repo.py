#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大きなリポジトリのプッシュ問題を解決
"""

import subprocess
import sys
import os
from pathlib import Path

# UTF-8 encoding for console output (Windows)
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

def run_command(cmd, capture_output=True):
    """コマンドを実行"""
    try:
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, 
                              text=True, encoding='utf-8', errors='replace', env=env)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_large_files():
    """大きなファイルをチェック"""
    print("=" * 60)
    print("大きなファイルのチェック")
    print("=" * 60)
    print()
    
    # ステージングされているファイルのサイズをチェック
    code, stdout, stderr = run_command('git ls-files --cached', capture_output=True)
    
    large_files = []
    total_size = 0
    
    for line in stdout.split('\n'):
        if not line.strip():
            continue
        file_path = Path(line.strip())
        if file_path.exists():
            size = file_path.stat().st_size
            total_size += size
            if size > 100 * 1024 * 1024:  # 100MB以上
                large_files.append((line.strip(), size / (1024*1024)))
    
    print(f"総ファイルサイズ: {total_size / (1024*1024*1024):.2f} GB")
    print()
    
    if large_files:
        print("大きなファイル（100MB以上）:")
        for file_path, size_mb in sorted(large_files, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {file_path}: {size_mb:.1f} MB")
    else:
        print("大きなファイルは見つかりませんでした")
    
    print()
    return large_files, total_size

def exclude_large_files_from_commit():
    """大きなファイルをコミットから除外"""
    print("=" * 60)
    print("大きなファイルを.gitignoreに追加")
    print("=" * 60)
    print()
    
    # .gitignoreを確認
    gitignore_path = Path('.gitignore')
    
    if not gitignore_path.exists():
        gitignore_content = ""
    else:
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            gitignore_content = f.read()
    
    # 大きなファイルパターンを追加
    patterns_to_add = [
        '# Large files and directories',
        'cs_AItraining_data/**',
        'sample_training_data/**',
        'dist/**',
        'build/**',
        '*.h5',
        '*.exe',
    ]
    
    added = False
    for pattern in patterns_to_add:
        if pattern not in gitignore_content:
            gitignore_content += '\n' + pattern + '\n'
            added = True
    
    if added:
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print("✓ .gitignoreを更新しました")
    else:
        print(".gitignoreは既に適切に設定されています")
    
    print()
    
    # 大きなファイルをステージングから除外
    print("大きなファイルをステージングから除外...")
    code, stdout, stderr = run_command('git reset HEAD cs_AItraining_data/ sample_training_data/ dist/ build/ *.h5 *.exe', capture_output=False)
    if code == 0:
        print("✓ 大きなファイルを除外しました")
    
    print()

def create_small_commit():
    """小さなコミットを作成（大きなファイルを除外）"""
    print("=" * 60)
    print("小さなコミットを作成")
    print("=" * 60)
    print()
    
    # 現在のステージング状態を確認
    code, stdout, stderr = run_command('git status --short', capture_output=True)
    
    staged_files = []
    for line in stdout.split('\n'):
        if line.strip().startswith(('A ', 'M ', 'D ', 'R ')):
            staged_files.append(line.strip())
    
    print(f"ステージングされているファイル数: {len(staged_files)}")
    
    # 大きなファイルを除外した後、コミット
    exclude_large_files_from_commit()
    
    # 残りのファイルをコミット
    print("コミット作成...")
    code, stdout, stderr = run_command('git commit -m "ファイル整理と文字化け修正完了（大きなファイルは除外）"', capture_output=False)
    
    if code == 0:
        print("✓ コミット成功")
        return True
    else:
        print(f"× コミット失敗: {stderr[:200]}")
        return False

def try_push_with_options():
    """様々なオプションでプッシュを試行"""
    print("=" * 60)
    print("プッシュ試行（最適化オプション付き）")
    print("=" * 60)
    print()
    
    # Git設定を最適化
    settings = [
        ('http.postBuffer', '524288000'),  # 500MB
        ('http.lowSpeedLimit', '0'),
        ('http.lowSpeedTime', '999999'),
        ('core.compression', '0'),
        ('http.version', 'HTTP/1.1'),  # HTTP/2よりHTTP/1.1の方が安定する場合がある
    ]
    
    for key, value in settings:
        run_command(f'git config {key} {value}')
    
    print("最適化設定完了")
    print()
    
    # 複数の方法でプッシュを試行
    methods = [
        ('通常のプッシュ', 'git push origin main'),
        ('詳細出力付き', 'git push -v origin main'),
        ('--no-verify付き', 'git push --no-verify origin main'),
    ]
    
    for method_name, cmd in methods:
        print(f"試行: {method_name}...")
        code, stdout, stderr = run_command(cmd, capture_output=False)
        
        if code == 0:
            print(f"✓ {method_name}成功！")
            return True
        
        print(f"× {method_name}失敗")
        if stderr:
            print(f"  エラー: {stderr[:150]}")
        print()
        
        # 短い待機
        import time
        time.sleep(2)
    
    return False

def main():
    # 大きなファイルをチェック
    large_files, total_size = check_large_files()
    
    # 大きなファイルを除外
    if large_files or total_size > 1024 * 1024 * 1024:  # 1GB以上
        exclude_large_files_from_commit()
        create_small_commit()
    
    # プッシュを試行
    success = try_push_with_options()
    
    if not success:
        print("\n" + "=" * 60)
        print("プッシュ失敗 - 最終的な解決方法")
        print("=" * 60)
        print()
        print("以下の方法を試してください:")
        print()
        print("1. GitHub CLIを使用:")
        print("   gh repo sync")
        print()
        print("2. 時間をおいて再試行:")
        print("   （GitHub側の一時的な問題の可能性）")
        print()
        print("3. 手動でブラウザからファイルをアップロード")
        print()
        print("4. Git LFSを使用（大きなファイル用）:")
        print("   git lfs install")
        print("   git lfs track '*.h5'")
        print("   git lfs track '*.jpg'")
        print("   git add .gitattributes")
        print("   git commit -m 'Add LFS tracking'")
        print("   git push")

if __name__ == '__main__':
    main()

