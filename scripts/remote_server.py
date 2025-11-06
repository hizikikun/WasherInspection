#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモート操作サーバー
別PCから学習を開始・停止・監視できるWebインターフェース
"""

import os
import sys
import json
import time
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_file
from flask_cors import CORS
import socket

# UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

app = Flask(__name__)
CORS(app)  # CORSを有効化（別PCからのアクセスを許可）

# グローバル状態
training_status = {
    'is_running': False,
    'start_time': None,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'accuracy': 0.0,
    'loss': 0.0,
    'logs': [],
    'process': None
}

# プロジェクトルート
project_root = Path(__file__).resolve().parents[1]
log_file = project_root / 'logs' / 'training.log'
log_file.parent.mkdir(parents=True, exist_ok=True)

# HTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WasherInspection リモート操作</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            padding: 30px;
        }
        h1 {
            color: #667eea;
            margin-bottom: 30px;
            text-align: center;
            font-size: 2.5em;
        }
        .status-card {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #667eea;
        }
        .status-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            padding: 10px;
            background: white;
            border-radius: 5px;
        }
        .status-label {
            font-weight: bold;
            color: #555;
        }
        .status-value {
            color: #667eea;
            font-weight: bold;
        }
        .status-running { color: #28a745; }
        .status-stopped { color: #dc3545; }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .controls {
            display: flex;
            gap: 15px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .btn-start {
            background: #28a745;
            color: white;
        }
        .btn-start:hover { background: #218838; transform: translateY(-2px); }
        .btn-stop {
            background: #dc3545;
            color: white;
        }
        .btn-stop:hover { background: #c82333; transform: translateY(-2px); }
        .btn-refresh {
            background: #17a2b8;
            color: white;
        }
        .btn-refresh:hover { background: #138496; transform: translateY(-2px); }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .logs {
            background: #1e1e1e;
            color: #d4d4d4;
            padding: 20px;
            border-radius: 10px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            margin-top: 20px;
        }
        .log-entry {
            margin-bottom: 5px;
            padding: 5px;
            border-left: 3px solid transparent;
        }
        .log-entry.error { border-left-color: #dc3545; color: #f8d7da; }
        .log-entry.info { border-left-color: #17a2b8; color: #d1ecf1; }
        .log-entry.success { border-left-color: #28a745; color: #d4edda; }
        .info-box {
            background: #e7f3ff;
            border: 1px solid #b3d9ff;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }
        .info-box h3 {
            color: #004085;
            margin-bottom: 10px;
        }
        .info-box code {
            background: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 WasherInspection リモート操作</h1>
        
        <div class="status-card">
            <h2>学習状態</h2>
            <div class="status-row">
                <span class="status-label">ステータス:</span>
                <span class="status-value" id="status">停止中</span>
            </div>
            <div class="status-row">
                <span class="status-label">進捗:</span>
                <span class="status-value" id="progress">0%</span>
            </div>
            <div class="status-row">
                <span class="status-label">エポック:</span>
                <span class="status-value" id="epoch">0 / 0</span>
            </div>
            <div class="status-row">
                <span class="status-label">精度:</span>
                <span class="status-value" id="accuracy">0.00%</span>
            </div>
            <div class="status-row">
                <span class="status-label">損失:</span>
                <span class="status-value" id="loss">0.0000</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressBar" style="width: 0%">0%</div>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-start" id="btnStart" onclick="startTraining()">学習開始</button>
            <button class="btn-stop" id="btnStop" onclick="stopTraining()" disabled>学習停止</button>
            <button class="btn-refresh" onclick="refreshStatus()">状態更新</button>
        </div>
        
        <div class="info-box">
            <h3>📡 アクセス情報</h3>
            <p>このページにアクセスするには:</p>
            <p><code>http://{{ server_ip }}:{{ server_port }}</code></p>
            <p>同じネットワーク上の他のPCからアクセスできます。</p>
        </div>
        
        <div class="logs" id="logs">
            <div class="log-entry info">ログを読み込み中...</div>
        </div>
    </div>
    
    <script>
        let refreshInterval;
        
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent = data.is_running ? '実行中' : '停止中';
                    document.getElementById('status').className = 'status-value ' + (data.is_running ? 'status-running' : 'status-stopped');
                    document.getElementById('progress').textContent = data.progress + '%';
                    document.getElementById('epoch').textContent = data.current_epoch + ' / ' + data.total_epochs;
                    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
                    document.getElementById('loss').textContent = data.loss.toFixed(4);
                    
                    const progressBar = document.getElementById('progressBar');
                    progressBar.style.width = data.progress + '%';
                    progressBar.textContent = data.progress + '%';
                    
                    document.getElementById('btnStart').disabled = data.is_running;
                    document.getElementById('btnStop').disabled = !data.is_running;
                    
                    // ログを更新
                    const logsDiv = document.getElementById('logs');
                    logsDiv.innerHTML = '';
                    data.logs.slice(-50).forEach(log => {
                        const entry = document.createElement('div');
                        entry.className = 'log-entry ' + (log.type || 'info');
                        entry.textContent = log.message;
                        logsDiv.appendChild(entry);
                    });
                    logsDiv.scrollTop = logsDiv.scrollHeight;
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }
        
        function startTraining() {
            fetch('/api/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    updateStatus();
                })
                .catch(error => {
                    alert('エラー: ' + error);
                });
        }
        
        function stopTraining() {
            if (confirm('学習を停止しますか？')) {
                fetch('/api/stop', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        alert(data.message);
                        updateStatus();
                    })
                    .catch(error => {
                        alert('エラー: ' + error);
                    });
            }
        }
        
        function refreshStatus() {
            updateStatus();
        }
        
        // 初期化
        updateStatus();
        refreshInterval = setInterval(updateStatus, 2000); // 2秒ごとに更新
        
        // ページを離れるときにインターバルをクリア
        window.addEventListener('beforeunload', () => {
            clearInterval(refreshInterval);
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """メインページ"""
    # サーバーのIPアドレスを取得
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    # 外部IPを取得（簡易版）
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        server_ip = s.getsockname()[0]
        s.close()
    except:
        server_ip = local_ip
    
    return render_template_string(
        HTML_TEMPLATE,
        server_ip=server_ip,
        server_port=5000
    )


@app.route('/api/status')
def get_status():
    """学習状態を取得"""
    # ログファイルから最新の進捗を読み取り
    logs = []
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines[-50:]:  # 最後の50行
                    if line.strip():
                        log_type = 'info'
                        if 'error' in line.lower() or 'エラー' in line:
                            log_type = 'error'
                        elif 'completed' in line.lower() or '完了' in line:
                            log_type = 'success'
                        logs.append({'message': line.strip(), 'type': log_type})
        except Exception as e:
            logs.append({'message': f'ログ読み込みエラー: {e}', 'type': 'error'})
    
    return jsonify({
        'is_running': training_status['is_running'],
        'start_time': training_status['start_time'],
        'progress': training_status['progress'],
        'current_epoch': training_status['current_epoch'],
        'total_epochs': training_status['total_epochs'],
        'accuracy': training_status['accuracy'],
        'loss': training_status['loss'],
        'logs': logs
    })


@app.route('/api/start', methods=['POST'])
def start_training():
    """学習を開始"""
    if training_status['is_running']:
        return jsonify({'success': False, 'message': '学習は既に実行中です'})
    
    try:
        # 学習スクリプトを起動
        train_script = project_root / 'scripts' / 'train_4class_sparse_ensemble.py'
        
        if not train_script.exists():
            return jsonify({'success': False, 'message': '学習スクリプトが見つかりません'})
        
        # バックグラウンドで実行
        process = subprocess.Popen(
            [sys.executable, str(train_script)],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        training_status['is_running'] = True
        training_status['start_time'] = datetime.now().isoformat()
        training_status['process'] = process
        
        # ログ監視スレッドを起動
        threading.Thread(target=monitor_training, args=(process,), daemon=True).start()
        
        return jsonify({'success': True, 'message': '学習を開始しました'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'エラー: {str(e)}'})


@app.route('/api/stop', methods=['POST'])
def stop_training():
    """学習を停止"""
    if not training_status['is_running']:
        return jsonify({'success': False, 'message': '学習は実行されていません'})
    
    try:
        if training_status['process']:
            training_status['process'].terminate()
            training_status['process'].wait(timeout=5)
        
        training_status['is_running'] = False
        training_status['process'] = None
        
        return jsonify({'success': True, 'message': '学習を停止しました'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'エラー: {str(e)}'})


@app.route('/api/logs')
def get_logs():
    """ログファイルを取得"""
    if log_file.exists():
        try:
            return send_file(str(log_file), mimetype='text/plain')
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'ログファイルが見つかりません'}), 404


def monitor_training(process):
    """学習プロセスを監視"""
    while process.poll() is None:
        time.sleep(1)
    
    # プロセスが終了した
    training_status['is_running'] = False
    training_status['process'] = None


def get_local_ip():
    """ローカルIPアドレスを取得"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='WasherInspection リモート操作サーバー')
    parser.add_argument('--host', default='0.0.0.0', help='ホストアドレス (デフォルト: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='ポート番号 (デフォルト: 5000)')
    parser.add_argument('--debug', action='store_true', help='デバッグモード')
    
    args = parser.parse_args()
    
    local_ip = get_local_ip()
    
    print("=" * 60)
    print("WasherInspection リモート操作サーバー")
    print("=" * 60)
    print(f"ローカルアクセス: http://127.0.0.1:{args.port}")
    print(f"リモートアクセス: http://{local_ip}:{args.port}")
    print("=" * 60)
    print("サーバーを起動しています...")
    print("Ctrl+C で停止")
    print()
    
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")


if __name__ == '__main__':
    main()





