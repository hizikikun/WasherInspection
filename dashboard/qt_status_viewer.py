#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from PyQt5 import QtWidgets, QtCore

STATUS_FILE = Path(__file__).resolve().parents[1] / 'logs' / 'training_status.json'


def human(seconds):
    if seconds is None:
        return "-"
    try:
        v = float(seconds)
    except Exception:
        return "-"
    if v >= 2 * 3600:
        return f"約{int(round(v/3600.0))}時間"
    if v >= 10 * 60:
        return f"約{int(round(v/60.0))}分"
    if v >= 60:
        mins = int(v // 60)
        secs = int(v % 60)
        return f"{mins}分{secs}秒"
    return f"{int(v)}秒"


def load_status():
    if not STATUS_FILE.exists():
        return None
    try:
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


class StatusWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('学習ステータス（スパースモデリング）')
        self.resize(760, 520)

        fontTitle = self.font(); fontTitle.setPointSize(12); fontTitle.setBold(True)
        fontMono = self.font(); fontMono.setPointSize(11)

        layout = QtWidgets.QVBoxLayout(self)
        self.title = QtWidgets.QLabel('学習の進行状況')
        self.title.setFont(fontTitle)
        layout.addWidget(self.title)
        
        # ステータス表示（学習中/モデル構築中など）
        self.statusLabel = QtWidgets.QLabel('ステータス: -')
        self.statusLabel.setFont(fontTitle)
        self.statusLabel.setStyleSheet('color:#0066cc; font-weight:bold;')
        layout.addWidget(self.statusLabel)

        # プログレスバー群
        self.pbEpoch = QtWidgets.QProgressBar()
        self.pbEpoch.setFormat('学習回の進み: %p%')
        self.pbTrain = QtWidgets.QProgressBar()
        self.pbTrain.setFormat('訓練の進行（エポック換算）: %p%')
        self.pbOverall = QtWidgets.QProgressBar()
        self.pbOverall.setFormat('全体の達成度: %p%')
        for pb in (self.pbEpoch, self.pbTrain, self.pbOverall):
            pb.setMinimum(0); pb.setMaximum(100)
            pb.setTextVisible(True)
            pb.setFixedHeight(22)
        layout.addWidget(self.pbEpoch)
        layout.addWidget(self.pbTrain)
        layout.addWidget(self.pbOverall)

        self.labels = []
        bullets = [
            'テモデル: -',
            'テ学習回: -',
            'テ訓練の進行: -',
            'テ全体の達成度: -',
            'テ経過時間: -',
            'テ残り時間(予測): -',
            'テ完了予定: -',
            'テ精度: -',
            'テ学習率(LR): -',
            'テCPU: -',
            'テCPU温度: -',
            'テメモリ: -',
            'テメモリ使用量: -',
            'テGPU: -',
            'テGPU温度: -',
            'テGPU電力: -',
        ]
        for t in bullets:
            lbl = QtWidgets.QLabel(t)
            lbl.setFont(fontMono)
            layout.addWidget(lbl)
            self.labels.append(lbl)

        # リソース使用率バー
        self.pbCPU = QtWidgets.QProgressBar(); self.pbCPU.setMinimum(0); self.pbCPU.setMaximum(100); self.pbCPU.setFormat('CPU 使用率: %p%')
        self.pbMEM = QtWidgets.QProgressBar(); self.pbMEM.setMinimum(0); self.pbMEM.setMaximum(100); self.pbMEM.setFormat('メモリ 使用率: %p%')
        self.pbGPU = QtWidgets.QProgressBar(); self.pbGPU.setMinimum(0); self.pbGPU.setMaximum(100); self.pbGPU.setFormat('GPU 使用率: %p%')
        for pb in (self.pbCPU, self.pbMEM, self.pbGPU):
            pb.setFixedHeight(18); pb.setTextVisible(True)
        layout.addWidget(self.pbCPU)
        layout.addWidget(self.pbMEM)
        layout.addWidget(self.pbGPU)

        self.updated = QtWidgets.QLabel('更新: -')
        self.updated.setStyleSheet('color:#666;')
        layout.addWidget(self.updated)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(2000)
        QtCore.QTimer.singleShot(100, self.refresh)

    def refresh(self):
        try:
            s = load_status()
            if not s:
                return
            
            # ステータス表示（学習中かどうか）
            stage = s.get('stage', '')
            section = s.get('section', '')
            message = s.get('message', '')
            
            # タイムスタンプから待機中かどうか判定（5分以上更新されていない場合は待機中とみなム）
            ts = s.get('timestamp')
            is_waiting = False
            if ts:
                try:
                    elapsed_since_update = time.time() - float(ts)
                    if elapsed_since_update > 300:  # 5分以上更新なし
                        is_waiting = True
                except Exception:
                    pass
            
            try:
                if not stage or stage == '-' or is_waiting:
                    if message and ('完了' in message or 'Complete' in message or 'Completed' in message):
                        status_text = 'ステータス: ⚪ 完了'
                        self.statusLabel.setStyleSheet('color:#999; font-weight:bold; font-size:12pt;')
                    else:
                        status_text = 'ステータス: ⚫ 待機中 / 停止中'
                        self.statusLabel.setStyleSheet('color:#888; font-weight:bold; font-size:12pt;')
                elif 'Training' in stage or 'training' in stage.lower() or 'training_epoch' in stage.lower() or '学習' in stage or 'train' in stage.lower():
                    status_text = f'ステータス: 🟢 学習中'
                    self.statusLabel.setStyleSheet('color:#00cc00; font-weight:bold; font-size:12pt;')
                elif 'Building' in stage or '構築' in stage or 'build' in stage.lower():
                    status_text = f'ステータス: 🟡 モデル構築中'
                    self.statusLabel.setStyleSheet('color:#ff9900; font-weight:bold; font-size:12pt;')
                elif 'Data Loading' in stage or '読み込み' in stage or 'Loading' in stage:
                    status_text = f'ステータス: 🔵 データ読み込み中'
                    self.statusLabel.setStyleSheet('color:#0066cc; font-weight:bold; font-size:12pt;')
                elif 'Evaluation' in stage or '評価' in stage or 'eval' in stage.lower():
                    status_text = f'ステータス: 🟣 評価中'
                    self.statusLabel.setStyleSheet('color:#9966cc; font-weight:bold; font-size:12pt;')
                elif 'Saving' in stage or '保存' in stage:
                    status_text = f'ステータス: 🔶 保存中'
                    self.statusLabel.setStyleSheet('color:#ff6600; font-weight:bold; font-size:12pt;')
                else:
                    status_text = f'ステータス: {stage}'
                    self.statusLabel.setStyleSheet('color:#666; font-weight:bold; font-size:12pt;')
                self.statusLabel.setText(status_text)
            except Exception as e:
                self.statusLabel.setText('ステータス: エラー')
            
            # 安全に値を取得
            try:
                model = s.get('model_name', '-') or '-'
                idx = s.get('model_index', 0) or 0
                tot = s.get('models_total', 0) or 0
                ep = s.get('epoch', 0) or 0
                epTot = s.get('epochs_total', 0) or 0
                epPct = s.get('epoch_progress_percent', 0) or 0
                trainPct = s.get('training_epoch_percent', 0) or 0
                overallPct = s.get('overall_progress_percent', 0) or 0
                overallElapsed = s.get('overall_elapsed_human') or human(s.get('overall_elapsed_seconds'))
                remain_secs = s.get('overall_remaining_est_seconds')
                overallRemain = s.get('overall_remaining_human') or human(remain_secs)
                eta = s.get('overall_eta_human') or human(s.get('overall_estimated_total_seconds'))
                acc = s.get('accuracy_percent')
                vacc = s.get('val_accuracy_percent')
                lr = str(s.get('learning_rate', '-')) or '-'
                cpu = s.get('cpu_percent')
                ctemp = s.get('cpu_temp_c')
                mem = s.get('mem_percent')
                mem_u = s.get('mem_used_mb')
                mem_t = s.get('mem_total_mb')
                gpu = s.get('gpu_util_percent')
                gmem_u = s.get('gpu_mem_used_mb')
                gmem_t = s.get('gpu_mem_total_mb')
                gtemp = s.get('gpu_temp_c')
                gpower = s.get('gpu_power_w')
            except Exception:
                return  # データ取得に失敗したらスキップ

            # 数値変換を安全に実行
            try:
                idx = int(idx) if idx else 0
                tot = int(tot) if tot else 0
                ep = int(ep) if ep else 0
                epTot = int(epTot) if epTot else 0
                epPct = float(epPct) if epPct else 0.0
                trainPct = float(trainPct) if trainPct else 0.0
                overallPct = float(overallPct) if overallPct else 0.0
            except Exception:
                pass

            try:
                self.labels[0].setText(f'テモデル: {model} ({idx}/{tot})')
            except Exception:
                self.labels[0].setText('テモデル: -')
            
            try:
                remain_ep = int(max(0, (epTot or 0) - (ep or 0)))
                self.labels[1].setText(f'テ学習回: 第{ep}/{epTot}（{epPct:.1f}%テ残り{remain_ep}）')
            except Exception:
                self.labels[1].setText('テ学習回: -')
            
            try:
                self.labels[2].setText(f'テ訓練の進行: {trainPct:.1f}%')
            except Exception:
                self.labels[2].setText('テ訓練の進行: -')
            
            try:
                self.labels[3].setText(f'テ全体の達成度: {int(round(overallPct))}%')
            except Exception:
                self.labels[3].setText('テ全体の達成度: -')
            
            try:
                self.labels[4].setText(f'テ経過時間: {overallElapsed}')
            except Exception:
                self.labels[4].setText('テ経過時間: -')
            
            # 完了予定の時計時刻
            try:
                ts_base = s.get('timestamp') or time.time()
                if remain_secs is not None:
                    try:
                        clock = time.strftime('%H:%M', time.localtime(ts_base + float(remain_secs)))
                    except Exception:
                        clock = '-'
                else:
                    clock = '-'
            except Exception:
                clock = '-'
            
            try:
                self.labels[5].setText(f'テあと: {overallRemain}')
            except Exception:
                self.labels[5].setText('テあと: -')
            
            try:
                self.labels[6].setText(f'テ終了見込み: {clock}（全体で{eta}）')
            except Exception:
                self.labels[6].setText('テ終了見込み: -')
            
            try:
                if acc is not None and vacc is not None:
                    self.labels[7].setText(f'テ精度: {float(acc):.2f}% | 検証: {float(vacc):.2f}%')
                else:
                    self.labels[7].setText('テ精度: -')
            except Exception:
                self.labels[7].setText('テ精度: -')
            
            try:
                self.labels[8].setText(f'テ学習率(LR): {lr}')
            except Exception:
                self.labels[8].setText('テ学習率(LR): -')
            
            # 使用率
            try:
                if cpu is not None:
                    self.labels[9].setText(f'テCPU: {float(cpu):.0f}%')
                    self.pbCPU.setValue(int(round(float(cpu))))
                else:
                    self.labels[9].setText('テCPU: -')
                    self.pbCPU.setValue(0)
            except Exception:
                self.labels[9].setText('テCPU: -')
                self.pbCPU.setValue(0)
            
            # CPU温度
            try:
                if ctemp is not None:
                    self.labels[10].setText(f'テCPU温度: {float(ctemp):.0f}℃')
                else:
                    self.labels[10].setText('テCPU温度: -')
            except Exception:
                self.labels[10].setText('テCPU温度: -')
            
            try:
                if mem is not None:
                    self.labels[11].setText(f'テメモリ: {float(mem):.0f}%')
                    self.pbMEM.setValue(int(round(float(mem))))
                else:
                    self.labels[11].setText('テメモリ: -')
                    self.pbMEM.setValue(0)
            except Exception:
                self.labels[11].setText('テメモリ: -')
                self.pbMEM.setValue(0)
            
            try:
                if mem_u is not None and mem_t is not None and mem_t > 0:
                    self.labels[12].setText(f'テメモリ使用量: {float(mem_u):.0f}/{float(mem_t):.0f} MB')
                else:
                    self.labels[12].setText('テメモリ使用量: -')
            except Exception:
                self.labels[12].setText('テメモリ使用量: -')
            
            try:
                if gpu is not None:
                    if gmem_u is not None and gmem_t is not None and gmem_t > 0:
                        self.labels[13].setText(f'テGPU: {float(gpu):.0f}% | VRAM: {gmem_u:.0f}/{gmem_t:.0f} MB')
                    else:
                        self.labels[13].setText(f'テGPU: {float(gpu):.0f}%')
                    self.pbGPU.setValue(int(round(float(gpu))))
                else:
                    self.labels[13].setText('テGPU: -')
                    self.pbGPU.setValue(0)
            except Exception:
                self.labels[13].setText('テGPU: -')
                self.pbGPU.setValue(0)

            # 温度テ電力
            try:
                if gtemp is not None:
                    self.labels[14].setText(f'テGPU温度: {float(gtemp):.0f}℃')
                else:
                    self.labels[14].setText('テGPU温度: -')
            except Exception:
                self.labels[14].setText('テGPU温度: -')
            
            try:
                if gpower is not None:
                    self.labels[15].setText(f'テGPU電力: {float(gpower):.1f} W')
                else:
                    self.labels[15].setText('テGPU電力: -')
            except Exception:
                self.labels[15].setText('テGPU電力: -')

            # プログレスバー値
            try:
                self.pbEpoch.setValue(int(round(float(epPct))))
            except Exception:
                self.pbEpoch.setValue(0)
            try:
                self.pbTrain.setValue(int(round(float(trainPct))))
            except Exception:
                self.pbTrain.setValue(0)
            try:
                self.pbOverall.setValue(int(round(float(overallPct))))
            except Exception:
                self.pbOverall.setValue(0)

            try:
                ts = s.get('timestamp')
                self.updated.setText('更新: ' + (time.strftime('%H:%M:%S', time.localtime(ts)) if ts else time.strftime('%H:%M:%S')))
            except Exception:
                self.updated.setText('更新: -')
        except Exception as e:
            # 完全にエラーが発生した場合でもクラッシュしないように
            try:
                self.statusLabel.setText('ステータス: 更新エラー')
                self.updated.setText(f'エラー: {str(e)[:50]}')
            except Exception:
                pass  # 最後の手段として何もしない


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    win = StatusWindow()
    win.show()
    app.exec_()

