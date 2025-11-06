"""学習開始機能の簡素化版（テンプレート）"""

def start_training_simple(self):
    """学習を開始（シンプル版）"""
    print("[INFO] ========== 学習開始 ==========")
    
    try:
        # 1. 既に実行中かチェック
        if hasattr(self, 'training_worker') and self.training_worker is not None:
            if hasattr(self.training_worker, 'isRunning') and self.training_worker.isRunning():
                print("[INFO] 学習は既に実行中です")
                try:
                    self.statusBar().showMessage('学習は既に実行中です', 3000)
                except:
                    pass
                return
        
        # 2. リソース設定（デフォルト値）
        if not hasattr(self, 'resource_config') or not self.resource_config:
            self.resource_config = {
                'batch_size': 32,
                'workers': 8,
                'max_epochs': 200,
            }
        
        # 3. WSL2モード確認
        use_wsl2 = False
        try:
            if hasattr(self, 'use_wsl2_checkbox') and self.use_wsl2_checkbox is not None:
                use_wsl2 = self.use_wsl2_checkbox.isChecked()
        except:
            pass
        
        # 4. TrainingWorkerを作成
        print("[INFO] TrainingWorkerを作成中...")
        from dashboard.integrated_washer_app import TrainingWorker
        
        self.training_worker = TrainingWorker(
            resource_config=self.resource_config,
            use_wsl2=use_wsl2
        )
        
        # 5. シグナル接続
        try:
            self.training_worker.finished.connect(self.training_finished)
            self.training_worker.log_message.connect(self.append_training_log)
        except Exception as e:
            print(f"[WARN] シグナル接続エラー: {e}")
        
        # 6. 確認ダイアログ（簡素化）
        try:
            from PyQt5.QtWidgets import QMessageBox
            env_mode = "WSL2 GPUモード" if use_wsl2 else "Windows CPUモード"
            reply = QMessageBox.question(
                self,
                '学習開始確認',
                f'学習を開始しますか？\n\n実行環境: {env_mode}',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply != QMessageBox.Yes:
                print("[INFO] ユーザーがキャンセルしました")
                self.training_worker = None
                return
        except Exception as e:
            print(f"[WARN] 確認ダイアログエラー: {e}")
            # エラーが発生しても続行
        
        # 7. ワーカーを起動
        print("[INFO] ワーカーを起動中...")
        try:
            self.training_worker.start()
            print("[INFO] ワーカーを起動しました")
        except Exception as e:
            print(f"[ERROR] ワーカー起動エラー: {e}")
            import traceback
            traceback.print_exc()
            try:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, 'エラー', f'ワーカーの起動に失敗しました:\n{str(e)}')
            except:
                pass
            self.training_worker = None
            return
        
        # 8. UI更新
        try:
            if hasattr(self, 'start_training_btn') and self.start_training_btn is not None:
                self.start_training_btn.setEnabled(False)
            if hasattr(self, 'stop_training_btn') and self.stop_training_btn is not None:
                self.stop_training_btn.setEnabled(True)
            self.statusBar().showMessage('学習を開始しました', 3000)
        except:
            pass
        
        print("[INFO] ========== 学習開始処理完了 ==========")
        
    except Exception as e:
        print(f"[ERROR] 学習開始エラー: {e}")
        import traceback
        traceback.print_exc()
        try:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.critical(self, 'エラー', f'学習開始時にエラーが発生しました:\n{str(e)}')
        except:
            pass
        try:
            if hasattr(self, 'start_training_btn') and self.start_training_btn is not None:
                self.start_training_btn.setEnabled(True)
        except:
            pass





