
# 精度改善のための再学習スクリプト
python resin_washer_trainer.py --train --improve-accuracy

# 改善されたモデルでテスト
python realtime_inspection.py --model improved_model.h5 --camera 0