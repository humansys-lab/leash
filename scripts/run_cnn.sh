# 現在の日時を取得してログファイル名を生成
current_time=$(TZ=Asia/Tokyo date +"%Y/%m/%d/%H:%M:%S")
log_filename="../data/logs/cnn_${current_time//\//_}.log"

# nohupでPythonスクリプトをバックグラウンドで実行し、ログをファイルにリダイレクト
nohup python3 baseline_cnn.py > "$log_filename" 2>&1 &