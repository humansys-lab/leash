import sys
import datetime
import pytz
import time

def train():
    # ここにモデル学習の進捗を出力する処理を記述
    for i in range(30):
        print(f"Epoch {i+1}")
        time.sleep(1)

def main():
    tz_japan = pytz.timezone('Asia/Tokyo')
    
    # 現在の日本時間を取得してログファイル名を生成
    current_time = datetime.datetime.now(tz_japan).strftime("%Y_%m_%d_%H_%M_%S")
    log_filename = f"../data/logs/lstm_{current_time}.log"
    
    # バッファリングなしでログファイルに出力をリダイレクト
    with open(log_filename, "w", buffering=1) as file:
        old_stdout = sys.stdout
        sys.stdout = file
        
        # 出力したい内容（例）
        train()
        
        # stdoutを元に戻す
        sys.stdout = old_stdout
    

if __name__ == '__main__':
    main()