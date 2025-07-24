import serial
import time
import csv
import os
from datetime import datetime

SERIAL_PORT = '/dev/c1_mpc_co2'
BAUDRATE = 115200
CSV_PATH = 'results/test.csv'
RB_COMMAND_PATH = 'results/rb_command.txt'  # 新增：R:B 指令文件路径

# 检查文件是否存在，决定是否写表头
def ensure_csv_header(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'co2'])

def main():
    ensure_csv_header(CSV_PATH)
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"开始监听串口 {SERIAL_PORT}，数据将写入 {CSV_PATH}")
    try:
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            while True:
                # 1. 监听 CO2 数据
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    try:
                        co2 = float(line)
                        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([ts, co2])
                        f.flush()
                        print(f"{ts}, {co2}")
                    except ValueError:
                        # 非法数据行，跳过
                        pass
                # 2. 检查是否有新的 R:B 指令需要发送
                if os.path.exists(RB_COMMAND_PATH) and os.path.getsize(RB_COMMAND_PATH) > 0:
                    with open(RB_COMMAND_PATH, 'r+') as rb_file:
                        rb_cmd = rb_file.read().strip()
                        if rb_cmd:
                            # 发送到串口，格式如 '120,80\n'
                            ser.write((rb_cmd + '\n').encode())
                            print(f"已发送 R:B 指令到串口: {rb_cmd}")
                            rb_file.seek(0)
                            rb_file.truncate()  # 清空文件
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("已停止监听。")
    finally:
        ser.close()

if __name__ == '__main__':
    main() 
