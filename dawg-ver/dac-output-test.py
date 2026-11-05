import Pyro4
from qick import QickConfig
from qick.averager_program import AveragerProgram

# ==============================================================================
# 1. 连接板卡
# ==============================================================================
BOARD_IP = "192.168.1.XXX"    # 改成你的板子 IP
NS_PORT = 8888
PROXY_NAME = "myqick"

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

ns = Pyro4.locateNS(host=BOARD_IP, port=NS_PORT)
soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
soccfg = QickConfig(soc.get_cfg())


# ==============================================================================
# 2. 定义基于 AveragerProgram 的持续发波程序
# ==============================================================================
class ConstantToneProgram(AveragerProgram):
    def initialize(self):
        cfg = self.cfg
        ch = cfg["ch"]
        nqz = cfg["nqz"]
        freq = cfg["freq_mhz"]
        gain = cfg["gain"]

        # 声明发生器
        self.declare_gen(ch=ch, nqz=nqz)

        # 设置脉冲寄存器为恒定波 (const 模式)
        self.set_pulse_registers(
            ch=ch,
            style="const",
            freq=self.freq2reg(freq, gen_ch=ch),
            gain=gain,
            length=cfg["pulse_len"],
            phase=0
        )
        self.synci(200)

    def body(self):
        # 触发 DAC 输出
        self.pulse(ch=self.cfg["ch"])
        # 等待该脉冲播放完毕
        self.sync_all(self.cfg["pulse_len"])


# ==============================================================================
# 3. 参数配置与循环播放
# ==============================================================================
cw_config = {
    "ch": 1,               # 1 对应 DAC_B (若测 DAC_A 则改 0)
    "nqz": 1,              # 30 MHz 使用第 1 奈奎斯特区
    "freq_mhz": 30.0,      # 30 MHz
    "gain": 30000,         # 接近满量程输出
    "pulse_len": 1000,     # 单段脉冲长度 (tProc 周期单位，约 2~3 µs)
    "reps": 5000000,       # 设定极大的循环次数（500万次约持续播放十秒以上）
}

# 实例化程序 (AveragerProgram 标准接收 soccfg 和 config)
# 实例化程序
prog = ConstantToneProgram(soccfg, cw_config)

print(f"[运行] 正在通过 DAC 通道 {cw_config['ch']} 输出 {cw_config['freq_mhz']} MHz 信号...")
print("请在示波器 CH2 查看正弦波，按 Ctrl+C 可中断退出。")

# 关键：用 load_program/run 启动硬件，不等待任何 ADC 数据
prog.load_program(soc)
soc.start_readout(0) # 如果没有 readout 甚至不需要这行，直接 run
prog.run(soc)

# 让主进程挂起保持输出，示波器观察完毕后按 Ctrl+C 退出
try:
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[中断] 用户手动停止播放。")
finally:
    soc.reset()
    print("[完成] DAC 输出已重置关闭。")