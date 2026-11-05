import time
import Pyro4
from qick import QickConfig, QickProgram

# ==============================================================================
# 1. 连接板卡
# ==============================================================================
BOARD_IP = "192.168.1.XXX"    # 改成你的 RFSoC4x2 板子 IP
NS_PORT = 8888
PROXY_NAME = "myqick"

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

ns = Pyro4.locateNS(host=BOARD_IP, port=NS_PORT)
soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
soccfg = QickConfig(soc.get_cfg())


# ==============================================================================
# 2. 定义 QICK v1 纯发波程序 (继承自 QickProgram)
# ==============================================================================
class SimpleCW_v1(QickProgram):
    def __init__(self, soccfg):
        super().__init__(soccfg)

    def initialize(self):
        ch = 1           # 1 对应 DAC_B (若接 DAC_A 则设为 0)
        nqz = 1          # 30 MHz 属于第 1 奈奎斯特区
        freq_mhz = 30.0  # 30 MHz，示波器可直接清晰观测
        gain = 30000     # 满量程附近
        pulse_len = 2000 # 脉冲长度 (tProc ticks)

        # QICK v1 声明发生器
        self.declare_gen(ch=ch, nqz=nqz)

        # QICK v1 脉冲寄存器配置
        self.set_pulse_registers(
            ch=ch,
            style="const",
            freq=self.freq2reg(freq_mhz, gen_ch=ch),
            gain=gain,
            length=pulse_len,
            phase=0
        )
        self.synci(200)

    def body(self):
        ch = 1
        # 触发脉冲并保持循环
        self.pulse(ch=ch)
        self.sync_all(2000)


# ==============================================================================
# 3. 编译与在 v1 固件上运行
# ==============================================================================
prog = SimpleCW_v1(soccfg)

# 编译生成 v1 的指令二进制码
binprog = prog.compile()

print("[运行] QICK v1 持续单音程序已编译...")
print("正在通过 DAC 通道 1 (DAC_B) 发送 30 MHz 信号...")

# 在 v1 中通过 soc 代理载入程序并启动 tProc
soc.load_bin_program(binprog)
soc.start_tproc()

print("输出中，请在示波器 CH2 (接 DAC_B) 观测。按 Ctrl+C 停止。")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n[中断] 用户停止。")
finally:
    soc.reset()
    print("[完成] 硬件已重置复位。")