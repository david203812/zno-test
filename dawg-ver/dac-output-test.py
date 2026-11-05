import Pyro4
from qick import QickConfig, QickProgram

# ==============================================================================
# 1. 连接板卡 (修改为你的实际 IP 和参数)
# ==============================================================================
BOARD_IP = "192.168.1.XXX"
NS_PORT = 8888
PROXY_NAME = "myqick"

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

ns = Pyro4.locateNS(host=BOARD_IP, port=NS_PORT)
soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
soccfg = QickConfig(soc.get_cfg())


# ==============================================================================
# 2. 定义持续单音程序
# ==============================================================================
class ConstantToneProgram(QickProgram):
    def initialize(self):
        ch = self.cfg["ch"]
        nqz = self.cfg["nqz"]
        freq = self.cfg["freq_mhz"]
        gain = self.cfg["gain"]

        # 声明发生器
        self.declare_gen(ch=ch, nqz=nqz)

        # 配置一个恒定脉冲 (style='const')
        # length 设置为一个较长的周期 (例如 1000 个时钟周期)
        self.set_pulse_registers(
            ch=ch,
            style="const",
            freq=self.freq2reg(freq, gen_ch=ch),
            gain=gain,
            length=1000,
            phase=0
        )
        self.synci(200)

    def body(self):
        # 触发脉冲
        self.pulse(ch=self.cfg["ch"])
        # 同步等待，留出微小缓冲后立即进入下一轮循环，形成连续波
        self.sync_all(1000)


# ==============================================================================
# 3. 参数配置与下发运行
# ==============================================================================
cw_config = {
    "ch": 1,             # 1 对应 DAC_B (若接 DAC_A 则改为 0)
    "nqz": 1,            # 30 MHz 使用第 1 奈奎斯特区
    "freq_mhz": 30.0,    # 30 MHz，200 MHz 示波器可无失真显示
    "gain": 30000,       # 接近满量程 (30000 / 32767)，示波器上约 0.8 ~ 1.0 Vpp
    "reps": 1,
}

prog = ConstantToneProgram(soccfg, cw_config)

print(f"[运行] 正在通过 DAC 通道 {cw_config['ch']} 输出 {cw_config['freq_mhz']} MHz 信号...")
print("请在示波器 CH2 (接 DAC_B) 查看波形。")
print("提示：如果想停止输出，在终端按 Ctrl+C 中断或运行 soc.reset()。")

# load_program() 会将 tProc 程序载入并循环运行，不等待数据采集返回
soc.load_program(prog)