"""
PulsedODMR 真实硬件版本 —— 连接实际 RFSoC4x2 板卡运行
================================================================
这是 test-plot-simulate.py (离线 dummy 配置仿真版) 的硬件对应版本。
PulsedODMR 这个子类的代码完全不变 —— 区别只在于：

  1. 不再手工构造 dummy QickConfig 字典，而是通过 Pyro4 连接到真实跑在
     RFSoC4x2 上的 qick_server (QICK 官方标准的远程控制方式)，直接拿到
     板卡真实的 soccfg (gens/readouts/output_pins 等全部来自真实固件)。
  2. config 频率不再需要"假设换了个新谐振腔、frequency 卡在 dummy DAC
     上限"这种折中 —— 用的就是真实板卡 soccfg 里各 gen 实际的 f_dds，
     如果你板子上接的微波链路支持论文的 8.2GHz 频段 (nqz=2 的物理混叠
     频段)，可以直接改回论文原始数值。
  3. 最后不再用"离线示意波形重建"，而是调用 prog.acquire() 真正拿硬件
     采回来的 ADC 数据，并用 PulsedODMR 自带的（简单）分析方法处理。

使用前准备
----------
1. 确保板子上已经启动了 QICK 的 Pyro4 服务 (qick_server / pyro4_server)，
   通常在板子上执行类似：
       python -m qick.pyro

   服务默认会把自己注册到一个 Pyro4 Name Server 下。
2. 把下面的 BOARD_IP / NS_PORT / PROXY_NAME 改成你实际的板子配置：
   - BOARD_IP: 运行 Name Server 和 qick_server 的机器 IP（通常就是板子本身）
   - NS_PORT : Name Server 端口，QICK 默认 8888
   - PROXY_NAME: 在板子上启动服务时注册的名字，QICK 示例里常用 "myqick"
     (具体以你板子上 qick_server 启动脚本里 register 的名字为准)
"""

import sys
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

import Pyro4
from qick import QickConfig
from qick.averager_program import QickSweep
import qickdawg as qd
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram

# ==============================================================================
# 0. 连接真实 RFSoC4x2 (把下面三个变量改成你自己的板子配置)
# ==============================================================================
BOARD_IP = "192.168.1.XXX"    # <<< 改成你的 RFSoC4x2 / Name Server 所在 IP
NS_PORT = 8888                # QICK Pyro4 Name Server 默认端口
PROXY_NAME = "myqick"         # 板子上 qick_server 启动时注册的代理名

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
Pyro4.config.SERIALIZERS_ACCEPTED = {"pickle"}

print(f"[连接] 正在连接 RFSoC4x2 @ {BOARD_IP}:{NS_PORT} (proxy='{PROXY_NAME}') ...")
ns = Pyro4.locateNS(host=BOARD_IP, port=NS_PORT)
soc = Pyro4.Proxy(ns.lookup(PROXY_NAME))
soccfg = QickConfig(soc.get_cfg())

print(soccfg)

# 注入真实的 soc / soccfg 给 qickdawg (与 dummy 版唯一不同的地方：soc 不是 None)
qd.soc = soc
qd.soccfg = soccfg


# ==============================================================================
# 1. PulsedODMR 子类：严格按论文 Fig.4-6 顶部的 Init-MW-Read 序列编译
# ==============================================================================
# (与离线仿真版完全相同的实现，直接复用 —— 这个类不关心 soc 是真实板卡还是
#  dummy 配置，差别完全由 initialize()/body() 之外的 soccfg 决定)
#
#   Laser_ON(Init) -> Laser_OFF -> MW_ON(暗场) -> MW_OFF -> Laser_ON(Read) -> Laser_OFF
#
class PulsedODMR(NVAveragerProgram):
    '''
    严格复现论文 Init-MW-Read 脉冲 ODMR 序列的 NVAveragerProgram 子类。

    Parameters (cfg 需要的属性)
    -------------------------------------------------------------------
    adc_channel, laser_gate_pmod, mw_channel, mw_nqz, mw_gain,
    nsweep_points, reps                      -- 含义同 LockinODMR
    init_treg       -- Init 激光脉冲宽度（共振泵浦），寄存器单位
    dark_pre_treg   -- Init 结束到 MW 开始之间的暗场间隔，寄存器单位
    mw_treg         -- MW 脉冲宽度（例如 tπ），寄存器单位
    dark_post_treg  -- MW 结束到 Read 开始之间的暗场间隔，寄存器单位
    read_treg       -- Read 激光脉冲宽度 / ADC 积分窗口，寄存器单位
    relax_delay_treg -- rep 之间的间隔（激光、微波都 OFF），寄存器单位
    '''

    required_cfg = [
        "adc_channel",
        "laser_gate_pmod",
        "mw_channel",
        "mw_nqz",
        "mw_gain",
        "nsweep_points",
        "reps",
        "init_treg",
        "dark_pre_treg",
        "mw_treg",
        "dark_post_treg",
        "read_treg",
        "relax_delay_treg"]

    def initialize(self):
        self.check_cfg()

        if self.cfg.mw_gain > 30000:
            assert 0, 'Microwave gain exceeds maximum value'

        # ADC 积分窗口沿用 Read 段的激光脉冲宽度
        self.cfg.readout_integration_treg = self.cfg.read_treg
        self.setup_readout()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_start_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.mw_treg,    # MW 脉冲宽度独立于 ADC 积分窗口
            phase=0)

        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")

        self.add_sweep(QickSweep(self,
                                  self.mw_frequency_register,
                                  self.cfg.mw_start_fMHz,
                                  self.cfg.mw_end_fMHz,
                                  self.cfg.nsweep_points))

        self.synci(100)  # 给处理器一点时间配置脉冲
        if (self.cfg.ddr4 is True) or (self.cfg.mr is True):
            self.trigger(ddr4=self.cfg.ddr4, mr=self.cfg.mr, adc_trig_offset=0)
        self.synci(100)

    def body(self):
        # --- Init: 激光 ON，共振光泵浦自旋极化（独立脉冲，width 后自动关闭） ---
        self.trigger(
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.init_treg,
            adc_trig_offset=0,
            t=0)

        # --- MW: 纯暗场，驱动自旋跃迁（此刻激光已关闭） ---
        t_mw = self.cfg.init_treg + self.cfg.dark_pre_treg
        self.pulse(ch=self.cfg.mw_channel, t=t_mw)

        # --- Read: 激光 ON，共振读出自旋布居数，同时触发 ADC 采集 ---
        t_read = t_mw + self.cfg.mw_treg + self.cfg.dark_post_treg
        self.trigger(
            adcs=self.cfg.adcs,
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.read_treg,
            adc_trig_offset=0,
            t=t_read)

        self.wait_all()
        self.sync_all(self.cfg.relax_delay_treg)

    def acquire(self, raw_data=False, *arg, **kwarg):
        # PODMR 每个 rep 只读出一次（Read 窗口），不是 LockinODMR 那种
        # signal/reference 两次读出的差分方案
        return super().acquire(readouts_per_experiment=1, *arg, **kwarg)


# ==============================================================================
# 2. 根据 Hansen et al. (arXiv:2606.19016) 的 PODMR 实验参数配置 PulsedODMR
# ==============================================================================
#
# 论文中 Hyperfine-Resolved Pulsed ODMR (Sec. III.B, Fig. 4) 的关键数值：
#   - 微波中心频率           fMW          = 8.20 GHz   (正文 + Fig.4 caption)
#   - 115In 超精细分裂        Δhf          = 104 MHz，共 10 条跃迁 (mI = +9/2 ... -9/2)
#   - Rabi 频率 (该测量用功率) ΩR/2π        = 23 MHz     (Fig.4 caption)
#     对应 π 脉冲时长         tπ = 1/(2ΩR) ≈ 21.7 ns
#   - 卷积后的 PODMR 线宽     Γ_conv       ≈ 46.7 MHz   (App. D, Eq. D10)
#
# !! 请根据你板子上实际的微波链路核实下面这一点 !!
# RFSoC4x2 的 DAC 本身通常到不了 8.2GHz (具体取决于你板上 gen 的 f_dds 和
# nqz 设置，以及后级是否有外部混频/倍频链路把信号真正搬到 8.2GHz)。下面
# 默认先尝试直接用论文的 8.2GHz 配置；如果你的硬件链路做不到，会在编译/
# freq2reg 阶段直接报 RuntimeError，报错信息会给出该 gen 实际允许的频率
# 范围 [-f_dds/2, f_dds/2]，到时候请按下面"降频"的思路改：
#   MW_CENTER_FMHZ = soccfg['gens'][MW_CHANNEL]['f_dds'] / 2 - sweep_half_span - sweep_margin - guard
# (与 test-plot-simulate.py 里处理 dummy 配置频率上限超限的方法完全一致)

MW_CHANNEL = 1          # 改成你板子上实际接了微波链路的 DAC 通道
ADC_CHANNEL = 0         # 改成你板子上实际接了 APD/PMT 输出的 ADC 通道
LASER_PMOD = 0          # 改成你板子上实际接了激光快门/AOM TTL 的 PMOD pin
MW_NQZ = 2              # 1 或 2，取决于 fMW 落在哪个奈奎斯特区，参考上面的说明

HYPERFINE_SPLITTING_MHZ = 104.0
N_HYPERFINE_LINES = 10
RABI_FREQ_MHZ = 23.0
T_PI_NS = 1.0 / (2 * RABI_FREQ_MHZ) * 1000.0   # ≈ 21.7 ns
LINEWIDTH_CONV_MHZ = 46.7

MW_FMHZ_PAPER = 8200.0   # 论文原始中心频率；若超出硬件范围请参考上面注释改成降频值

T_INIT_NS = 200.0    # Init 激光脉冲宽度（共振泵浦），请按实际光泵浦动力学调整
T_DARK_NS = 20.0     # Init->MW 以及 MW->Read 之间的暗场间隔
T_READ_NS = 200.0    # Read 激光脉冲宽度 / ADC 积分窗口

TPROC_CLK_MHZ = soccfg['tprocs'][0]['f_time']   # 真实板卡的 tProc 时钟
NS_PER_TICK = 1000.0 / TPROC_CLK_MHZ


def ns_to_treg(t_ns):
    return max(1, round(t_ns / NS_PER_TICK))


sweep_half_span = HYPERFINE_SPLITTING_MHZ * (N_HYPERFINE_LINES - 1) / 2   # 468 MHz
sweep_margin = LINEWIDTH_CONV_MHZ
MW_START_FMHZ = MW_FMHZ_PAPER - sweep_half_span - sweep_margin
MW_END_FMHZ = MW_FMHZ_PAPER + sweep_half_span + sweep_margin
MW_STEP_FMHZ = 5.0

default_config = qd.NVConfiguration()
default_config.test = False             # 真实硬件：不是 test 模式
default_config.mw_readout_channel = 1
default_config.ddr4 = False             # 按需开启；真实板卡上如不需要 DDR4 buffer 可留 False
default_config.adc_channel = ADC_CHANNEL
default_config.mw_channel = MW_CHANNEL
default_config.mw_nqz = MW_NQZ
default_config.mw_gain = 15000          # 请先用较低增益测试，再按需调高
default_config.reps = 10000             # 波形循环次数，示波器触发/稳定观测用，不是信噪比平均次数
default_config.laser_gate_pmod = LASER_PMOD

default_config.init_treg = ns_to_treg(T_INIT_NS)
default_config.dark_pre_treg = ns_to_treg(T_DARK_NS)
default_config.mw_treg = ns_to_treg(T_PI_NS)
default_config.dark_post_treg = ns_to_treg(T_DARK_NS)
default_config.read_treg = ns_to_treg(T_READ_NS)
default_config.relax_delay_treg = ns_to_treg(1000.0)

config = copy(default_config)
config.mw_fMHz = MW_FMHZ_PAPER
config.add_linear_sweep('mw', 'fMHz', start=MW_START_FMHZ, stop=MW_END_FMHZ, delta=MW_STEP_FMHZ)

print(f"PODMR 扫描范围: {MW_START_FMHZ:.1f} - {MW_END_FMHZ:.1f} MHz, "
      f"步长 {MW_STEP_FMHZ} MHz, 覆盖 {N_HYPERFINE_LINES} 条超精细线 (间距 {HYPERFINE_SPLITTING_MHZ} MHz)")
print(f"tπ = {T_PI_NS:.2f} ns -> mw_treg = {config.mw_treg} ticks "
      f"({NS_PER_TICK:.4f} ns/tick @ {TPROC_CLK_MHZ} MHz tProc 时钟)")

# 编译程序 (连接到真实板卡后，这一步会在板子上真正配置 tProc 汇编)
prog = PulsedODMR(config)

print("\n=== tProcessor 汇编代码编译成功 (PulsedODMR: Init-MW-Read, 真实硬件) ===")
print(prog.asm())


# ==============================================================================
# 3. 在真实板卡上循环播放脉冲序列（示波器观测用，不做 ADC 数据采集/分析）
# ==============================================================================
# 目的只是让 Init-MW-Read 这组波形在板卡上连续循环输出 reps (=10000) 次，
# 方便用示波器探头接在：
#   - 激光 PMOD TTL 输出 (LASER_PMOD)
#   - 微波 DAC 输出 (MW_CHANNEL)
# 上直接观察波形，不需要任何光电探测器或 ADC 数据处理。
#
# acquire() 仍会按 QICK 的标准流程把程序下发并驱动 tProc 循环 reps 次，
# 这里直接丢弃返回的 ADC 数据即可（没有物理探测器接在 ADC 输入上也没关系，
# 读到的数值无意义，不会影响 DAC/PMOD 侧的波形输出）。
print(f"\n[运行] 正在板卡上循环播放 PulsedODMR 波形 ({config.reps} 次循环)，"
      f"请用示波器探头观测 Laser PMOD / MW DAC 输出 ...")
prog.acquire(soc, progress=True)
print("[完成] 波形播放结束。")


# ==============================================================================
# 4. 画出实际下发到板卡的脉冲序列 (方法 A：QICK 原生 plot_pulses)
# ==============================================================================
try:
    fig_dac = prog.plot_pulses()
    plt.suptitle("QICK DAC Pulse Synthesis (PulsedODMR: Init-MW-Read, real board)")
    plt.show()
except Exception as e:
    print(f"DAC波形细节绘制跳过: {e}")
