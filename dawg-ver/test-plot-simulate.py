import sys
from copy import copy
import matplotlib.pyplot as plt
import numpy as np

from qick import QickConfig
from qick.averager_program import QickSweep
import qickdawg as qd
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram

# ==============================================================================
# 1. 完整 QICK 0.2.x 虚拟硬件字典 (补齐 output_pins 门控定义)
# ==============================================================================
def get_complete_dummy_cfg():
    gen_template = {
        'type': 'axis_signal_gen_v4',
        'f_fabric': 384.0,       # MHz
        'fs': 6144.0,            # MHz
        'f_dds': 6144.0,
        'interpolation': 16,
        'samps_per_clk': 16,
        'maxlen': 65536,
        'b_dds': 32,
        'fs_mult': 16,
        'fdds_div': 1,
        'fs_div': 1,
        'has_mixer': False,
    }

    gens = []
    for i in range(3):
        g = copy(gen_template)
        g['tproc_ch'] = i
        gens.append(g)

    ro_template = {
        'ro_type': 'iq',
        'type': 'axis_readout_v2',
        'f_fabric': 384.0,
        'fs': 4096.0,
        'f_dds': 4096.0,
        'f_output': 384.0,
        'decimation': 8,
        'samps_per_clk': 8,
        'buffer_len': 1024,
        'b_dds': 32,
        'fs_mult': 1,
        'fdds_div': 1,
        'fs_div': 1,
        'has_dwnproc': False,
        'has_outsel': True,
        'has_modesel': True,
        'adc': 0,
        'tproc_ch': 0,
        'trigger_port': 0,
        'trigger_bit': 0,
    }

    readouts = []
    for i in range(2):
        r = copy(ro_template)
        r['tproc_ch'] = i
        r['adc'] = i
        r['trigger_port'] = 0
        r['trigger_bit'] = i
        readouts.append(r)

    cfg = {
        'board': 'RFSoC4x2',
        'refclk_freq': 384.0,
        'tprocs': [
            {
                'type': 'axis_tproc64x32_x8',
                'f_time': 384.0,
                'pmem_size': 65536,
                'dmem_size': 1024,
                # output_pins 必须是 (porttype, port, pin, name) 的元组列表。
                # qick 的 QickConfig.description() 和 NVAveragerProgram.trigger_no_off()
                # 都用 pincfg[1] / pincfg[2] 做位置索引取 port/bit (元组解包)，
                # 之前写成 dict {'port':.., 'bit':..} 会被当作 dict 的 key 去查找，
                # 因此报 KeyError: 1
                'output_pins': [
                    ('pmod', 0, 0, 'PMOD0_0'),
                    ('pmod', 0, 1, 'PMOD0_1'),
                    ('pmod', 0, 2, 'PMOD0_2'),
                    ('pmod', 0, 3, 'PMOD0_3'),
                ],
            }
        ],
        'gens': gens,
        'readouts': readouts,
        'ddr4_buf': {
            'tproc_ch': 0,
            'burst_len': 64,
            'trigger_port': 0,
            'trigger_bit': 2,
        },
        'mr_buf': {
            'tproc_ch': 0,
            'trigger_port': 0,
            'trigger_bit': 3,
        },
        'b_time': 32,
        'dual_dac': False,
    }
    return cfg

# 注入全局环境
soccfg = QickConfig(get_complete_dummy_cfg())
qd.soc = None
qd.soccfg = soccfg


# ==============================================================================
# 2. PulsedODMR 子类
# ==============================================================================
#
# 时序条 Init | MW | Read 是三段互不重叠的脉冲——
#
#   Laser_ON(Init) -> Laser_OFF -> MW_ON(暗场) -> MW_OFF -> Laser_ON(Read) -> Laser_OFF
#
# Init 用共振光脉冲把施主自旋极化到 |up> 或 |down>，MW 必须在纯暗场中驱动自旋
# 跃迁 (若激光仍亮着，持续光泵浦会不断把自旋拉回极化态，掩盖/破坏微波翻转的
# 信号)，Read 用第二个共振光脉冲读出微波驱动后的自旋布居数。
#
# 关键实现点：
#   - trigger(pins=..., width=...) 会在 t 时刻把 pin 拉高，并在 width 之后
#     自动拉低 —— 是一个独立的 ON/OFF 脉冲，因此 Init 和 Read 各自调用一次
#     trigger() 就能得到两段互不相连、各自完整开关的激光脉冲。
#   - MW 脉冲宽度 (mw_treg) 与 ADC 积分窗口 (read_treg) 是两个独立寄存器，
#     不再像 LockinODMR 那样共用同一个 readout_integration_treg。
#   - Read 段的 trigger() 同时带上 adcs=self.cfg.adcs，在激光读出窗口内触发
#     ADC 采集。Init 段不触发 ADC（只做光学初始化，不采集数据）。

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
        "relax_delay_treg"
    ]

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
        # --- Init: Laser ON，pump to polarize the spins ---
        self.trigger(
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.init_treg,
            adc_trig_offset=0,
            t=0)

        # --- MW: 纯暗场，驱动自旋跃迁（Laser OFF） ---
        t_mw = self.cfg.init_treg + self.cfg.dark_pre_treg
        self.pulse(ch=self.cfg.mw_channel, t=t_mw)

        # --- Read: Laser ON，共振读出自旋布居数，同时触发 ADC 采集 ---
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
# 3. 根据 Hansen et al. (arXiv:2606.19016) 的 PODMR 实验参数配置 PulsedODMR
# ==============================================================================
#
# 论文中 Hyperfine-Resolved Pulsed ODMR (Sec. III.B, Fig. 4) 的关键数值：
#   - 微波中心频率           fMW          = 8.20 GHz   (正文 + Fig.4 caption, 论文实际用值)
#   - 115In 超精细分裂        Δhf          = 104 MHz，共 10 条跃迁 (mI = +9/2 ... -9/2)
#   - Rabi 频率 (该测量用功率) ΩR/2π        = 23 MHz     (Fig.4 caption)
#     对应 π 脉冲时长         tπ = 1/(2ΩR) ≈ 21.7 ns    (App. D 用同一功率推出 tπ=21.7ns)
#   - 卷积后的 PODMR 线宽     Γ_conv       ≈ 46.7 MHz   (App. D, Eq. D10)
#   - 实测的核自旋极化        P_N          = -0.19       (正文，Fig.4(c) 对应)
#
# 论文正文/附录没有单独给出 Init、Read 激光脉冲各自的具体时长(只定性描述为
# "共振初始化脉冲"/"共振读出脉冲")，这里选用代表性数值(与 Appendix A 中
# 光学抽运测量的量级大致相当，但并非论文逐字数值)，并在打印信息里明确标注
# 为假设值；μ波脉宽则严格采用论文给出的 tπ≈21.7ns。
#
# ！！本脚本的硬件约束调整！！
# 这块 dummy 硬件配置里微波生成器的 DDS 时钟只有 f_dds = 6144 MHz，论文的
# fMW=8.2GHz 直接超过了这个 DAC 理论上能合成的最高频率，不能直接套用。
# 这里假设："我们换了一个新的微波谐振腔，其谐振频率恰好卡在这块 dummy 板子
# DAC 能合成的频率上限"，因此重新计算一个贴着 dummy DAC 合成上限的微波中心
# 频率，同时完整保留论文里 10 条超精细线、104MHz 间距、46.7MHz 线宽等相对
# 结构不变。

TPROC_CLK_MHZ = 384.0            # tProc 时钟 (f_time)，来自 dummy 配置
NS_PER_TICK = 1000.0 / TPROC_CLK_MHZ   # 每个寄存器 tick 对应的纳秒数


def ns_to_treg(t_ns):
    """把物理时间(ns)按 tProc 时钟换算成寄存器 tick 数(至少为 1)。"""
    return max(1, round(t_ns / NS_PER_TICK))


# ---- 论文物理参数 (超精细结构、线宽、Rabi频率、核极化 —— 这些是donor本身的
#      内禀物理量，与"用什么载波频率去合成/测量它"无关，因此原样保留) --------
HYPERFINE_SPLITTING_MHZ = 104.0  # 115In 超精细分裂
N_HYPERFINE_LINES = 10            # mI = 9/2 ... -9/2
RABI_FREQ_MHZ = 23.0              # ΩR/2π, Fig.4 caption 使用的功率点
T_PI_NS = 1.0 / (2 * RABI_FREQ_MHZ) * 1000.0   # tπ = 1/(2 ΩR) ≈ 21.7 ns
LINEWIDTH_CONV_MHZ = 46.7          # Γ_conv, App. D, Eq. D10
NUCLEAR_POLARIZATION = -0.19       # P_N, 正文

# ---- Init / Read 激光脉冲时长、暗场间隔 (论文未逐字给出，此处为代表性假设值) ----
T_INIT_NS = 200.0    # Init 激光脉冲宽度（共振泵浦），假设值
T_DARK_NS = 20.0     # Init->MW 以及 MW->Read 之间的暗场间隔，假设值（远小于T2*用来快速切换）
T_READ_NS = 200.0    # Read 激光脉冲宽度 / ADC 积分窗口，假设值

# ---- 根据 dummy 板子的实际合成能力重新设定微波中心频率 ----------------------
# 注意：QICK 的 freq2reg 接受的寄存器频率值，合法范围始终是 [-f_dds/2, +f_dds/2]，
# 这个范围和 nqz 无关——nqz 只决定 DAC 物理输出会把这个寄存器频率"混叠"到第几个
# 奈奎斯特区（nqz=2 时物理频率 = f_dds - f_reg，从而落到 (f_dds/2, f_dds) 这个
# 更高的物理频段），但传给 config.mw_fMHz / freq2reg 的值本身不能超过 f_dds/2。
# 所以这块 dummy 板子微波通道能"配置"的频率上限其实是 f_dds/2，而不是 f_dds。

sweep_half_span = HYPERFINE_SPLITTING_MHZ * (N_HYPERFINE_LINES - 1) / 2   # 468 MHz
sweep_margin = LINEWIDTH_CONV_MHZ                                         # 46.7 MHz
FREQ_GUARD_MHZ = 50.0   # 额外留出的安全余量，避免扫描边缘贴着 DAC 合成上限

MW_NQZ = 2   # 使用第二奈奎斯特区（物理输出会混叠到高频段），但寄存器频率值仍需 <= f_dds/2

_f_dds_mw_probe = soccfg['gens'][1]['f_dds']   # MHz, 与下方 mw_channel=1 一致
MAX_SYNTH_FMHZ = _f_dds_mw_probe / 2.0   # freq2reg 的合法上限是 f_dds/2，不是 f_dds

MW_CENTER_FMHZ = MAX_SYNTH_FMHZ - sweep_half_span - sweep_margin - FREQ_GUARD_MHZ
MW_START_FMHZ = MW_CENTER_FMHZ - sweep_half_span - sweep_margin
MW_END_FMHZ = MW_CENTER_FMHZ + sweep_half_span + sweep_margin
MW_STEP_FMHZ = 5.0   # 远小于线宽(46.7MHz)，保证能分辨每条谱线的形状

print(f"[硬件上限仿真] dummy 板子微波通道 f_dds/2 = {MAX_SYNTH_FMHZ:.1f} MHz (nqz={MW_NQZ}), "
      f"将微波中心频率从论文的 8200 MHz 降到 {MW_CENTER_FMHZ:.1f} MHz "
      f"(扫描上边界 {MW_END_FMHZ:.1f} MHz，留有 {FREQ_GUARD_MHZ:.0f} MHz 安全余量)")

default_config = qd.NVConfiguration()
default_config.test = True
default_config.mw_readout_channel = 1
default_config.ddr4 = True
default_config.ddr4_channel = 0
default_config.n_ddr4_chunks = 1200
default_config.adc_channel = 0
default_config.mw_channel = 1
default_config.mw_nqz = MW_NQZ   # =2, 把可合成频率上限推到 f_dds 附近，匹配"新谐振腔"的上限假设
default_config.mw_gain = 15000          # 任意的中等功率 DAC 增益 (无法与论文的 W 功率直接换算)
default_config.reps = 100000            # 代表性的平均次数，量级参考光子计数类实验
default_config.laser_gate_pmod = 0

# --- PulsedODMR 专用的 Init-MW-Read 时序寄存器 ---
default_config.init_treg = ns_to_treg(T_INIT_NS)
default_config.dark_pre_treg = ns_to_treg(T_DARK_NS)
default_config.mw_treg = ns_to_treg(T_PI_NS)          # MW 脉冲宽度 = 论文 tπ
default_config.dark_post_treg = ns_to_treg(T_DARK_NS)
default_config.read_treg = ns_to_treg(T_READ_NS)
default_config.relax_delay_treg = ns_to_treg(1000.0)   # rep 间隔，论文未给出，取 ~1us 量级

config = copy(default_config)
config.mw_fMHz = MW_CENTER_FMHZ
config.add_linear_sweep('mw', 'fMHz', start=MW_START_FMHZ, stop=MW_END_FMHZ, delta=MW_STEP_FMHZ)

print(f"PODMR 扫描范围: {MW_START_FMHZ:.1f} - {MW_END_FMHZ:.1f} MHz, "
      f"步长 {MW_STEP_FMHZ} MHz, 覆盖 {N_HYPERFINE_LINES} 条超精细线 (间距 {HYPERFINE_SPLITTING_MHZ} MHz)")
print(f"tπ = {T_PI_NS:.2f} ns -> mw_treg = {config.mw_treg} ticks "
      f"({NS_PER_TICK:.4f} ns/tick @ {TPROC_CLK_MHZ} MHz tProc 时钟)")
print(f"Init/Read 激光脉宽假设值 = {T_INIT_NS:.0f} ns -> {config.init_treg} ticks, "
      f"暗场间隔假设值 = {T_DARK_NS:.0f} ns -> {config.dark_pre_treg} ticks")

prog = PulsedODMR(config)

print("\n=== tProcessor 汇编代码编译成功 (PulsedODMR: Init-MW-Read) ===")
print(prog.asm())


# ==============================================================================
# 4. 离线波形仿真可视化
# ==============================================================================

# 方法 A：使用 QICK 原生 plot_pulses 绘制编译后的真实 DAC/门控波形
# ------------------------------------------------------------------
# 因为 PulsedODMR.body() 本身就严格按 Init-MW-Read 三段分离编译，这里直接用
# 编译结果画图即可，不再需要像之前那样另外手工"纠正"时序。
try:
    fig_dac = prog.plot_pulses()
    plt.suptitle("QICK DAC Pulse Synthesis Simulation (PulsedODMR: Init-MW-Read)")
    plt.show()
except Exception as e:
    print(f"DAC波形细节绘制跳过 (依赖具体 dummy 端口映射): {e}")


# 方法 B：按 PulsedODMR 实际配置的寄存器时长重建门控/载波示意图 (单图放大显示)
# ------------------------------------------------------------------
# 与之前几版不同，这里的每一段长度都直接取自 PulsedODMR 实际编译用的
# config.init_treg / dark_pre_treg / mw_treg / dark_post_treg / read_treg /
# relax_delay_treg，与 body() 里的时序完全一致，不再是手工猜测的数组。
# 只画一个完整循环，且不再单独放大 MW 窗口子图——整张图直接放大显示。

t_init = config.init_treg
t_dark_pre = config.dark_pre_treg
t_mw = config.mw_treg
t_dark_post = config.dark_post_treg
t_read = config.read_treg
t_gap = config.relax_delay_treg

idx_init_end = t_init
idx_mw_start = idx_init_end + t_dark_pre
idx_mw_end = idx_mw_start + t_mw
idx_read_start = idx_mw_end + t_dark_post
idx_read_end = idx_read_start + t_read
idx_cycle_end = idx_read_end + t_gap   # 一个完整 rep，含 relax_delay 间隙

OVERSAMPLE = 20  # 每个寄存器 tick 内插这么多点，让微波振荡在图上清晰可见

time_axis = np.linspace(0, idx_cycle_end, int(idx_cycle_end * OVERSAMPLE), endpoint=False)

sim_laser = np.zeros_like(time_axis)
sim_laser[(time_axis >= 0) & (time_axis < idx_init_end)] = 1.0               # Laser_ON (Init)
sim_laser[(time_axis >= idx_read_start) & (time_axis < idx_read_end)] = 1.0  # Laser_ON (Read)
# idx_init_end..idx_read_start 之间（含 MW 窗口及两段暗场间隔）激光保持 OFF

# 微波载波：频率按 mw_fMHz 相对于该生成器 DDS 时钟 f_dds 做归一化，
# mw_fMHz 已被限制在 dummy DAC 的可合成范围内 (<= f_dds/2，配合 nqz=2)。
f_dds_mw = soccfg['gens'][config.mw_channel]['f_dds']   # MHz
freq_norm = config.mw_fMHz / f_dds_mw                   # 每个时钟周期对应的归一化相位增量

mw_mask = (time_axis >= idx_mw_start) & (time_axis < idx_mw_end)
t_mw_local = time_axis[mw_mask] - idx_mw_start
sim_mw_i = np.zeros_like(time_axis)
sim_mw_q = np.zeros_like(time_axis)
sim_mw_i[mw_mask] = np.cos(2 * np.pi * freq_norm * t_mw_local)
sim_mw_q[mw_mask] = np.sin(2 * np.pi * freq_norm * t_mw_local)

print(f"\nmw_fMHz = {config.mw_fMHz} MHz, f_dds (mw channel) = {f_dds_mw} MHz, "
      f"归一化振荡频率 freq_norm = mw_fMHz / f_dds = {freq_norm:.6f} cycles/clk")

# 单张放大图：完整的一个 Init-MW-Read-gap 循环
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(time_axis, sim_laser, label='Laser (Gate/TTL)', color='green', linewidth=2.2)
ax.plot(time_axis, sim_mw_i, label='Microwave I (Ch 1)', color='tab:blue', linewidth=1.0)
ax.plot(time_axis, sim_mw_q, label='Microwave Q (Ch 1)', color='tab:orange', linewidth=1.0)

ax.axvspan(idx_mw_start, idx_mw_end, color='tab:orange', alpha=0.15, label='MW window (dark, tπ)')
for x in (idx_init_end, idx_mw_start, idx_mw_end, idx_read_start, idx_read_end, idx_cycle_end):
    ax.axvline(x, color='gray', linestyle=':', linewidth=1)

ax.text(t_init / 2, 1.08, 'Init\n(Laser ON)', ha='center', fontsize=11, color='green')
ax.text((idx_mw_start + idx_mw_end) / 2, 1.08, 'MW\n(dark)', ha='center', fontsize=11, color='darkorange')
ax.text((idx_read_start + idx_read_end) / 2, 1.08, 'Read\n(Laser ON)', ha='center', fontsize=11, color='green')
ax.text((idx_read_end + idx_cycle_end) / 2, 1.08, 'relax_delay\n(gap)', ha='center', fontsize=10, color='gray')

ax.set_ylim(-1.3, 1.5)
ax.set_xlabel('FPGA Clock Cycles / Registers', fontsize=12)
ax.set_ylabel('Simulated Signal Amplitude', fontsize=12)
ax.set_title(
    f'Simulated PulsedODMR Sequence (ZnO:In, Init-MW-Read, one full cycle)\n'
    f'tπ≈{T_PI_NS:.1f}ns, Init/Read≈{T_INIT_NS:.0f}/{T_READ_NS:.0f}ns (assumed), '
    f'fMW={config.mw_fMHz:.0f}MHz',
    fontsize=12)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()


# ==============================================================================
# 5. 论文 Fig.4 风格的 PODMR 频谱"预期曲线"仿真 (解析模型，非硬件采集数据)
# ==============================================================================
# 本脚本是离线/无硬件仿真，没有真实光子计数，因此这里不是重放实验数据，而是
# 按论文正文 + App. D 给出的解析关系，构造一条"如果真的做这个频率扫描 PODMR，
# 预期会测到什么样的谱线"的示意曲线，用来检验上面配置的扫描参数(中心频率、
# 超精细间距、线宽)是否自洽，并大致重现 Fig. 4(a) 的形状：
#
#   - 10 条等间距(104MHz)的超精细谱线，线宽取 App.D 卷积线宽 Γ_conv≈46.7MHz
#   - 核自旋极化 P_N=-0.19 (正文) 用一个简单的玻尔兹曼权重近似，产生
#     "+9/2 高、-9/2 低" 的非对称包络 (对应 Fig.4(a) 的 σ- 泵浦谱)
#   - 中心谱线压低的"碗状"包络是论文里尚未解释的现象(正文 Sec.III.B)，
#     这里只用一个高斯凹陷做定性示意，不代表任何物理推导
#
# 这部分完全是解析可视化，与上面的 tProc 汇编/寄存器时序无关。

mI_values = np.arange(4.5, -5.5, -1.0)     # +9/2 ... -9/2，与 Fig.4 x 轴方向一致
line_freqs_MHz = MW_CENTER_FMHZ + mI_values * ( -HYPERFINE_SPLITTING_MHZ)
freq_axis_MHz = np.arange(MW_START_FMHZ, MW_END_FMHZ, 0.5)

# 由核自旋极化 P_N=<Iz>/I 反解一个等效玻尔兹曼权重参数 beta:
#   <mI> = I * P_N = sum(mI * exp(-beta*mI)) / sum(exp(-beta*mI))，数值求解 beta
I_spin = 4.5
target_mean_mI = I_spin * NUCLEAR_POLARIZATION


def mean_mI(beta, mI_vals=mI_values):
    w = np.exp(-beta * mI_vals)
    return np.sum(mI_vals * w) / np.sum(w)


beta_lo, beta_hi = -1.0, 1.0
for _ in range(60):
    beta_mid = 0.5 * (beta_lo + beta_hi)
    if mean_mI(beta_mid) > target_mean_mI:
        beta_lo = beta_mid
    else:
        beta_hi = beta_mid
beta_fit = 0.5 * (beta_lo + beta_hi)

population_weights = np.exp(-beta_fit * mI_values)
population_weights /= population_weights.max()

# "碗状"包络：中心跃迁(mI接近0)幅度被压低，定性示意 Sec.III.B 描述的现象
bowl_envelope = 1.0 - 0.45 * np.exp(-(mI_values**2) / (2 * 2.5**2))

line_amplitudes = population_weights * bowl_envelope

sigma_MHz = LINEWIDTH_CONV_MHZ / (2 * np.sqrt(2 * np.log(2)))  # FWHM -> Gaussian sigma
spectrum = np.zeros_like(freq_axis_MHz)
for f0, amp in zip(line_freqs_MHz, line_amplitudes):
    spectrum += amp * np.exp(-0.5 * ((freq_axis_MHz - f0) / sigma_MHz) ** 2)

plt.figure(figsize=(9, 5))
plt.plot(freq_axis_MHz, spectrum, color='darkorange', linewidth=1.8)
for f0, mI in zip(line_freqs_MHz, mI_values):
    plt.axvline(f0, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
plt.text(line_freqs_MHz[0], 1.05 * line_amplitudes.max(), "+9/2", ha='center')
plt.text(line_freqs_MHz[-1], 1.05 * line_amplitudes.max(), "-9/2", ha='center')
plt.xlabel('Microwave Frequency (MHz)')
plt.ylabel('Simulated PL Contrast (a.u.)')
plt.title('Simulated PODMR Spectrum (115In in ZnO, 10 hyperfine lines, '
          f'Γ≈{LINEWIDTH_CONV_MHZ:.1f} MHz, P_N={NUCLEAR_POLARIZATION})')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print(f"\n拟合得到的玻尔兹曼参数 beta={beta_fit:.4f}，使 <mI>/I = "
      f"{mean_mI(beta_fit)/I_spin:.3f} (目标 P_N={NUCLEAR_POLARIZATION})")
