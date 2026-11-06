"""
PulsedODMR 本機執行版本 —— 直接在 RFSoC4x2 板卡上運行
=====================================================
修改重點：
  1. 移除 Pyro4 連線，直接使用 QickSoc() 本機初始化。
  2. 微波降頻至 30 MHz（第 1 奈奎斯特區 nqz=1），適配 200 MHz 示波器觀測。
  3. MW 脈衝寬度暫時調整為 200 ns，以便在示波器上清楚看到射頻週期。
"""

import sys
import time
from copy import copy
import numpy as np
import matplotlib.pyplot as plt

# 本機模式直接匯入 QickSoc
from qick import QickSoc
from qick.averager_program import QickSweep
import qickdawg as qd
from qickdawg.nvpulsing.nvaverageprogram import NVAveragerProgram

# ==============================================================================
# 0. 本機初始化 RFSoC
# ==============================================================================
print("[系統] 正在本機初始化 QickSoc (載入板載 bitstream)...")
soc = QickSoc()
soccfg = soc

# 注入給 qickdawg
qd.soc = soc
qd.soccfg = soccfg


# ==============================================================================
# 1. PulsedODMR 脈衝序列定義 (Init-MW-Read)
# ==============================================================================
class PulsedODMR(NVAveragerProgram):
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

        # ADC 積分窗口沿用 Read 段的激光脈衝寬度
        self.cfg.readout_integration_treg = self.cfg.read_treg
        self.setup_readout()

        self.declare_gen(ch=self.cfg.mw_channel, nqz=self.cfg.mw_nqz)

        self.set_pulse_registers(
            ch=self.cfg.mw_channel,
            style='const',
            freq=self.cfg.mw_start_freg,
            gain=self.cfg.mw_gain,
            length=self.cfg.mw_treg,
            phase=0)

        self.mw_frequency_register = self.get_gen_reg(self.cfg.mw_channel, "freq")

        self.add_sweep(QickSweep(self,
                                  self.mw_frequency_register,
                                  self.cfg.mw_start_fMHz,
                                  self.cfg.mw_end_fMHz,
                                  self.cfg.nsweep_points))

        self.synci(100)
        if (self.cfg.ddr4 is True) or (self.cfg.mr is True):
            self.trigger(ddr4=self.cfg.ddr4, mr=self.cfg.mr, adc_trig_offset=0)
        self.synci(100)

    def body(self):
        # 1. Init: 激光 ON (PMOD 輸出)
        self.trigger(
            pins=[self.cfg.laser_gate_pmod],
            width=self.cfg.init_treg,
            adc_trig_offset=0,
            t=0)

        # 2. MW: 暗場中觸發微波 (DAC 輸出)
        t_mw = self.cfg.init_treg + self.cfg.dark_pre_treg
        self.pulse(ch=self.cfg.mw_channel, t=t_mw)

        # 3. Read: 激光 ON，同時觸發 ADC
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
        return super().acquire(readouts_per_experiment=1, *arg, **kwarg)


# ==============================================================================
# 2. 參數配置 (適配 200 MHz 示波器量測)
# ==============================================================================
MW_CHANNEL = 1          # 1 對應 DAC_B
ADC_CHANNEL = 0         # 板載 ADC 0
LASER_PMOD = 0          # PMODA Pin 1
MW_NQZ = 1              # 30 MHz 使用第 1 奈奎斯特區 (原為 2)

# --- 示波器量測調整 ---
MW_FMHZ_TEST = 30.0     # 降至 30 MHz，確保 SDS824X HD 能清晰觀測
T_MW_NS = 200.0         # 將 MW 脈寬暫時拉長到 200 ns (約包含 6 個正弦波週期)

T_INIT_NS = 200.0       # Init 激光脈衝長度
T_DARK_NS = 50.0        # 暗場間隔拉長至 50 ns，與 MW 脈寬間隔清楚區隔
T_READ_NS = 200.0       # Read 激光脈衝長度

TPROC_CLK_MHZ = soccfg['tprocs'][0]['f_time']
NS_PER_TICK = 1000.0 / TPROC_CLK_MHZ

def ns_to_treg(t_ns):
    return max(1, round(t_ns / NS_PER_TICK))

default_config = qd.NVConfiguration()
default_config.test = False
default_config.mw_readout_channel = 1
default_config.ddr4 = False
default_config.adc_channel = ADC_CHANNEL
default_config.mw_channel = MW_CHANNEL
default_config.mw_nqz = MW_NQZ
default_config.mw_gain = 30000          # 提高增益至接近滿量程，便於示波器觀察
default_config.reps = 100000            # 設定較大重複次數，確保示波器穩定捕獲
default_config.laser_gate_pmod = LASER_PMOD

default_config.init_treg = ns_to_treg(T_INIT_NS)
default_config.dark_pre_treg = ns_to_treg(T_DARK_NS)
default_config.mw_treg = ns_to_treg(T_MW_NS)
default_config.dark_post_treg = ns_to_treg(T_DARK_NS)
default_config.read_treg = ns_to_treg(T_READ_NS)
default_config.relax_delay_treg = ns_to_treg(1000.0)

config = copy(default_config)
config.mw_fMHz = MW_FMHZ_TEST

# 示波器量測單音波形，將頻率範圍固定在 30 MHz
config.add_linear_sweep('mw', 'fMHz', start=MW_FMHZ_TEST, stop=MW_FMHZ_TEST, delta=1.0)

print(f"[配置] 微波頻率: {MW_FMHZ_TEST} MHz (DAC_B, 通道 {MW_CHANNEL})")
print(f"[配置] PMOD 門控通道: {LASER_PMOD} (PMODA Pin 1)")


# ==============================================================================
# 3. 編譯並在板卡本機播放
# ==============================================================================
prog = PulsedODMR(config)

print("\n=== tProcessor 組合語言編譯成功 ===")
print(prog.asm())

print(f"\n[運行] 正在板卡本機循環播放波形 ({config.reps} 次循環)...")
print("示波器設定：")
print("  - CH1 (PMODA Pin 1): 1 MΩ, 1.0 V/div, 設為觸發源 (上升沿 1.5V)")
print("  - CH2 (DAC_B SMA)  : 100 mV/div, 水平時基 100 ns/div")

try:
    prog.acquire(soc, progress=True)
    print("\n[完成] 波形播放結束。")
except KeyboardInterrupt:
    print("\n[中斷] 手動停止。")
finally:
    soc.reset()
    print("[復位] 板卡輸出已復位關閉。")