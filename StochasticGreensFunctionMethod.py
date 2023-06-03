import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from config import Config
from get_source_param import SourceParamReader
from basenspectrum import BasenSpectrum
from calculus_in_freq import Calculus

class Stochastic_Greens_Function_method():
    """
    Stochastic Greens Function method
    統計的グリーン関数法
    """
    def __init__(self,id):
        self.id = id
        self.conf = Config()
        self.calc = Calculus()
        self.source_param = SourceParamReader()
        
        self.elementFaultsPath = f"{self.conf.input_path}/SourceParameters_{self.conf.EQid}_ID{int(self.id):03d}.xlsx"

        self.base_waveforms_path = f"{self.conf.wave_path}/BaseWaves/BaseWaveforms_{self.conf.EQid}_ID{int(self.id):03d}.csv"
        self.base_wave_fft_path  = f"{self.conf.wave_path}/BaseWaves/BaseFFT_{self.conf.EQid}_ID{int(self.id):03d}.csv"
        
        self.timeseries_path     = f"{self.conf.element_path}/Apq_Waveforms_{self.conf.EQid}_ID{int(self.id):03d}.csv"
        self.fftabs_path         = f"{self.conf.element_path}/Apq_fftabs_{self.conf.EQid}_ID{int(self.id):03d}.csv"

        self.waveforms_fig_path  = f"{self.conf.fig_path}/BaseWaves/waveforms_{self.conf.EQid}_ID{int(self.id):03d}.png"
        self.fftabs_fig_path     = f"{self.conf.fig_path}/BaseWaves/fftabs_{self.conf.EQid}_ID{int(self.id):03d}.png"
        
    def set_source_info(self):
        """
        set source parameters
        """
        self.source_param.read_xlsx(self.conf.source_parameters_path)
        self.source_param.calc_element_coordinates_Information()
        self.source_param.calc_element_epi_distance(self.conf.lon_station,self.conf.lat_station,self.conf.depth_station)
        print(f"Hypocentral distance :{self.source_param.epi_distance:.2f}km")

    def save_source_info(self):
        """
        save source parameters
        """
        self.source_param.saveElementModel(self.elementFaultsPath)
        print(f"save source parameters to {self.elementFaultsPath}")

    def read_element_info(self):
        """
        set element parameters
        """
        self.element_param = pd.read_csv(self.elementFaultsPath)
        print(f"read element parameters from {self.elementFaultsPath}")
        print(self.element_param)

    def set_base_spectrum(self):
        """
        set base spectrum
        """
        self.basenSpectrum = BasenSpectrum( 
                                        N       = self.conf.N,
                                        dt      = self.conf.dt,
                                        M0      = self.source_param.M0,
                                        rho     = self.source_param.rho,
                                        Vs      = self.source_param.Vs,
                                        fmax    = self.source_param.fmax,
                                        rho_sb  = self.source_param.rho_sb,
                                        Vs_sb   = self.source_param.Vs_sb,
                                        N_elements  =self.source_param.N_elements,
                                        faultArea = self.source_param.faultArea
                                    )
        # set seed for white noise
        self.basenSpectrum.fix_seed(self.conf.SEED)


    def calc_element_spectrum(self):
        """
        calculate element spectrum
        """
        self.Ue_pq = pd.DataFrame() 
        self.CorrectionFactor_pq = pd.DataFrame() 
        self.time_delay = pd.DataFrame() 
        self.time_delay_factor = pd.DataFrame() 
        self.Ue_pq_correction = pd.DataFrame() 
        self.Ue_pq_correction_time_delay = pd.DataFrame() 

        for index, elementFault in self.element_param.iterrows():

            print("Calc. ElementSpectrum for each elemental fault") 
            print(f"Element (No. = {elementFault['No.']} )")
            
            print(f"    M0              : {elementFault['M0']:.3g} MPa")
            print(f"    stressDrop      : {elementFault['stressDrop']:.2f} MPa")
            print(f"    slip            : {elementFault['slip']:.2f} cm")
            print(f"    epiDistance     : {elementFault['epiDistance']:.2f} km")
            print(f"    slipVelocity    : {elementFault['slipVelocity']:.2f} cm/s")
            print(f"    area            : {elementFault['area']:.2f} km2")
            
            _upq = self.basenSpectrum.calc_Spectrum(
                                            sigma_pq    = elementFault["stressDrop"],
                                            D_pq        = elementFault["slip"],
                                            M0_pq       = elementFault["M0"],
                                            r_pq        = elementFault["epiDistance"],
                                            V_pq        = elementFault["slipVelocity"],
                                            area_pq     = elementFault["area"],
                                            )

            time_delay = elementFault["ruptureTime"] + elementFault["randomDelay"]
            
            _time_delay_factor = np.exp(-1j * 2.0 * np.pi * self.basenSpectrum.freq * time_delay)
            
            self.time_delay_factor[index]       = _time_delay_factor
            
            self.CorrectionFactor_pq[index]    = _upq[0]
            self.Ue_pq[index]                  = _upq[1]
            self.time_delay[index]             = [time_delay]
            self.Ue_pq_correction[index]       = _upq[1] * _upq[0]
            self.Ue_pq_correction_time_delay[index] = _upq[1] * _upq[0] * _time_delay_factor

    def calc_acceleration_waveforms(self):
        """
        calculate acceleration waveforms
        """
        self.A_pq_timeseries= pd.DataFrame() 
        self.A_pq_fftabs= pd.DataFrame()     
        self.basenSpectrum.calc_white_Noise()
        
        for index, elementFault in self.element_param.iterrows():

            print("Calc. Acceleration waveforms for each elemental fault")
            print(f"Element (No. = {elementFault['No.']} )")
            
            us = self.Ue_pq_correction[index]
            us = np.abs(us.values)
            
            rij =  elementFault["epiDistance"]
            M0ij = elementFault["M0"]
            t_pq = elementFault["ruptureTime"] + elementFault["randomDelay"]
            D_pq = elementFault["slip"]
            
            if M0ij != 0:
                envelope = self.basenSpectrum.calc_Envelope(X = rij, M0 =M0ij)   
                noise_Envelope = self.basenSpectrum.calc_Envelope_white_Noise(envelope)
                noise_phase = self.basenSpectrum.calc_noise_Spectrum(noise_Envelope)
                
                acc_random = self.basenSpectrum.phase_assignment(
                    spectrum = us,
                    phase = noise_phase
                    )
                acc_random_Envelope = acc_random * envelope
                acc_random_spectrum = self.basenSpectrum.calc_AccFFT(acc = acc_random_Envelope)
                
                phase_roop = np.angle(acc_random_spectrum)
                for roop in range(self.conf.phaseRoop):
                    acc_roop = self.basenSpectrum.phase_assignment(
                                    spectrum = us,
                                    phase    = phase_roop
                                    )
                    accEnvelope_roop = acc_roop * np.power(envelope,1/self.conf.phaseRoop)
                    accSpectrum_roop = self.basenSpectrum.calc_AccFFT(acc = accEnvelope_roop)            
                    phase_roop = np.angle(accSpectrum_roop)   
                
                acc_ij = accEnvelope_roop
                t_pq_arg = np.argmax(self.basenSpectrum.time>t_pq+self.source_param.epi_distance/self.source_param.Vs) 
                acc_ij = np.roll(acc_ij, t_pq_arg)
                
                self.A_pq_timeseries[index] = acc_ij
                self.A_pq_fftabs[index]     = np.abs(accSpectrum_roop)
                
                print(f"    →→→ maxAcc.: {np.abs(acc_ij).max():.2f}gal")

            print()

    def calc_wave_summation(self):
        """
        calculate acceleration waveforms
        """
        self.AccWave = self.A_pq_timeseries.sum(axis=1)
        self.AccWave = self.AccWave.values
        print(f"BasePlane_SyntheticWave maxAcc. : {np.abs(self.AccWave).max():.2f}gal")
        
    def save_data(self):
        """
        save data
        """
        print("Save Data")
        
        t,AccWave_filterd = self.calc.calcDerivative(
                            time=self.basenSpectrum.time,
                            wave=self.AccWave,
                            derivative_times=0,
                            isTaper=self.conf.isTaper,
                            left=self.conf.left,start=self.conf.start,end=self.conf.end,right=self.conf.right
                        )
                        
        t,VelWave = self.calc.calcDerivative(
                            time=self.basenSpectrum.time,
                            wave=self.AccWave,
                            derivative_times=-1,
                            isTaper=self.conf.isTaper,
                            left=self.conf.left,start=self.conf.start,end=self.conf.end,right=self.conf.right
                        )

        freq,self.FFTabs,phase = self.calc.calcFFT(
                            time=self.basenSpectrum.time,
                            wave=self.AccWave,
                            derivative_times=0,
                            isTaper=self.conf.isTaper,
                            left=self.conf.left,start=self.conf.start,end=self.conf.end,right=self.conf.right,
                            isParzen=True,parzen_setting=0.1
                            )
        
        BasePlane_SyntheticWaveforms= pd.DataFrame()
        BasePlane_SyntheticWaveforms["Time(sec)"] = self.basenSpectrum.time
        BasePlane_SyntheticWaveforms["Acc(cm/s/s)"] = self.AccWave
        BasePlane_SyntheticWaveforms["Acc_filterd(cm/s/s)"] = AccWave_filterd
        BasePlane_SyntheticWaveforms["Vel(cm/s)"] = VelWave
        
        BasePlane_SyntheticWaveFFT= pd.DataFrame()
        BasePlane_SyntheticWaveFFT["Freq(Hz)"] = freq
        BasePlane_SyntheticWaveFFT["FourierSpectrum(cm/s)"] = self.FFTabs
        BasePlane_SyntheticWaveFFT["phase(rad)"] = phase
        
        BasePlane_SyntheticWaveforms.to_csv(self.base_waveforms_path, index=False)
        BasePlane_SyntheticWaveFFT.to_csv(self.base_wave_fft_path, index=False)


    def save_elementsdata(self):
        """
        save data
        """
        print("Save Data")
        
        self.A_pq_timeseries["Time"] = self.basenSpectrum.time
        cols = self.A_pq_timeseries.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        self.A_pq_timeseries = self.A_pq_timeseries[cols]

        self.A_pq_fftabs["Freq"]     = self.basenSpectrum.freq
        cols = self.A_pq_fftabs.columns.tolist()
        cols = [cols[-1]] + cols[:-1]
        self.A_pq_fftabs = self.A_pq_fftabs[cols]

        self.A_pq_timeseries.to_csv(self.timeseries_path, index=False)
        self.A_pq_fftabs.to_csv(self.fftabs_path, index=False)

    def save_figure(self):
        """
        save figure
        """
        print("Save Figure")

        # Use style sheets for improved aesthetics
        plt.style.use('bmh')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.basenSpectrum.time,self.AccWave,label="BasePlane_SyntheticWave", color='blue', linewidth=0.1)
        ax.set_title('Time Domain Waveform', fontsize=16)
        ax.set_xlabel("Time (sec)", fontsize=14)
        ax.set_ylabel("Acceleration (cm/s/s)", fontsize=14)
        ax.legend(fontsize=12)
        ax.set_ylim(-150, 150)
        fig.savefig(self.waveforms_fig_path, dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.basenSpectrum.freq, self.FFTabs, label="BasePlane_SyntheticWave", color='red', linewidth=2)
        ax.set_title('Frequency Spectrum', fontsize=16)
        ax.set_xlabel("Frequency (Hz)", fontsize=14)
        ax.set_ylabel("Fourier Spectrum (cm/s)", fontsize=14)
        ax.legend(fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.01, 100)
        fig.savefig(self.fftabs_fig_path, dpi=300)
        plt.close()


    def rogic(self):
        """
        統計的グリーン関数法の実行
        """
        print("Start Rogic")

        self.set_source_info()# excelファイルから情報を読み込む
        self.save_source_info()# 読み込んだ情報を保存する
        self.read_element_info()

        self.set_base_spectrum()

        self.calc_element_spectrum()
        self.calc_acceleration_waveforms()
        self.calc_wave_summation()
        
        self.save_data()
        # self.save_elementsdata()# 要素断層の時刻歴と周波数スペクトルを保存する

        self.save_figure()

        print("Finish Rogic")

def main():
    conf = Config()
    for i in range(conf.Number_of_Generations):
        print(f"step:{i}")
        stochastic_greens_function_method = Stochastic_Greens_Function_method(id=i)
        stochastic_greens_function_method.rogic()
        
if __name__ == '__main__':
    main()           
