from cmath import log
import sys, os, json
import math
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, Future, wait
from numba import jit

from config import Config

@jit(nopython=True)
def calc(g_acc,a11,a12,b11,b12,
            a21,a22,b21,b22,
            h,w):

    b_acc=np.zeros_like(g_acc)
    b_vel=np.zeros_like(g_acc)
    b_dis=np.zeros_like(g_acc)
    b_acc_abs=np.zeros_like(g_acc)
    b_acc[0] = -g_acc[0]
    length=len(g_acc)
    for i  in range(1,length):
        b_dis[i] = a11 * b_dis[i - 1] + a12 * b_vel[i - 1] + b11 * g_acc[i - 1] + b12 * g_acc[i]
        b_vel[i] = a21 * b_dis[i - 1] + a22 * b_vel[i - 1] + b21 * g_acc[i - 1] + b22 * g_acc[i]
        b_acc[i] = - g_acc[i] - 2.0 * h * w * b_vel[i] - w * w * b_dis[i]
        b_acc_abs[i] = b_acc[i] + g_acc[i]
    return b_dis, b_vel, b_acc_abs


#計算クラス
class Calculus:
    def __init__(self):
        None
        
    # Nigam・Jennings法による応答計算
    def nigam_jennings(self,period, g_acc, dt, h):
        """
        Calculate response using Nigam-Jennings method.

        Parameters
        ----------
        period : float
            The period for which to calculate the response.
        g_acc : np.ndarray
            The ground acceleration.
        dt : float
            The time step.
        h : float
            The damping factor.

        Returns
        -------
        tuple
            The period, maximum acceleration, maximum velocity, and maximum displacement.
        """
        if period == 0:
            _period, acc_max, vel_max, dis_max= 0.0, np.max(np.abs(g_acc)), 0, 0 
            return _period, acc_max, vel_max, dis_max
        
        # print("period",period,"dt", dt,"h", h)
        w = 2.0 * np.pi / period
        h_ = np.sqrt(1.0 - h * h)
        w_ = h_ * w
        ehw = np.exp(-h * w * dt);
        hh_ = h / h_
        sinw_ = np.sin(w_*dt)
        cosw_ = np.cos(w_ * dt)
        hw1 = (2.0 * h * h - 1.0) / (w * w * dt)
        hw2 = h / w
        hw3 = (2.0 * h) / (w * w * w * dt)
        ww = 1.0 / (w * w)
        a11 = ehw * (hh_ * sinw_ + cosw_)
        a12 = ehw / w_ * sinw_
        a21 = -w / h_ * ehw * sinw_
        a22 = ehw * (cosw_ - hh_ * sinw_)
        b11 = ehw * ((hw1 + hw2) * sinw_ / w_ + (hw3 + ww) * cosw_) - hw3
        b12 = -ehw * (hw1 * sinw_ / w_ + hw3 * cosw_) - ww + hw3
        b21 = ehw * ((hw1 + hw2) * (cosw_ - hh_ * sinw_) - (hw3 + ww) * (w_ * sinw_ + h * w * cosw_)) + ww / dt
        b22 = -ehw * (hw1 * (cosw_ - hh_ * sinw_) - hw3 * (w_ * sinw_ + h * w * cosw_)) - ww / dt
        
        b_dis, b_vel, b_acc_abs = calc(g_acc,a11,a12,b11,b12,
                                             a21,a22,b21,b22,
                                            h,w)
        dis_max, vel_max, acc_max = np.max(np.abs(b_dis)),np.max(np.abs(b_vel)),np.max(np.abs(b_acc_abs))
        
        return period, acc_max, vel_max, dis_max
    
    
    # 応答スペクトル計算
    def spectrum(self,g_acc,dt,periods,h):
        """
        Calculate the response spectrum.

        Parameters
        ----------
        g_acc : np.ndarray
            The ground acceleration.
        dt : float
            The time step.
        periods : list
            The  periods for which to calculate the spectrum.
        h : float
            The damping factor.

        Returns
        -------
        pd.DataFrame
            The response spectrum.
        """
        
        df = pd.DataFrame()

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            print("nigam法の計算をします",'並列計算:CPU', os.cpu_count())
            futures = [executor.submit(Calculus.nigam_jennings, self, period, g_acc, dt, h) for period in periods]
            results = [future.result() for future in futures]
            df['period'] = [result[0] for result in results]
            df['acc']    = [result[1] for result in results]
            df['vel']    = [result[2] for result in results]
            df['dis']    = [result[3] for result in results]

        return df


class ResponseSpectrumCalculator:
    def __init__(self, conf,calculus, damping):
        self.conf = conf
        self.calculus = calculus
        self.damping = damping

        self.id = 0

        self.target_id = 0
        self.plus_sigma_min_squared_difference_id = 0
        self.minus_sigma_min_squared_difference_id = 0

        self.response_spectrum = pd.DataFrame()
        self.dt = self.conf.dt

        # input
        self.base_waveforms_path = f"{self.conf.wave_path}/BaseWaves/BaseWaveforms_{self.conf.EQid}_ID{self.id:03d}.csv"
        self.base_wave_fft_path = f"{self.conf.wave_path}/BaseWaves/BaseFFT_{self.conf.EQid}_ID{self.target_id:03d}.csv"
        
        # output
        self.target_id_path = f"{self.conf.wave_path}/targetID_{self.conf.EQid}.txt"
        self.response_spectrum_path = f"{self.conf.wave_path}/ResponseSpectrums_{self.conf.EQid}.csv"
        # average , +sigma, -sigma
        self.average_spectrum_path = f"{self.conf.wave_path}/BaseAverageFourierSpectrum_{self.conf.EQid}.csv"
        self.plus_sigma_spectrum_path = f"{self.conf.wave_path}/BasePlusSigmaFourierSpectrum_{self.conf.EQid}.csv"
        self.minus_sigma_spectrum_path = f"{self.conf.wave_path}/BaseMinusSigmaFourierSpectrum_{self.conf.EQid}.csv"

        self.Rep_fig_path = f"{self.conf.fig_path}/ResponseSpectrum_of_BaseWaves_{self.conf.EQid}.png"
        

    @property
    def id(self):
        return self._id
    @id.setter
    def id(self, value):
        self._id = value
        # IDが変更されると、base_waveforms_pathも自動的に更新されます
        self.base_waveforms_path = f"{self.conf.wave_path}/BaseWaves/BaseWaveforms_{self.conf.EQid}_ID{self._id:03d}.csv"
    @property
    def target_id(self):
        return self._target_id
    @target_id.setter
    def target_id(self, value):
        self._target_id = value
        # target_idが変更されると、base_wave_fft_pathも自動的に更新されます
        self.base_wave_fft_path = f"{self.conf.wave_path}/BaseWaves/BaseFFT_{self.conf.EQid}_ID{self._target_id:03d}.csv"

    @property
    def base_waveforms_path(self):
        return self._base_waveforms_path
    @base_waveforms_path.setter
    def base_waveforms_path(self, path):
        self._base_waveforms_path = path
    @property
    def base_wave_fft_path(self):
        return self._base_wave_fft_path
    @base_wave_fft_path.setter
    def base_wave_fft_path(self, path):
        self._base_wave_fft_path = path

    def load_waveforms(self):
        """Load base waveforms from a CSV file."""

        base_waveforms = pd.read_csv(self.base_waveforms_path)

        return base_waveforms

    def calculate_response_spectrum(self):
        """Calculate the response spectrum."""
        
        for id in range(self.conf.Number_of_Generations):
            self.id = id
            base_waveforms = self.load_waveforms()
            g_acc = base_waveforms["Acc_filterd(cm/s/s)"].values
            dt = base_waveforms["Time(sec)"][1] - base_waveforms["Time(sec)"][0]

            N = len(g_acc) // 2
            max_freq = 1 / (dt * 2)
            df = max_freq / N
            freq = np.arange(start=df, stop=max_freq, step=df)
            self.periods = 1 / freq

            max_val = np.max(np.abs(g_acc))
            index = len(g_acc) - next(i for i, val in enumerate(reversed(g_acc), 1) if val >= max_val/10)
            g_acc = g_acc[:index+1]

            result = self.calculus.spectrum(g_acc=g_acc, dt=dt, periods= self.periods, h=self.damping)
            omega = 2 * np.pi / result["period"]
            pSv = result["acc"] / omega
            new_column = pd.DataFrame(pSv, columns=[f"{id:03d}"])
            if id == 0:
                self.response_spectrum = new_column
            else:
                self.response_spectrum = pd.concat([self.response_spectrum, new_column], axis=1)

            self.print_response_spectrum_info(id, N, g_acc, pSv, result)
        # To avoid fragmentation
        self.response_spectrum = self.response_spectrum.copy()

    def print_response_spectrum_info(self, id, N, g_acc, pSv, result):
        """Print information about the response spectrum."""
        print(f"EQid:{self.conf.EQid} ID:{id}")
        print(f"    input N     :{N}")
        print(f"    calc  N     :{len(g_acc)}")
        print(f"    MaxAcc      :{np.max(np.abs(g_acc)):.2f}cm/s/s")
        print(f"    MaxpSv      :{np.max(pSv):.2f}cm/s")
        print(f"    MaxpSvPeriod:{result['period'][np.argmax(pSv)]:.2f}sec")
        print()

    def save_response_spectrum(self):
        """Save the response spectrum to a CSV file."""

        # Create a new DataFrame with  self.periods as the first column
        spectrum_df = pd.DataFrame( self.periods, columns=['periods(sec)'])
        # Concatenate with the response_spectrum DataFrame
        spectrum_df = pd.concat([spectrum_df, self.response_spectrum], axis=1)
        # Save the DataFrame
        spectrum_df.to_csv(self.response_spectrum_path, index=False)


    def calculate_statistics(self):
        """Calculate the average and standard deviation of the response spectrum."""
        self.Sv_avg = self.response_spectrum.mean(axis=1)
        self.Sv_std = self.response_spectrum.std(axis=1)
        self.Sv_plus_sigma = self.Sv_avg + self.Sv_std
        self.Sv_minus_sigma = self.Sv_avg - self.Sv_std

    def find_target_id(self):
        """Find the ID with the smallest squared difference from the mean."""
        squared_differences = self.response_spectrum.sub(self.Sv_avg, axis=0).pow(2)
        self.target_id = int(squared_differences.sum().idxmin())    
        # +sigma, -sigma
        squared_differences = self.response_spectrum.sub(self.Sv_plus_sigma, axis=0).pow(2)
        self.plus_sigma_min_squared_difference_id = squared_differences.sum().idxmin()
        squared_differences = self.response_spectrum.sub(self.Sv_minus_sigma, axis=0).pow(2)
        self.minus_sigma_min_squared_difference_id = squared_differences.sum().idxmin()
        
        print('ID with smallest squared difference from mean:', self.target_id)
        # target_idをtextに保存
        with open(self.target_id_path, mode='w') as f:
            f.write(f"minimum squared difference ID: {self.target_id}")

    def load_fft_phase_spectrum(self):
        """Load the FFT phase spectrum from a CSV file."""
        self.target_id = int(self.target_id)# ここでidに代入したidに適合させる
        base_fft = pd.read_csv(self.base_wave_fft_path)
        self.fft_phase_spectrum = base_fft["phase(rad)"].values
        self.fft_spectrum = base_fft["FourierSpectrum(cm/s)"].values
        

    def create_average_spectrum(self):
        """Create the average spectrum."""
        freq = 1/self.periods # Frequency (Hz)
        self.average_spectrum = pd.DataFrame()
        self.average_spectrum["Freq(Hz)"] = freq
        self.average_spectrum["FourierSpectrum(cm/s)"] = self.Sv_avg[::-1]  # Average velocity response spectrum = acceleration Fourier amplitude spectrum
        self.average_spectrum["FourierSpectrum_plus_sigma(cm/s)"] = self.Sv_plus_sigma[::-1]
        self.average_spectrum["FourierSpectrum_minus_sigma(cm/s)"] = self.Sv_minus_sigma[::-1]
        new_row = pd.DataFrame([[0, 0, 0, 0]], columns= self.average_spectrum.columns)
        self.average_spectrum = pd.concat([new_row,  self.average_spectrum]).reset_index(drop=True)
        self.average_spectrum["phase(rad)"] = self.fft_phase_spectrum # FFT phase spectrum of the minimum squared difference ID
        self.freq = self.average_spectrum["Freq(Hz)"].values


    def save_average_spectrum(self):
        """Save the average spectrum to a CSV file."""
        self.average_spectrum.to_csv(self.average_spectrum_path, index=False)

    def plot_response_spectrum(self):
        """Plot the response spectrum."""
        plt.style.use('bmh')
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title(f"Response Spectrum EQID{self.conf.EQid} h={self.damping:.2f}", fontsize=18, fontweight='bold')
        ax.set_xlabel("Period (sec)", fontsize=14)
        ax.set_ylabel("Pseudo Velocity Response Spectrum (cm/s)", fontsize=14)
        ax.set_xlim(0.02, 10)
        ax.set_ylim(0.1, 200)
        ax.grid(True, linestyle='--', alpha=0.6)

        for id in range(self.conf.Number_of_Generations):
            color =  'black'
            linewidth =  0.1
            alpha =  0.8
            ax.plot( self.periods, self.response_spectrum[f"{id:03d}"], color=color, linewidth=linewidth, alpha=alpha)
            if id == self.target_id:
                pass

        ax.plot( self.periods, self.response_spectrum[f"{self.target_id:03d}"], label=f"ID{self.target_id:03d}", color='blue', linewidth=1, alpha=1)
        ax.plot( self.periods, self.Sv_avg, label="Average Sv", color='red', linewidth=1)
        ax.plot( self.periods, self.Sv_avg + self.Sv_std, label="+std Sv", color='red', linewidth=1)
        ax.plot( self.periods, self.Sv_avg - self.Sv_std, label="-std Sv", color='red', linewidth=1)
        ax.fill_between( self.periods, self.Sv_avg - self.Sv_std, self.Sv_avg + self.Sv_std, color='red', alpha=0.2)
        
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(self.Rep_fig_path, dpi=300)  # Save in high resolution
        plt.close()
    

    
    def phase_assignment(self, spectrum, phase):
        """
        加速度フーリエスペクトル振幅に乱数位相を割り当てて、加速度波形を返す

        Parameters:
        spectrum (np.ndarray): 加速度フーリエスペクトル振幅
        phase (np.ndarray): 乱数位相

        Returns:
        np.ndarray: 加速度波形
        """
        # 位相割り当て　F(ω) = |F(ω)|exp(jθ(ω))
        
        nfft = len(spectrum)
        # ntime = nfft * 2
        FFT = spectrum * np.exp(1j*phase)
        FFT = np.append(FFT, np.zeros_like(FFT))
        
        wave    = np.fft.ifft(FFT.real)  / self.dt
        wave = wave.real * 4 # 折り返し振動数以下の実部のみ→4倍
        
        return wave

    def calc_AccFFT(self,acc):
        """
        加速度のFFTを計算する

        Parameters:
        acc (np.ndarray): 加速度

        Returns:
        np.ndarray: 加速度のFFT
        """

        nfft = len(acc)//2
        acc = acc - acc.mean()
        AccSpectrum = np.fft.fft(acc)[:nfft] * self.dt
        
        return AccSpectrum

    
    # 平均スペクトル適合波形の計算する関数を作成する
    def calculate_average_spectrum_fit_waveform(self):
        """Calculate the average spectrum fit waveform."""
        
        waveform = self.phase_assignment(self.average_spectrum["FourierSpectrum(cm/s)"],self.average_spectrum["phase(rad)"])
        time = np.arange(0, len(waveform)*self.dt, self.dt)


        # # 包絡波形 Envelopeの計算
        # # best_fit_waveの包絡曲線を計算する
        # self.id == self.target_id
        # best_fit_wave = self.load_waveforms()
        # from scipy.signal import hilbert, find_peaks# Hilbert変換を用いて包絡線を計算する
        # from scipy.interpolate import interp1d, UnivariateSpline
        # analytic_signal = hilbert(best_fit_wave['Acc_filterd(cm/s/s)'])
        # amplitude_envelope = np.abs(analytic_signal)

        # peaks, _ = find_peaks(amplitude_envelope)
        # f = interp1d(peaks, amplitude_envelope[peaks], kind='linear', fill_value='extrapolate')
        # envelope_peaks = f(np.arange(len(waveform)))
        
        # spline = UnivariateSpline(peaks, amplitude_envelope[peaks])
        # envelope_spline = spline(np.arange(len(waveform)))
        # peaks, _ = find_peaks(envelope_spline)
        # f = interp1d(peaks, envelope_spline[peaks], kind='linear', fill_value='extrapolate')
        # envelope_spline_peaks  = f(np.arange(len(waveform)))
        
        # # envelope_spline_peaksを0以下を0にする
        # envelope = envelope_spline_peaks if np.min(envelope_spline_peaks) > 0 else envelope_spline_peaks - np.min(envelope_spline_peaks)
        # envelope = envelope / np.max(envelope)

        # plt.plot(time, envelope, label="Envelope")
        # plt.legend()
        # plt.show()

        # print(envelope)
        # print(f"max envelope = {np.max(envelope)}")
        # print(f"min envelope = {np.min(envelope)}")
        # print(self.conf.phaseRoop)

        # phase_roop = self.average_spectrum["phase(rad)"]
        # for roop in range(self.conf.phaseRoop):
        #     waveform = self.phase_assignment(
        #                     spectrum = self.average_spectrum["FourierSpectrum(cm/s)"],
        #                     phase    = phase_roop
        #                     )
        #     accEnvelope_roop = waveform * np.power(envelope,1/self.conf.phaseRoop)
        #     accSpectrum_roop = self.calc_AccFFT(acc = accEnvelope_roop)            
        #     phase_roop = np.angle(accSpectrum_roop)   
        # waveform = accEnvelope_roop

        # print(f"dt : {self.dt}")
        # print(f"Max time : {len(waveform)*self.dt}")
        # print(f"Max Acc : {max(best_fit_wave['Acc_filterd(cm/s/s)']):.2f} cm/s/s")
        # print(f"Max Wave : {max(waveform):.2f} cm/s/s")

        # plt.plot(time, waveform)
        # plt.show()
        # plt.close()






def main():
    conf = Config()
    calculus = Calculus()
    damping = conf.damping

    calculator = ResponseSpectrumCalculator(conf, calculus, damping)

    calculator.calculate_response_spectrum()
    print(calculator.response_spectrum)
    calculator.save_response_spectrum()
    calculator.calculate_statistics()
    calculator.find_target_id()
    calculator.load_fft_phase_spectrum()
    calculator.create_average_spectrum()
    calculator.save_average_spectrum()
    calculator.plot_response_spectrum()
    calculator.calculate_average_spectrum_fit_waveform()


if __name__ == "__main__":
    main()



