import os
import pandas as pd 
import numpy as np
import math
import random
from typing import Tuple

class BasenSpectrum(object):
    """
    地震基盤面のスペクトルUpq(f)の計算をする
    """
    def __init__(self,N,dt,M0,rho,Vs,fmax,rho_sb,Vs_sb,N_elements,faultArea):
        """
        Parameters:
        N (int): データ数
        dt (float): 刻み
        M0 (float): 地震モーメント Nm
        rho (float): 震源の密度 g/cm3
        Vs (float): 震源のせん断波速度 km/s
        fmax (float): 高周波遮断振動数 Hz
        rho_sb (float): 地震基盤面密度 g/cm3
        Vs_sb (float): 地震基盤面せん断波速度 km/s
        N_elements (int): 重ね合わせ数
        faultArea (float): 断層面積 km2
        """
        
        ## この.pyの絶対パス
        self.folder = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)
        
        # データ数
        self.N_time = N
        self.N_freq = N//2 # ナイキスト振動数の半分(加速度フーリエスペクトル振幅の配列の大きさ)                                                     
        
        # 刻み
        self.dt = dt
        self.df = 1.0/dt/N
        
        #有効振動数
        self.maxFreq_Effect = 1.0/dt/2.0
        
        # 周波数軸（折り返し振動数以下）
        self.freq = np.arange(0,self.maxFreq_Effect,self.df)

        # すべての要素断層共通のパラメータ
        self.M0 = M0                                            # 地震モーメント Nm
        self.RP = 0.63                                          # ラディエーションパターン　RadiationPattern 等方とした場合のみに対応
        self.FS = 2.0                                           # 自由表面による増幅係数
        self.rho = rho * 1000.0                                 # 震源の密度 g/cm3→kg/m3
        self.Vs =  Vs  * 1000.0                                 # 震源のせん断波速度 km/s→m/s
        self.fmax = fmax                                        # 高周波遮断振動数 Hz
        self.m = 4.2                                            # 定数
        self.Qf = self.calc_Q(flag="satoh1997")                 # Q値 
        self.rho_sb = rho_sb * 1000.0                           # 地震基盤面密度 g/cm3→kg/m3
        self.Vs_sb =Vs_sb * 1000.0                              # 地震基盤面せん断波速度 km/s→m/s
        self.N_elements = N_elements                            # 重ね合わせ数
        self.faultArea = faultArea * 10.0**6                    # 断層面積　km2→m2

    def fix_seed(self,seed: int = 0):
        """
        乱数のシードを固定する

        Parameters:
        seed (int): 乱数のシード
        """
        random.seed(seed)
        np.random.seed(seed)


    def calc_Q(self, flag: str = "satoh1997") -> np.ndarray:
        """
        Q値の計算

        Parameters:
        flag (str): Q値の計算方法を指定するフラグ

        Returns:
        np.ndarray: Q値
        """
        if flag=="satoh1997":
            a = 110.0
            b = 0.69
            Qf = np.zeros_like(self.freq,dtype=float)
            target_index = (self.freq <= 0.8).argmax() # 0.8Hz以下のQ値のインデックス
            Qf[:target_index]= 94.0                                         # f <= 0.8Hz
            Qf[target_index:] = a * np.power(self.freq[target_index:], b)   # f > 0.8Hz
            return Qf
        else:
            None

    def calc_Spectrum(self,sigma_pq,D_pq,M0_pq,r_pq,V_pq,area_pq):
        """
        地震基盤面の加速度フーリエスペクトルUpq(f)（単位:cm/s）と
        点震源と面震源の補正係数の計算 

        Parameters:
        sigma_pq (float): 要素断層の実効応力 (MPa)
        D_pq (float): 要素断層の最大すべり量 (cm)
        M0_pq (float): 要素断層の地震モーメント (Nm)
        r_pq (float): 要素断層震源距離 (km)
        V_pq (float): 要素断層のすべり速度最大値 (cm/s)
        area_pq (float): 要素断層の面積 (km2)

        Returns:
        Tuple[np.ndarray, np.ndarray]: 補正係数と加速度フーリエスペクトル
        """
        if M0_pq < 0.001:
            print("M0_pq is too small → skip calculation")
            return np.zeros_like(self.freq,dtype=float),np.zeros_like(self.freq,dtype=float)

        # 要素断層ごとのパラメータ
        # 単位をそろえる
        sigma_pq = sigma_pq * 10.0**6 # 要素断層の実効応力 MPa→N/m2
        r_pq     = r_pq *1000.0       # 要素断層震源距離 km→m
        M0_pq    = M0_pq              # 要素断層の地震モーメント Nm
        D_pq     = D_pq /100.0        # 要素断層の最大すべり量　cm→m
        V_pq     = V_pq/100           # 要素断層のすべり速度最大値　cm/s→m/s
        area_pq  = area_pq * 10.0**6  # 要素断層の面積　km2→m2

        #コ―ナー振動数 Hz
        fc_pq = ((7.0/16.0)**(1.0/6.0)) / ((np.pi**(1.0/2.0))) * self.Vs * (sigma_pq/M0_pq)**(1.0/3.0)
        
        ## 地震基盤における加速度フーリエスペクトルA(ω)[cm/s]の作成
        # 震源スペクトルS(f)　s1~s4
        
        # s1:ラディエーションパターンの影響
        s1 = (self.RP * self.FS) / (4.0*np.pi*self.rho*np.power(self.Vs,3.0))
        
        # s2:Brooneの震源変位スペクトル M0pq / (1+(f/fcpq)^2 )
        s2 = M0_pq  / (1.0 + np.power(self.freq/fc_pq, 2)) 
        
        # s3:fmaxの影響
        s3 = np.power((1.0 + np.power(self.freq/self.fmax, self.m)), -1.0/2.0)
        
        # s4:距離減衰
        s4 = 1.0/r_pq
        
        # 伝播特性R(f) r1~r2
        r1 = np.exp(-np.pi * self.freq * r_pq / self.Qf / self.Vs)
        r2 = np.sqrt( (self.rho *self.Vs) / (self.rho_sb*self.Vs_sb) )
        
        # 要素地震波の加速度スペクトル * (iω)^2
        ue_pq = (s1*s2*s3*s4) * (r1*r2) * np.power(2*np.pi*self.freq,2)
        # 0除算が含まれる→ω=0の処理
        ue_pq[0] = M0_pq*self.FS*self.RP/4.0/np.pi/self.rho/self.Vs**3/r_pq*np.sqrt( (self.rho *self.Vs) / (self.rho_sb*self.Vs_sb) )
        # m → cm
        ue_pq = ue_pq * 100.0
        
        
        ## 補正係数 点震源から面震源への補正
        # （すべり速度時間関数を考慮していることになる）
        # ω2モデル→ハスケルモデルへの補正に相当
        # 断層の非一様すべり破壊を考慮した半経験的波形合成法による強震動予測 壇 一男, 佐藤 俊明 1998
        
        # V_pq = 2.0 * sigma_pq / self.rho * self.Vs
        lamda_pq = math.sqrt(area_pq/np.pi) # 要素断層の等価半径（要素断層のサイズ一定）
        fD_pq = 1.0/(2.0*np.pi) * (V_pq/D_pq)
        fS_pq = 1.0/(2.0*np.pi) * (2.0*self.Vs/lamda_pq)
        
        f1 = (1.0 + 1j * self.freq /fc_pq)**2 
        f2 = 1.0 + 1j * self.freq/fD_pq
        f3 = 1.0 + 1j * self.freq/fS_pq
        correction_factor = f1/(f2*f3)
        
        return (correction_factor, ue_pq)    

    def calc_Envelope(self,X,M0):
        """
        包絡関数を計算する。

        Parameters:
        X (float): 震源距離 km
        M0 (float): 地震モーメント Nm

        Returns:
        np.ndarray: 包絡関数
        """
        # 佐藤ほか1994による
        # 「ボアホール観測記録を用いた表層地盤同定手法による工学的基盤波の推定及びその統計的経時特性」
        
        self.time     = np.arange(0, self.N_time*self.dt, self.dt)
        self.envelope = np.zeros_like(self.time,dtype=float)

        # 気象庁マグニチュード
        Mj = 1/1.605*(math.log10(M0*(10**7))-15.507) #←J-SHISの式
        # Mj = 1/1.17*(math.log10(M0)-10.172)　# https://jishin.go.jp/main/choukihyoka/04mar_kakuritsu/setsumei_3.pdf
        # print(f"    M_JMA: {Mj:.3f}")
        
        tb = 10**(0.229 * Mj - 1.112)
        tc = tb + 10**(0.433 * Mj - 1.936)
        td = tc + 10**(0.778 * math.log10(X) - 0.340)
        
        tb_flag = np.argmax(self.time > tb)
        tc_flag = np.argmax(self.time > tc)
        td_flag = np.argmax(self.time > td)

        self.envelope[:tb_flag]         = np.power(self.time[:tb_flag]/tb,2)
        self.envelope[tb_flag:tc_flag]  = 1.0
        self.envelope[tc_flag:td_flag]  = np.exp(-(math.log(10)) * (self.time[tc_flag:td_flag] - tc)/(td - tc))

        return self.envelope
    
    def calc_white_Noise(self):
        """
        ホワイトノイズの作成
        """
        self.white_noise = np.random.randn(self.N_time)


    def calc_Envelope_white_Noise(self,envelope):
        """
        ホワイトノイズに包絡関数を乗じて作成した時刻歴波形

        Parameters:
        envelope (np.ndarray): 包絡関数

        Returns:
        np.ndarray: ホワイトノイズに包絡関数を乗じた波形
        """

        noise_Envelope = envelope * self.white_noise
        
        return noise_Envelope

    def calc_noise_Spectrum(self, noise_Envelope: np.ndarray) -> np.ndarray:
        """
        ホワイトノイズｘ包絡関数のスペクトル、位相スペクトル(-π~π)を計算する (折り返し振動数以下の範囲)
        """
        _noise_Spectrum =  np.fft.fft(noise_Envelope) * self.dt
        noise_phaseSpectrum = np.angle(_noise_Spectrum)
        
        return noise_phaseSpectrum
    
    def calc_phase(self, FFT: np.ndarray) -> np.ndarray:
        """
        FFTの位相を計算する

        Parameters:
        FFT (np.ndarray): FFT

        Returns:
        np.ndarray: FFTの位相
        """
        return np.angle(FFT)
    
    
    def phase_assignment(self, spectrum, phase):
        """
        加速度フーリエスペクトル振幅に乱数位相を割り当てて、加速度波形を返す

        Parameters:
        spectrum (np.ndarray): 加速度フーリエスペクトル振幅
        phase (np.ndarray): 乱数位相

        Returns:
        np.ndarray: 加速度波形
        """
        # F(ω) = A・exp(iω)
        
        randomFFT = spectrum[:self.N_freq] * np.exp(1j*phase[:self.N_freq])
        randomFFT = np.append(randomFFT, np.zeros_like(randomFFT))
        
        acc = np.fft.ifft(randomFFT.real) / self.dt * 4 # 折り返し振動数以下の実部のみ→4倍
        acc = acc.real[:self.N_time]
        
        return acc

    def calc_AccFFT(self,acc):
        """
        加速度のFFTを計算する

        Parameters:
        acc (np.ndarray): 加速度

        Returns:
        np.ndarray: 加速度のFFT
        """
        
        acc = acc - acc.mean()
        AccSpectrum = np.fft.fft(acc)[:self.N_freq] * self.dt
        
        return AccSpectrum


if __name__ == "__main__":
    None
    

    # def calc_time_delay(self, r_pq, r_o, Vs, Xi_pq, Vr,W):
    #     """
    #     要素断層から観測点までの時間遅れを計算する。

    #     Parameters:
    #     r_pq (float): 要素断層から観測点までの距離 km
    #     r_o (float): 観測点から震源までの距離 km
    #     Vs (float): 要素断層破壊伝播速度 km/s
    #     Xi_pq (float): 破壊開始点と要素断層の距離 km
    #     Vr (float): 地震波伝播速度 km/s
    #     W (float): 要素断層の最小辺の長さ km

    #     Returns:
    #     Tuple[float, float]: 要素断層から観測点までの時間遅れとランダムな時間遅れ
    #     """

    #     p = random.uniform(-0.5, 0.5) # -0.5から0.5で乱数を生成する

    #     e_pq = p * W / Vr # ランダムな時間遅れ
    #     _t_pq  =  (r_pq - r_o) / Vs + Xi_pq/Vr # 要素断層から観測点までの時間遅れ

    #     return _t_pq,e_pq