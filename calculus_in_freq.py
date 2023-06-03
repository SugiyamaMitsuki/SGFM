import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy import signal
import os
import warnings
warnings.simplefilter('ignore')

class Calculus(object):
    """
    周波数領域で微積分、フィルター処理をするプログラム
    """
    
    def __init__(self):
        self.folder = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)
    
    def calcDerivative(self,time=None,wave=None,derivative_times=0,isTaper=True,left=0.05,start=0.1,end=20,right=30):
        """
        Function to perform derivative or integral operations in the frequency domain
        
        Args:
        time : ndarray
            Time series
        wave : ndarray
            Ground motion series
        derivative_times : int, optional
            Order of differentiation. Positive for differentiation, negative for integration, 0 for filtering only
        isTaper : bool, optional
            If True, applies a tapering function
        left, start, end, right : float, optional
            Frequency band parameters for the tapering function
        
        Returns:
        ndarray
            Processed time series
        """
        
        # Error checking
        if not isinstance(time, np.ndarray) or not isinstance(wave, np.ndarray):
            raise ValueError('Time and wave must be numpy arrays')
        if len(time) != len(wave):
            raise ValueError('Time and wave must have the same length')
        if not isinstance(derivative_times, int):
            raise ValueError('Derivative times must be an integer')
        
        data_length = len(wave)
        
        dt=time[1]-time[0]
        
        #基線補正した元データ(時間領域)
        wave = wave - wave.mean()  
        
        #FFTのために元データの2倍の長さ以上の後続の0を付ける
        flag = math.floor(math.log2(len(time)*2))  #切り捨て 
        N=int(math.pow(2,flag+1))               #flagに1追加する
        t = np.arange(0, N, 1) * dt
        add_zeros = np.zeros(len(t)-len(wave))
        freq = np.linspace(0, 1.0/dt, N)  # 周波数軸
        
        wave=np.append(wave, add_zeros)
        
        # 高速フーリエ変換        
        fft = np.fft.fft(wave)
        
        if isTaper != False:
            #周波数フィルターの配列番号
            f_left,f_start,f_end,f_right = [int(x/(1/(N*dt))) for x in [left, start, end, right]]
            #コサインテーパー作成     
            self.cos_taper = np.zeros(len(freq))        
            self.cos_taper[f_left:f_start] = [1/2*(1+math.cos( 2*math.pi/(2*(f_start-f_left))*(x*N*dt - f_start))) for x in freq[f_left:f_start] ]
            self.cos_taper[f_start:f_end] = 1
            self.cos_taper[f_end:f_right] = [1/2*(1+math.cos( 2*math.pi/(2*(f_right-f_end))*(x*N*dt -  f_end ))) for x in freq[f_end:f_right] ]
            #costaper
            fft=fft*self.cos_taper
        
        #周波数領域で微積分
        calculus_array = np.zeros(len(t), dtype=complex)
        non_zero_inds = np.where(freq != 0)
        calculus_array[non_zero_inds] = np.power((1j*2*np.pi*freq[non_zero_inds]), derivative_times) * fft[non_zero_inds]
        
        calculus_array[N//2+1:] = 0
        wave_out = np.fft.ifft(calculus_array.real)*4 
        wave_out=wave_out.real
        
        return t[:data_length],wave_out[:data_length]
    
    
    def calcFFT(self,time=None,wave=None,derivative_times=0,isTaper=True,left=0.05,start=0.1,end=20,right=30,isParzen=True,parzen_setting=0.1):
        
        data_length = len(wave)
        dt = time[1] - time[0]
        
        #基線補正した元データ(時間領域)
        wave = wave - wave.mean()
        
        # data_lengthが2で割り切れない場合、FFTのために後続の0を付ける configで継続時間の2倍の長さを指定している
        if data_length % 2 != 0:
            flag = math.floor(math.log2(len(time)*2))  #切り捨て
            N=int(math.pow(2,flag+1))               #flagに1追加する
            t = np.arange(0, N, 1) * dt
            add_zeros = np.zeros(len(t)-len(wave))
            freq = np.linspace(0, 1.0/dt, N)  # 周波数軸
            wave=np.append(wave, add_zeros)
        else:
            N = data_length
            freq = np.linspace(0, 1.0/dt, N)
        
        # 高速フーリエ変換  フーリエスペクトル振幅
        fft = np.fft.fft(wave)/(1/dt)
        
        #コサインテーパー
        if isTaper != False:
            #周波数フィルターの配列番号
            f_left,f_start,f_end,f_right = int(left/(1/(N*dt))),int(start/(1/(N*dt))),int(end/(1/(N*dt))),int(right/(1/(N*dt)))
            #コサインテーパー作成     
            self.cos_taper = np.zeros(len(freq))        
            self.cos_taper[f_left:f_start] = [1/2*(1+math.cos( 2*math.pi/(2*(f_start-f_left))*(x*N*dt - f_start))) for x in freq[f_left:f_start] ]
            self.cos_taper[f_start:f_end] = 1
            self.cos_taper[f_end:f_right] = [1/2*(1+math.cos( 2*math.pi/(2*(f_right-f_end))*(x*N*dt -  f_end ))) for x in freq[f_end:f_right] ]
            #costaper
            fft=fft*self.cos_taper
        
        
        abs_fft=None
        #parzen window
        if parzen_setting:
            parzen_num=int(parzen_setting/(freq[1]-freq[0]))
            if parzen_num % 2 != 0:
                parzen_num+=1
            #奇数点のparzen
            w_parzen = signal.parzen(parzen_num)*parzen_setting
            abs_fft=np.convolve(w_parzen,np.abs(fft) , mode ='same') # valid same full  
        else:
            abs_fft=np.abs(fft)
        
        phase=np.angle(fft)
        
        return freq[:int(data_length/2)],abs_fft[:int(data_length/2)],phase[:int(data_length/2)]
    
    
        
if __name__ == '__main__':
    calc = Calculus()
    dt = 0.01
    N = 2**14
    t = np.arange(0, N*dt, dt)
    wave = np.random.randn(N)

    print("input N",N)
    t,wave_out = calc.calcDerivative(
                                    time=t,
                                    wave=wave,
                                    derivative_times=0,
                                    isTaper=True,
                                    left=1,
                                    start=7,
                                    end=15,
                                    right=20)


    plt.plot(t,wave)
    plt.plot(t,wave_out)
    plt.show()

    f = np.linspace(0, 1.0/dt//2, len(calc.cos_taper)//2)
    
    plt.plot(f,calc.cos_taper[:len(f)])
    plt.show()