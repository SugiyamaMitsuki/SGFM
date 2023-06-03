import os

class Config(object):
    def __init__(self):

        ## このconfig.pyの絶対パス
        self.folder = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)
        # self.input_path = str(self.folder).replace("program","InputData")
        self.input_path = self.folder + "/InputData"
        self.wave_path = self.folder + "/WaveData"
        self.element_path = self.folder + "/ElementData"
        self.fig_path = self.folder + "/Figures"
        
        ## 諸設定（震源情報以外の設定）
        self.EQid = "1923KANTO" # ファイル名など
        self.Number_of_Generations = 300
        
        # 震源情報ファイル
        self.source_parameters_path = "./InputData/SourceParameters.xlsx"
        
        # ボーリングデータ
        self.soilData_path = "./InputData/SoilData.xlsx"
        
        # 観測点（計算対象緯度経度）
        self.lon_station = 139.67052772#←笹塚
        self.lat_station =  35.67425031
        self.depth_station = 0.0     #km
        # self.lon_station = 139.741337#←品川
        # self.lat_station = 35.626151
        # self.depth_station = 0.0     #km
        
        # 計算条件
        self.N  = 2**14         # 種地震の時刻歴におけるデータ数(2^N)
                                # FFT時に後続の0を追加する処理を省いているため、
                                # 包絡曲線の倍以上の長さを確保する
        self.dt = 0.01
        self.df = 1.0/self.dt/self.N
        self.needTimeLength = 120 # 秒
        
        # 位相割り当て回数
        self.phaseRoop = 50

        # seed値 for white noise
        self.SEED = 42
        
        # 有効振動数帯域
        self.isTaper=True
        self.left  = 0.01
        self.start = 0.1
        self.end   = 20.0
        self.right = 30.0

        # response spectrum 
        self.damping = 0.05 
        
        # ラディエーションパターン(未対応)
        self.Radiation_pattern = "Isotropic"  #Frequency_dependence, Isotropic
        
if __name__ == "__main__":
    param = Config()

    print(f"param.folder:{param.folder}")
    print(f"param.N:{param.N}")
    print(f"Time:{param.N*param.dt} [sec]")