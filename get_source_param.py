import os
import random
import warnings

import numpy as np
import pandas as pd
import pygmt
from pyproj import Geod

class SourceParamReader(object):
    """
    エクセルファイルから震源情報を取得するクラス
    このクラスではzは上方向（上空方向）を正とする
    """
    
    def __init__(self):
        
        ## この.pyの絶対パス
        self.folder = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath(__file__)
        self.grid_template = None
    
    def read_xlsx(self,input_path):
        """
        震源情報の読み込み。要素断層は断層全体を同一面内で等分割している
        """
        # エクセルファイルからデータの読み取り
        source_parameters = pd.read_excel(input_path, sheet_name="Source_Parameters", header=0, index_col=0)
        epi_param = pd.read_excel(input_path, sheet_name="epicenter", header=0, index_col=0)
        V_ij = pd.read_excel(input_path, sheet_name="V_ij", header=1, index_col=0)
        D_ij = pd.read_excel(input_path, sheet_name="D_ij", header=1, index_col=0)
        sigma_ij = pd.read_excel(input_path, sheet_name="sigma_ij", header=1, index_col=0)
        
        # 巨視的震源特性
        self.M          = source_parameters.loc["震源規模"]["値"]
        self.L          = source_parameters.loc["断層全体の長さ"]["値"]
        self.TotalArea  = source_parameters.loc["断層総面積"]["値"]
        self.M0         = source_parameters.loc["総地震モーメント"]["値"]
        self.Mw         = source_parameters.loc["モーメントマグニチュード"]["値"]
        self.W          = source_parameters.loc["断層幅"]["値"]
        self.strike        = source_parameters.loc["走向角"]["値"]
        self.dip        = source_parameters.loc["傾斜角"]["値"]
        self.depth_min  = source_parameters.loc["断層上端深さ"]["値"]
        self.D          = source_parameters.loc["平均すべり量"]["値"]
        
        # その他の震源特性
        self.mu = source_parameters.loc["剛性率"]["値"]
        self.Vs = source_parameters.loc["平均S波速度"]["値"]
        self.Vr = source_parameters.loc["破壊伝播速度"]["値"]
        self.rho = source_parameters.loc["密度"]["値"]
        
        # 地震基盤面
        self.rho_sb = source_parameters.loc["地震基盤面密度"]["値"]
        self.Vs_sb  = source_parameters.loc["地震基盤面密度S波速度"]["値"]
        
        #高周波遮断振動数
        self.fmax = source_parameters.loc["高周波遮断振動数"]["値"] 
        
        # 要素断層震源特性
        self.D_ij = D_ij.to_numpy()
        self.V_ij = V_ij.to_numpy()
        self.sigma_ij = sigma_ij.to_numpy()

        # 破壊開始点緯度経度
        self.epi_lat = epi_param.iloc[0][1]
        self.epi_lon = epi_param.iloc[1][1]
        self.epi_depth = epi_param.iloc[2][1]
        # 断層端部と破壊開始点の距離（断層面内の距離）
        self.delta_W = epi_param.iloc[3][1]
        self.delta_L = epi_param.iloc[4][1]

        # 要素断層のテンプレート配列
        self.grid_template = np.zeros_like(self.D_ij,dtype=float)
        
        # 重ね合わせ数
        self.N_elements = self.grid_template.size
        
        #　要素分割数、要素断層サイズ
        self.x_num = self.grid_template.shape[0]
        self.y_num = self.grid_template.shape[1]
        self.dx = float(self.W/self.x_num)
        self.dy = float(self.L/self.y_num)
        self.faultArea = self.W * self.L
        self.ElementArea = self.dx * self.dy
        
        # 統計的グリーン関数法による1923年関東地震(MJ7.9)の広域強震動評価_壇・他_2000_日本建築学会構造系論文集
        # self.M0_ij = np.ones_like(self.grid_template) * self.mu * 13.0 * 10.0 * 177.0 *10**4        
        # 各要素断層のパラメータから要素断層の地震モーメントを求める。M=μSD N/m2 km2 cm
        self.M0_ij = self.mu * self.dx * self.dy * self.D_ij * 10**4    
        
        # 緯度経度の縮尺比調整
        Rx = 2.0 * np.pi * 6378137 /1000 #赤道周囲 km　https://ja.wikipedia.org/wiki/GRS80
        Ry = 2.0 * np.pi * 6356752.314140356 /1000 #子午線周囲 km
        self.lon_per_km = abs(1/ (Rx * np.cos(np.radians(self.epi_lat))/360.0)) #経度/km      #ここにlonを入れてた注意
        self.lat_per_km = abs(1/ (Ry / 360.0))      #緯度/km
    
    
    def calc_element_coordinates_Information(self):
        """
        要素断層の位置情報を計算する。
        """
        # kmで表現した座標　
        # 要素断層の回転の基準点は回転前の南西端要素=(0,0,0)
        self.element_x = np.zeros_like(self.grid_template,dtype=float)
        self.element_y = np.zeros_like(self.grid_template,dtype=float)
        self.element_z = np.zeros_like(self.grid_template,dtype=float)
        
        x_temp = np.arange(0, self.dx*self.x_num, 1) * self.dx    
        y_temp = np.arange(0, self.dy*self.y_num, 1) * self.dy
        
        # 回転前
        # X座標
        element_x_temp = np.zeros_like(self.grid_template)
        for i in range(self.grid_template.shape[0]):
            for j in range(self.grid_template.shape[1]):
                element_x_temp[i][j] = x_temp[i]

        # Y座標
        element_y_temp = np.zeros_like(self.grid_template)
        for i in range(self.grid_template.shape[0]):
            for j in range(self.grid_template.shape[1]):
                element_y_temp[i][j] = y_temp[j]
        
        # Z座標
        element_z_temp = np.zeros_like(self.grid_template)

        # 回転行列を求める 
        px = 0/180*np.pi
        py = self.dip/180*np.pi # 傾斜dip
        pz = -self.strike/180*np.pi # 走向strike
        print(f"走向:N{self.strike}°E  傾斜:{self.dip}°")
        rotationMatrix = self.calc_rotation_matrix(px,py,pz)

        # 回転後のxyz座標(km)
        for i in range(self.grid_template.shape[0]):
            for j in range(self.grid_template.shape[1]):
                
                point = np.array([element_x_temp[i][j],element_y_temp[i][j],element_z_temp[i][j]]) # 回転の対象点
                point0_rot = np.dot(rotationMatrix, point.T)

                self.element_x[i][j] = point0_rot[0]
                self.element_y[i][j] = point0_rot[1]
                self.element_z[i][j] = point0_rot[2]
        
        # 震源のxyz(km)
        epi_point = np.array([self.delta_W, self.delta_L, 0])
        epi_point_rot = np.dot(rotationMatrix, epi_point.T)
        self.epi_point_x = epi_point_rot[0]
        self.epi_point_y = epi_point_rot[1]
        self.epi_point_z = epi_point_rot[2]

        # xyz　破壊開始点との相対値(km)
        self.element_dx = self.element_x - self.epi_point_x
        self.element_dy = self.element_y - self.epi_point_y
        self.element_dz = self.element_z - self.epi_point_z
        self.element_dL = np.power(self.element_dx**2 + self.element_dy**2 + self.element_dz**2 ,1.0/2.0)

    
    def calc_element_epi_distance(self,lon_station,lat_station,depth_station):
        """要素断層ごとの震源距離(km)を計算する。

        Args:
            lon_station (float): 観測点経度(degree)
            lat_station (float): 観測点緯度(degree)
            depth_station (float): 観測点緯度深さ(km)
        """
        
        self.element_epi_distance = np.zeros_like(self.grid_template,dtype=float)
        self.element_epi_lon = np.zeros_like(self.grid_template,dtype=float)
        self.element_epi_lat = np.zeros_like(self.grid_template,dtype=float)
                
        # 破壊開始点の震源距離 km
        g = Geod(ellps='WGS84')
        azimuth, back_azimuth, distance_2d = g.inv(lon_station, lat_station, self.epi_lon, self.epi_lat) #(m)
        self.epi_distance = np.power( (distance_2d/1000.0)**2 + (self.epi_depth-depth_station)**2 ,1.0/2.0)
        
        for i in range(self.grid_template.shape[0]):
            for j in range(self.grid_template.shape[1]):
        
                dx_lon = self.element_dx[i][j] * self.lon_per_km
                dy_lat = self.element_dy[i][j] * self.lon_per_km

                self.element_epi_lon[i][j] = self.epi_lon + dx_lon
                self.element_epi_lat[i][j] = self.epi_lat + dy_lat
                
                azimuth, back_azimuth, distance_2d = g.inv(lon_station, lat_station, self.element_epi_lon[i][j], self.element_epi_lat[i][j]) #(m)
                element_depth = self.element_z[i][j] #km
                self.element_epi_distance[i][j] = np.power( (distance_2d/1000)**2 + (element_depth-depth_station)**2 ,1/2)
    
    
    def calc_rotation_matrix(self,px,py,pz):
        """回転行列を計算する関数
           Function to calculate the rotation matrix.
           (参考)
           https://rikei-tawamure.com/entry/2019/11/04/184049 https://org-technology.com/posts/rotational-transformation-matrix.html
        Args:
            px (float): X軸の回転(radian)
            py (float): Y軸の回転(radian)
            pz (float): Z軸の回転(radian)

        Returns:
            ndarray: rotationMatrix
        """
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(px), np.sin(px)],
                       [0, -np.sin(px), np.cos(px)]])
        Ry = np.array([[np.cos(py), 0, -np.sin(py)],
                        [0, 1, 0],
                        [np.sin(py), 0, np.cos(py)]])
        Rz = np.array([[np.cos(pz), np.sin(pz), 0],
                        [-np.sin(pz), np.cos(pz), 0],
                        [0, 0, 1]])
        # R = Rz.dot(Ry).dot(Rx)
        R = Rx.dot(Ry).dot(Rz)
        rotationMatrix = R.T
        
        return rotationMatrix
    
    
    def saveElementModel(self,output,rake=143.0):
        """
        エクセルファイルから読み込んだ断層モデルをcsvファイルに保存する
        Save the fault model read from the Excel file to a csv file.
        """
        epiInfo = pd.DataFrame(columns=["No.","lon","lat","depth","M0","strike","dip","rake","area","epiDistance","ruptureTime","randomDelay","slip","slipVelocity","stressDrop"])

        for i, row in enumerate(self.element_dx.reshape(-1)):
            
            lon    = self.element_dx.reshape(-1)[i] * self.lon_per_km + self.epi_lon
            lat    = self.element_dy.reshape(-1)[i] * self.lat_per_km + self.epi_lat
            depth  = self.element_dz.reshape(-1)[i] - self.epi_depth
            m0     = self.M0_ij.reshape(-1)[i]
            strike    = self.strike
            dip       = self.dip
            rake      = rake #https://www.bousai.metro.tokyo.lg.jp/_res/projects/default_project/_page_/001/000/401/assumption.part2-3-2.pdf
            area      = self.ElementArea
            epiDistance = self.element_epi_distance.reshape(-1)[i]
            ruptureTime ,randomDelay = self.calc_time_delay(
                                r_pq        = self.element_epi_distance.reshape(-1)[i],
                                r_o         = self.epi_distance,
                                Vs          = self.Vs,
                                Xi_pq       = self.element_dL.reshape(-1)[i],
                                Vr          = self.Vr,
                                W           = np.minimum(self.dx,self.dy)
                                )
            slip      = self.D_ij.reshape(-1)[i]
            slipVelocity = self.V_ij.reshape(-1)[i]
            stressDrop = self.sigma_ij.reshape(-1)[i]

            epiInfo.loc[i] = [i,lon,lat,depth,m0,strike,dip,rake,area,epiDistance,ruptureTime,randomDelay,slip,slipVelocity,stressDrop]

        epiInfo.to_csv(output,index=False)
    

    def calc_time_delay(self, r_pq, r_o, Vs, Xi_pq, Vr,W):
        """
        要素断層から観測点までの時間遅れを計算する。

        Parameters:
        r_pq (float): 要素断層から観測点までの距離 km
        r_o (float): 観測点から震源までの距離 km
        Vs (float): 地震波伝播速度 km/s
        Xi_pq (float): 破壊開始点と要素断層の距離 km
        Vr (float): 要素断層破壊伝播速度 km/s 
        W (float): 要素断層の最小辺の長さ km

        Returns:
        Tuple[float, float]: 要素断層から観測点までの時間遅れとランダムな時間遅れ
        """
        # seed値の解放
        random.seed()
        p = random.uniform(-0.5, 0.5) # -0.5から0.5で乱数を生成する

        e_pq = p * W / Vr # ランダムな時間遅れ
        _t_pq  =  (r_pq - r_o) / Vs + Xi_pq/Vr # 要素断層から観測点までの時間遅れ

        return _t_pq,e_pq
    

    def draw_FaultModelMap(self,Map_left=138.5,Map_right=141.0,Map_up=36,Map_down=34.5,output="test.png",lon_station=None,lat_station=None):
        """断層モデルを地図上に描写するクラス
           Class for depicting fault models on maps.
        Args:
            Map_left (float, optional) : _左端経度_.      Defaults to 138.5.
            Map_right (float, optional): _右端経度_.      Defaults to 141.
            Map_up (float, optional)   : _上端緯度_.      Defaults to 36.5.
            Map_down (float, optional) : _下端緯度_.      Defaults to 34.
            output (str, optional)     : _出力ファイル名_. Defaults to "test.png".
        """
        
        self.Map_left  = Map_left
        self.Map_right = Map_right
        self.Map_up    = Map_up
        self.Map_down  = Map_down
        self.projection = "M15c"
        self.resolution ='15s' # 解像度 度d分m秒s "01d", "30m", "20m", "15m", "10m", "06m", "05m", "04m", "03m", "02m", "01m", "30s", "15s", "03s", or "01s"
        self.scaleKM = 50
        self.plot_cmap = "seis"
        
        # 地図の箱を用意する
        fig = self.makeBaseMap()
        
        for i, row in enumerate(self.element_dx.reshape(-1)):    
            
            centerLon       = self.element_dx.reshape(-1)[i] * self.lon_per_km + self.epi_lon
            centerLat       = self.element_dy.reshape(-1)[i] * self.lat_per_km + self.epi_lat
            centerDepth_km  = self.element_dz.reshape(-1)[i]
            Dx_km           = self.dx
            Dy_km           = self.dy
            strike          = self.strike
            dip             = self.dip
            fillvalue       = round(self.D_ij.reshape(-1)[i]/np.max(self.D_ij.reshape(-1)),3)

            cornerPoints_km = [
                            [-Dx_km/2.0,-Dy_km/2.0,0],
                            [ Dx_km/2.0,-Dy_km/2.0,0],
                            [ Dx_km/2.0, Dy_km/2.0,0],
                            [-Dx_km/2.0, Dy_km/2.0,0],
                            [-Dx_km/2.0,-Dy_km/2.0,0]
                            ]
            edgeLonList = []
            edgeLatList = []
            edgeDepthList = []
            for cornerPoint in cornerPoints_km:
                
                point = np.array([cornerPoint[0],cornerPoint[1],0])
                px = 0/180*np.pi
                py = self.dip/180*np.pi # 傾斜dip
                pz = -self.strike/180*np.pi # 走向strike
                rotationMatrix = self.calc_rotation_matrix(px,py,pz)
                point0_rot = np.dot(rotationMatrix, point)
                
                element_Lon = point0_rot[0] * self.lon_per_km + centerLon
                element_Lat = point0_rot[1] * self.lat_per_km + centerLat                      
                element_Depth = point0_rot[2] + centerDepth_km
                edgeLonList.append(element_Lon)
                edgeLatList.append(element_Lat)
                edgeDepthList.append(element_Depth)
            
            # セグメントの枠
            fig.plot(
                    x = edgeLonList,
                    y = edgeLatList,
                    pen="0.0001p,black",
                    transparency = 40,
                    )
            
            # 透過度で色分け
            transparency = 100-int(fillvalue*100)  if fillvalue < 0.98 else 0
            fig.plot(
                    x     = edgeLonList,
                    y     = edgeLatList,
                    color =  "violetred3",#deeppink3 red3 violetred3 https://docs.generic-mapping-tools.org/dev/_images/GMT_RGBchart.png
                    # pen="0.001p,violetred3",
                    transparency = transparency,
                    )
            
            # パラメータ記載
            text = self.D_ij.reshape(-1)[i]
            fig.text(
                    x = centerLon,
                    y = centerLat,
                    text = f"{text:.0f}",
                    )
        fig.text(
                    x = (self.Map_right-self.Map_left)*0.7+self.Map_left,
                    y = (self.Map_up-self.Map_down)*0.1+self.Map_down,
                    text = f"Values on the map are in Disp (cm).",
                    )
        
        # 破壊開始点のプロット
        fig.plot(
                x            = float(self.epi_lon),
                y            = float(self.epi_lat),
                fill         = '230/150/5',          # 塗りつぶし色の指定 red
                style        = 'a0.6c',             # 固定サイズの場合は (symbol)(size) 指定
                pen          = 'thinner,black',     # 縁取りのペン
                transparency = 0                # コマンド全体に影響する透明度設定
            )
        
        if lon_station != None:
            # 観測点のプロット
            fig.plot(
                    x            = float(lon_station),
                    y            = float(lat_station),
                    fill         = 'blue',          # 塗りつぶし色の指定 red
                    style        = 'c0.5c',             # 固定サイズの場合は (symbol)(size) 指定
                    pen          = 'thinner,black',     # 縁取りのペン
                    transparency = 0                # コマンド全体に影響する透明度設定
            )

        self.savePyGMT(fig,output)
             
    def makeBaseMap(self):
        """
        PyGMTの下絵を作成する
        Create a base sketch for PyGMT.
        """
        import pygmt
        fig = pygmt.Figure()
        fig.basemap(
                region      = [self.Map_left, self.Map_right, self.Map_down, self.Map_up],
                projection  = self.projection,
                frame       = ['WSen', 'xaf', 'yaf','zaf'], #タイトルを付ける場合'WSen+t"Seismicity Test"'
                )
        
        # 標高水深      
        grid_data = pygmt.datasets.load_earth_relief(resolution=self.resolution, region = [self.Map_left, self.Map_right, self.Map_down, self.Map_up],)

        # 勾配計算
        gradient_data = pygmt.grdgradient(grid = grid_data, azimuth = [(self.Map_down+self.Map_up)/2,(self.Map_left+self.Map_right)/2],normalize = 'e0.7')
        fig.grdimage(
                        region   = [self.Map_left, self.Map_right, self.Map_down, self.Map_up],
                        grid     = grid_data, 
                        shading  = gradient_data,
                        cmap     = 'geo',#gray geo batlow
                        transparency = 50,        # コマンド全体に影響する透明度設定
                    )
        #縮尺スケールの設置
        map_scale = f"{self.Map_right-(self.Map_right-self.Map_left)*0.2}/{self.Map_down+(self.Map_up-self.Map_down)*0.10}/{(self.Map_down+self.Map_up)/2}/{self.scaleKM}"
        
        # https://www.pygmt.org/latest/api/generated/pygmt.Figure.coast.html
        fig.coast(
                        shorelines  = 'thinner,black@40',
                        area_thresh = '100',
                        resolution  = 'f',# 'c', 'l', 'i', 'h', 'f' の順に高くなる
                        map_scale   = map_scale,
                        transparency = 80,        # コマンド全体に影響する透明度設定
                        borders     = "1/0.5p",
                        rivers      = "a/1p,skyblue,solid",
                        water       = "10@90",#'90@10',   # 海域をすこし暗くする "skyblue",# lightblue
                    )        
        fig.coast(shorelines=True)
        
        return fig
    
    def savePyGMT(self,fig,output_map_filepath):     
        """
        PyGMTで作成した図を保存する
        Saving diagrams created with PyGMT.
        """   
        try:
            os.remove("test.png")
        except:
            None
        try:
            os.remove(output_map_filepath)
        except:
            None
            
        fig.savefig("test.png")
        os.rename("test.png",output_map_filepath)
    
    
        

if __name__ == "__main__":
    param = SourceParamReader()
    param.read_xlsx("./InputData/Source_Parameters.xlsx")
    param.calc_element_coordinates_Information()




    # param.calc_element_epi_distance(139.76718460478475,35.681304321743575,10)
    
    # lon_station = 139.67052772#←笹塚
    # lat_station =  35.67425031
    
    # print("地図描写")
    # param.draw_FaultModelMap(output="./fig/faultMAP.png",lon_station=lon_station,lat_station=lat_station)
    
    
