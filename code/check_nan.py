import xarray as xr
import numpy as np
import geopandas as gpd
from shapely import vectorized

# ================== 1. 路径 ==================
nc_path  = "/exports/d4/WXY/Tropomi_avr_XCH4/China_XCH4/2023_china_0.01.nc"
city_shp = "/exports/d4/WXY/中国边界/city.shp"

# ================== 2. 读取 nc，自动识别变量和经纬度 ==================
ds = xr.open_dataset(nc_path)

print("数据变量:", list(ds.data_vars))
print("坐标:", list(ds.coords))

# 自动找 lat / lon
lat_name = None
lon_name = None
for coord in ds.coords:
    low = coord.lower()
    if lat_name is None and low.startswith("lat"):
        lat_name = coord
    if lon_name is None and (low.startswith("lon") or low.startswith("long")):
        lon_name = coord

if lat_name is None or lon_name is None:
    raise ValueError("找不到纬度/经度坐标，请检查 ds.coords")

# 自动找 XCH4 变量
var_name = None
for v in ds.data_vars:
    low = v.lower()
    if any(k in low for k in ["xch4", "ch4", "methane"]):
        var_name = v
        break

if var_name is None:
    raise ValueError("找不到包含 xch4/ch4/methane 的变量，请手动指定")

print(f"使用变量: {var_name}, lat: {lat_name}, lon: {lon_name}")
ch4 = ds[var_name].squeeze()  # 去掉 time 之类的维度

# ================== 3. 读取陕西边界 ==================
gdf_city = gpd.read_file(city_shp)
PROV_COL  = "省"
PROV_NAME = "陕西省"

gdf_sx_city = gdf_city[gdf_city[PROV_COL] == PROV_NAME].copy()
if gdf_sx_city.empty:
    raise ValueError(f"在 {PROV_COL} 中找不到 {PROV_NAME}")

gdf_sx_prov = gdf_sx_city.dissolve(by=PROV_COL)
poly = gdf_sx_prov.geometry.unary_union

# ================== 4. 先按陕西外接矩形裁剪，再用多边形掩膜 ==================
minx, miny, maxx, maxy = gdf_sx_prov.total_bounds
print("陕西 bounds:", minx, miny, maxx, maxy)

lat_vals = ds[lat_name].values
lon_vals = ds[lon_name].values

if lat_vals[0] < lat_vals[-1]:
    lat_slice = slice(miny, maxy)
else:
    lat_slice = slice(maxy, miny)

if lon_vals[0] < lon_vals[-1]:
    lon_slice = slice(minx, maxx)
else:
    lon_slice = slice(maxx, minx)

ch4_sx = ch4.sel({lat_name: lat_slice, lon_name: lon_slice})
data   = ch4_sx.values

# 网格经纬度
lon_grid, lat_grid = np.meshgrid(
    ch4_sx[lon_name].values,
    ch4_sx[lat_name].values,
)

# 陕西省内掩膜
inside = vectorized.contains(poly, lon_grid, lat_grid)

# 只在陕西省内统计
data_in_sx = data[inside]

total_in_sx = data_in_sx.size
nan_in_sx   = np.isnan(data_in_sx).sum()
finite_in_sx = np.isfinite(data_in_sx).sum()

print("\n===== 陕西省内 XCH4 空值检查 =====")
print(f"总像元数: {total_in_sx}")
print(f"非空像元数(有限值): {finite_in_sx}")
print(f"NaN 像元数: {nan_in_sx}")
if total_in_sx > 0:
    print(f"NaN 占比: {nan_in_sx / total_in_sx * 100:.2f}%")
else:
    print("⚠️ 陕西范围内没有像元（可能裁剪有问题）")
