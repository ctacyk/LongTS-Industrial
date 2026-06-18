"""
EngineMT-QA 传感器分组定义
基于NASA C-MAPSS发动机物理架构
"""

# 33个传感器通道（按原始顺序）
SENSOR_NAMES = [
    'alt',      # 0: Altitude - 飞行高度
    'Mach',     # 1: Mach number - 马赫数
    'TRA',      # 2: Throttle Resolver Angle - 油门角度
    'T2',       # 3: Fan inlet temperature
    'T24',      # 4: LPC outlet temperature
    'T30',      # 5: HPC outlet temperature
    'T40',      # 6: HPT inlet temperature (combustor exit)
    'T48',      # 7: HPT outlet temperature
    'T50',      # 8: LPT outlet temperature
    'P15',      # 9: Total pressure in bypass duct
    'P2',       # 10: Fan inlet total pressure
    'P21',      # 11: Fan outlet total pressure
    'P24',      # 12: LPC outlet total pressure
    'Ps30',     # 13: HPC outlet static pressure
    'P40',      # 14: Combustor pressure
    'P50',      # 15: LPT outlet pressure
    'Nf',       # 16: Physical fan speed (low spool)
    'Nc',       # 17: Physical core speed (high spool)
    'Wf',       # 18: Fuel flow
    'T40',      # 19: HPT inlet temp (duplicate?)
    'P30',      # 20: HPC outlet total pressure
    'P45',      # 21: HPT outlet pressure
    'W21',      # 22: Fan exit mass flow
    'W22',      # 23: Bypass mass flow
    'W25',      # 24: Bleed mass flow
    'W31',      # 25: HPT cooling bleed
    'W32',      # 26: LPT cooling bleed
    'W41',      # 27: HPT exit mass flow
    'W50',      # 28: LPT exit mass flow
    'SmFan',    # 29: Fan surge margin
    'SmLPC',    # 30: LPC surge margin
    'SmHPC',    # 31: HPC surge margin
    'phi',      # 32: Fuel-air ratio
]

# 按物理子系统分组
SENSOR_GROUPS = {
    'Operating Conditions': {
        'indices': [0, 1, 2],
        'names': ['alt', 'Mach', 'TRA'],
        'description': 'Flight conditions and throttle',
        'color': '#1f77b4'  # 蓝色
    },
    'Temperatures': {
        'indices': [3, 4, 5, 6, 7, 8],
        'names': ['T2', 'T24', 'T30', 'T40', 'T48', 'T50'],
        'description': 'Temperature sensors along gas path',
        'color': '#d62728'  # 红色
    },
    'Pressures': {
        'indices': [9, 10, 11, 12, 13, 14, 15, 20, 21],
        'names': ['P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'P30', 'P45'],
        'description': 'Pressure sensors',
        'color': '#2ca02c'  # 绿色
    },
    'Rotational Speeds': {
        'indices': [16, 17],
        'names': ['Nf', 'Nc'],
        'description': 'Fan and core shaft speeds',
        'color': '#ff7f0e'  # 橙色
    },
    'Fuel & Flows': {
        'indices': [18, 22, 23, 24, 25, 26, 27, 28],
        'names': ['Wf', 'W21', 'W22', 'W25', 'W31', 'W32', 'W41', 'W50'],
        'description': 'Fuel flow and mass flow rates',
        'color': '#9467bd'  # 紫色
    },
    'Health Indicators': {
        'indices': [29, 30, 31, 32],
        'names': ['SmFan', 'SmLPC', 'SmHPC', 'phi'],
        'description': 'Surge margins and fuel-air ratio',
        'color': '#8c564b'  # 棕色
    }
}

if __name__ == '__main__':
    print("Sensor Groups for EngineMT-QA Dataset")
    print("=" * 50)
    total = 0
    for group_name, group_info in SENSOR_GROUPS.items():
        print(f"\n{group_name}:")
        print(f"  Indices: {group_info['indices']}")
        print(f"  Sensors: {group_info['names']}")
        print(f"  Count: {len(group_info['indices'])}")
        total += len(group_info['indices'])
    print(f"\nTotal sensors: {total}")
