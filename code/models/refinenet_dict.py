# ==============================================================================
# refinenet_dict.py —— 时空交互精化网络(R-CLSTM 系列)注册表
# ------------------------------------------------------------------------------
# 把 R_CLSTM_modules.py 中实现的各版本 R-CLSTM 类注册成字典，
# 供 net.py 按 --refinenet 字符串选取粗网络内部使用的时空交互模块。
# 注：只有 R_CLSTM_5 与 R_CLSTM_6 在当前代码中被实际启用，其余被注释掉。
# ==============================================================================
# from models.R_CLSTM_modules import (R_CLSTM_1, R_CLSTM_2, R_CLSTM_3,
#                                             R_CLSTM_4, R_CLSTM_5, R_CLSTM_6,
#                                             R_CLSTM_7, R_CLSTM_8, R_CLSTM_9)

from models.R_CLSTM_modules import R_CLSTM_5, R_CLSTM_6

refinenet_dict = {
    # 'R_CLSTM_1': R_CLSTM_1,
    # 'R_CLSTM_2': R_CLSTM_2,
    # 'R_CLSTM_3': R_CLSTM_3,
    # 'R_CLSTM_4': R_CLSTM_4,
    'R_CLSTM_5': R_CLSTM_5,
    'R_CLSTM_6': R_CLSTM_6,
    # 'R_CLSTM_7': R_CLSTM_7,
    # 'R_CLSTM_8': R_CLSTM_8,
    # 'R_CLSTM_9': R_CLSTM_9
}