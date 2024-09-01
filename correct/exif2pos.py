import numpy as np

def getSignOf(chifre):
    if chifre >= 0:
        return 1
    else:
        return -1


def hrp2opk(Roll, Pitch, heading):
    Roll = np.deg2rad(Roll)
    Pitch = np.deg2rad(Pitch)
    heading = np.deg2rad(heading)

    A_SINH = np.sin(heading)
    A_SINR = np.sin(Roll)
    A_SINP = np.sin(Pitch)

    A_COSH = np.cos(heading)
    A_COSR = np.cos(Roll)
    A_COSP = np.cos(Pitch)

    MX = np.zeros((3, 3))
    MX[0][0] = (A_COSH * A_COSR) + (A_SINH * A_SINP * A_SINR)
    MX[0][1] = (-A_SINH * A_COSR) + (A_COSH * A_SINP * A_SINR)
    MX[0][2] = -A_COSP * A_SINR

    MX[1][0] = A_SINH * A_COSP
    MX[1][1] = A_COSH * A_COSP
    MX[1][2] = A_SINP

    MX[2][0] = (A_COSH * A_SINR) - (A_SINH * A_SINP * A_COSR)
    MX[2][1] = (-A_SINH * A_SINR) - (A_COSH * A_SINP * A_COSR)
    MX[2][2] = A_COSP * A_COSR

    P = np.zeros((3, 3))
    P[0][0] = MX[0][0]
    P[0][1] = MX[1][0]
    P[0][2] = MX[2][0]

    P[1][0] = MX[0][1]
    P[1][1] = MX[1][1]
    P[1][2] = MX[2][1]

    P[2][0] = MX[2][0]
    P[2][1] = MX[1][2]
    P[2][2] = MX[2][2]

    Omega = 0
    Phi = 0
    Kappa = 0

    Omega = np.arctan(-P[2][1] / P[2][2])
    Phi = np.arcsin(P[2][2])
    Kappa = np.arctan(-P[1][0] / P[0][0])

    Phi = abs(np.arcsin(P[2][0]))
    Phi = Phi * getSignOf(P[2][0])
    Omega = abs(np.arccos((P[2][2] / np.cos(Phi))))
    Omega = Omega * (getSignOf(P[2][1] / P[2][2] * -1))
    Kappa = np.arccos(P[0][0] / np.cos(Phi))

    if getSignOf(P[0][0]) == getSignOf((P[1][0] / P[0][0])):
        Kappa = Kappa * -1

    Omega = np.rad2deg(Omega)
    Phi = np.rad2deg(Phi)
    Kappa = np.rad2deg(Kappa)

    return (Omega, Phi, Kappa)


def Get_Image_angle(file_path):
    """
    :param file_path: 输入图片路径
    :return: 图片的偏航角
    """
    # 获取图片偏航角
    # 定义字节模式 b 和 a，用于查找大疆EXIF数据的起始和结束标记
    b = b"\x3c\x2f\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x3e"
    a = b"\x3c\x72\x64\x66\x3a\x44\x65\x73\x63\x72\x69\x70\x74\x69\x6f\x6e\x20"
    # 打开图片文件，以二进制模式读取
    img = open(file_path, 'rb')
    # 初始化一个字节数组用于存储EXIF数据
    data = bytearray()
    # 初始化一个标志，用于判断是否已经找到EXIF数据的起始标记
    flag = False
    # 逐行读取图片文件内容
    for line in img.readlines():
        # 如果当前行包含EXIF数据的起始标记，则设置标志为True
        if a in line:
            flag = True
            # 如果标志为True，则将当前行添加到EXIF数据中
        if flag:
            data += line
            # 如果当前行包含EXIF数据的结束标记，则跳出循环
        if b in line:
            break
            # 如果提取到的EXIF数据不为空
    dict = {}
    # 遍历过滤后的行，并提取键值对存入字典中
    if len(data) > 0:
        # 将字节数据解码为ASCII字符串
        data = str(data.decode('ascii'))
        # 过滤出包含drone-dji的行，并分割每行为键值对
        lines = list(filter(lambda x: 'dji:' in x, data.split("\n")))
        # 初始化一个空字典用于存储提取到的数据
        for d in lines:
            d = d.strip()[10:]  # 去除每行的前后空格和'\n'字符，并从第10个字符开始处理（因为drone-dji:占据了前9个字符）
            k, v = d.split("=")  # 将当前行分割为键和值两部分
            dict[k] = v.replace('"', '')  # 将键值对存入字典中

    RtkFlag = int(dict.get('RtkFlag'))

    # 检查RtkFlag的值是否等于50(固定解）
    if RtkFlag != 50:
        # 如果不等于50，则报告错误并终止代码执行
        error_message = "RtkFlag的值为{}，非固定解".format(RtkFlag)
        print(error_message)

    B = dict.get('GpsLatitude')
    L = dict.get('GpsLongitude') or dict.get('GpsLongtitude')
    H = dict.get('AbsoluteAltitude')
    Pitch = float(dict.get('GimbalPitchDegree'))+90
    Yaw = float(dict.get('GimbalYawDegree'))
    Roll = float(dict.get('GimbalRollDegree'))

    #hrp2opk角度制
    omega, phi, kappa = hrp2opk(Roll, Pitch, Yaw)
    print(omega)
    print(phi)

    omega_rad = np.deg2rad(omega)
    phi_rad = np.deg2rad(phi)
    kappa_rad = np.deg2rad(-kappa)

    return [B,L,H,omega_rad, phi_rad,kappa_rad ]   # 返回主要信息的值。如果未找到，则返回None。
