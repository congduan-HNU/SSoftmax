'''
Author: Cong Duan
Date: 2023-08-02 19:34:19
LastEditTime: 2023-09-13 21:14:18
LastEditors: your name
Description: Driver100 原数据集
FilePath: /Driver-Action-Monitor/trainAcrossDatasets/datasetDriver100.py
可以输入预定的版权声明、个性签名、空行等
'''
import os
import sys
import copy
sys.path.append("..")
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)
from pythonUtils import osp, printPlus, projectInfo
computer_name = os.popen("hostname").read()




Server2080ti = "/user/duancong/DC/datasets/Dataset"
Server4090 = "/home/caomen/Desktop/DC/Dataset"
PC3060 = "/home/duancong/FileFolder/Datasets"

if computer_name.strip() == 'caomen-Z390-AORUS-ULTRA':
    OSROOTDIR = Server2080ti
elif computer_name.strip() == 'caomen-MS-4090':
    OSROOTDIR = Server4090
elif computer_name.strip() == 'duancong-MS-7C94':
    OSROOTDIR = PC3060
   
class Driver100CrossDataset(object):
    def __init__(self, root):
        self.D1 = os.path.join(OSROOTDIR, root, 'Day/Cam1_size224')
        self.D2 = os.path.join(OSROOTDIR, root, 'Day/Cam2_size224')
        self.D3 = os.path.join(OSROOTDIR, root, 'Day/Cam3_size224')
        self.D4 = os.path.join(OSROOTDIR, root, 'Day/Cam4_size224')
        self.D1_A = os.path.join(OSROOTDIR, root, 'Day/Cam1_Augment')
        self.D2_A = os.path.join(OSROOTDIR, root, 'Day/Cam2_Augment')
        self.D3_A = os.path.join(OSROOTDIR, root, 'Day/Cam3_Augment')
        self.D4_A = os.path.join(OSROOTDIR, root, 'Day/Cam4_Augment')
        self.N1 = os.path.join(OSROOTDIR, root, 'Night/Cam1_size224')
        self.N2 = os.path.join(OSROOTDIR, root, 'Night/Cam2_size224')
        self.N3 = os.path.join(OSROOTDIR, root, 'Night/Cam3_size224')
        self.N4 = os.path.join(OSROOTDIR, root, 'Night/Cam4_size224')
        self.N1_A = os.path.join(OSROOTDIR, root, 'Night/Cam1_Augment')
        self.N2_A = os.path.join(OSROOTDIR, root, 'Night/Cam2_Augment')
        self.N3_A = os.path.join(OSROOTDIR, root, 'Night/Cam3_Augment')
        self.N4_A = os.path.join(OSROOTDIR, root, 'Night/Cam4_Augment')


# # Cross-camera-setting D1 to D2 D3 D4
# self.CCS_D1_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam1_to_2_3_4/Cam1_train.txt')
# self.CCS_D1_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-camera-setting/Day/Cam1_to_2_3_4/Cam1_train_Augment.txt')
# self.CCS_D1_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam1_to_2_3_4/Cam1_val.txt')
# self.CCS_D1_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam1_to_2_3_4/Cam1_test.txt')

# # Cross-camera-setting D2 to D1 D3 D4
# self.CCS_D2_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam2_to_1_3_4/Cam2_train.txt')
# self.CCS_D2_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-camera-setting/Day/Cam2_to_1_3_4/Cam2_train_Augment.txt')
# self.CCS_D2_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam2_to_1_3_4/Cam2_val.txt')
# self.CCS_D2_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam2_to_1_3_4/Cam2_test.txt')

# # Cross-camera-setting D3 to D1 D2 D4
# self.CCS_D3_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam3_to_1_2_4/Cam3_train.txt')
# self.CCS_D3_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-camera-setting/Day/Cam3_to_1_2_4/Cam3_train_Augment.txt')
# self.CCS_D3_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam3_to_1_2_4/Cam3_val.txt')
# self.CCS_D3_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam3_to_1_2_4/Cam3_test.txt')

# # Cross-camera-setting D4 to D1 D2 D3
# self.CCS_D4_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam4_to_1_2_3/Cam4_train.txt')
# self.CCS_D4_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-camera-setting/Day/Cam4_to_1_2_3/Cam4_train_Augment.txt')
# self.CCS_D4_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam4_to_1_2_3/Cam4_val.txt')
# self.CCS_D4_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-camera-setting/Day/Cam4_to_1_2_3/Cam4_test.txt')


# # Cross-modality-setting D1 to N1
# self.CMS_D1_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D1_to_N1/D1_train.txt')
# self.CMS_D1_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D1_to_N1/D1_train_Augment.txt')
# self.CMS_D1_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D1_to_N1/D1_val.txt')
# self.CMS_N1_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D1_to_N1/N1_train.txt')
# self.CMS_N1_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D1_to_N1/N1_train_Augment.txt')
# self.CMS_N1_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D1_to_N1/N1_test.txt')

# # Cross-modality-setting D2 to N2
# self.CMS_D2_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D2_to_N2/D2_train.txt')
# self.CMS_D2_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D2_to_N2/D2_train_Augment.txt')
# self.CMS_D2_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D2_to_N2/D2_val.txt')
# self.CMS_N2_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D2_to_N2/N2_train.txt')
# self.CMS_N2_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D2_to_N2/N2_train_Augment.txt')
# self.CMS_N2_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D2_to_N2/N2_test.txt')

# # Cross-modality-setting D3 to N3
# self.CMS_D3_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D3_to_N3/D3_train.txt')
# self.CMS_D3_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D3_to_N3/D3_train_Augment.txt')
# self.CMS_D3_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D3_to_N3/D3_val.txt')
# self.CMS_N3_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D3_to_N3/N3_train.txt')
# self.CMS_N3_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D3_to_N3/N3_train_Augment.txt')
# self.CMS_N3_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D3_to_N3/N3_test.txt')

# # Cross-modality-setting D4 to N4
# self.CMS_D4_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D4_to_N4/D4_train.txt')
# self.CMS_D4_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D4_to_N4/D4_train_Augment.txt')
# self.CMS_D4_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D4_to_N4/D4_val.txt')
# self.CMS_N4_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D4_to_N4/N4_train.txt')
# self.CMS_N4_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-modality-setting/D4_to_N4/N4_train_Augment.txt')
# self.CMS_N4_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-modality-setting/D4_to_N4/N4_test.txt')
class Driver100CrossLabel(object):
    def __init__(self, root=osp.join(projectInfo.ROOT, 'trainAcrossDatasets/groundtruth/100Driver')):
        CrossTypeDict = ["Cross-camera-setting",
                         "Cross-modality-setting",
                         "Cross-individual-vehicle",
                         "Cross-individual-vehicle",
                         ]
        CamS = ['Cam1', 'Cam2', 'Cam3', 'Cam4']
        times = ['Day', 'Night']

        # Cross-camera-setting D1 to D2 D3 D4 ...
        for crossKind in ["Cross-camera-setting"]:
            crossSimple = crossKind.split('-')[0][0].upper()+ crossKind.split('-')[1][0].upper()+ crossKind.split('-')[2][0].upper()
            for time in times:
                for view in copy.deepcopy(CamS):
                    left_cams = copy.deepcopy(CamS)
                    left_cams.remove(view)
                    for subset in ['Train', 'Train_A', 'Val', 'Test']:
                        # print(f"{crossSimple}_{time[0]}{view[3]}_{subset}")
                        addStr = '_Augment' if '_A' in subset else ''
                        subset_str = subset.split('_')[0]
                        self.__setattr__(f"{crossSimple}_{time[0]}{view[3]}_{subset}", os.path.join(root+addStr[1:], crossKind, time, f'{view}_to_{left_cams[0][3]}_{left_cams[1][3]}_{left_cams[2][3]}', f'{view}_{subset_str.lower()}{addStr}.txt'))


        # Cross-camera-setting D1 to D2 D3 D4 ...
        for crossKind in ["Cross-modality-setting"]:
            crossSimple = crossKind.split('-')[0][0].upper()+ crossKind.split('-')[1][0].upper()+ crossKind.split('-')[2][0].upper()
            for time in times:
                left_time = copy.deepcopy(times)
                left_time.remove(time)
                for view in copy.deepcopy(CamS):
                    for subset in ['Train', 'Train_A', 'Val', 'Test']:
                        if time[0] == 'D' and subset == 'Test':
                            continue
                        if time[0] == 'N' and subset == 'Val':
                            continue
                        # print(f"{crossSimple}_{time[0]}{view[3]}_{subset}")
                        addStr = '_Augment' if '_A' in subset else ''
                        subset_str = subset.split('_')[0]
                        self.__setattr__(f"{crossSimple}_{time[0]}{view[3]}_{subset}", os.path.join(root+addStr[1:], crossKind, f'{times[0][0]}{view[3]}_to_{times[1][0]}{view[3]}', f'{time[0]}{view[3]}_{subset_str.lower()}{addStr}.txt'))


        # Cross-individual-vehicle train on mazda
        # self.CIV_D1_Mazda_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/mazda_train.txt')
        # self.CIV_D1_Mazda_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/mazda_train_Augment.txt')
        # self.CIV_D1_Mazda_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/mazda_test.txt')
        # self.CIV_D1_Ankai_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/ankai.txt')
        # self.CIV_D1_Hyundai_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/hyundai.txt')
        # self.CIV_D1_Lynk_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-individual-vehicle/Day/Cam1/lynk.txt')

        for i in range(1, 5):
            self.__setattr__(f"CIV_D{i}_Mazda_Train", os.path.join(root, 'Cross-vehicle-setting/Cross-individual-vehicle/Day', f'Cam{i}', 'mazda_train.txt'))
            self.__setattr__(
                f"CIV_D{i}_Mazda_Train_A",
                os.path.join(
                    f'{root}Augment',
                    'Cross-vehicle-setting/Cross-individual-vehicle/Day',
                    f'Cam{i}',
                    'mazda_train_Augment.txt',
                ),
            )
            self.__setattr__(f"CIV_D{i}_Mazda_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-individual-vehicle/Day', f'Cam{i}', 'mazda_test.txt'))
            self.__setattr__(f"CIV_D{i}_Ankai_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-individual-vehicle/Day', f'Cam{i}', 'ankai.txt'))
            self.__setattr__(f"CIV_D{i}_Hyundai_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-individual-vehicle/Day', f'Cam{i}', 'hyundai.txt'))
            self.__setattr__(f"CIV_D{i}_Lynk_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-individual-vehicle/Day', f'Cam{i}', 'lynk.txt'))

        # Cross-vehicle-type train on sedan
        # self.CVT_D1_Sedan_Train = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-vehicle-type/Cam1/sedan_train.txt')
        # self.CVT_D1_Sedan_Train_A = os.path.join(OSROOTDIR, root, '100DriverAugment/Cross-vehicle-setting/Cross-vehicle-type/Cam1/sedan_train_Augment.txt')
        # self.CVT_D1_Sedan_Val = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-vehicle-type/Cam1/sedan_val.txt')
        # self.CVT_D1_SUV_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-vehicle-type/Cam1/SUV.txt')
        # self.CVT_D1_Van_Test = os.path.join(OSROOTDIR, root, '100Driver/Cross-vehicle-setting/Cross-vehicle-type/Cam1/Van.txt')

        for i in range(1, 5):
            self.__setattr__(f"CVT_D{i}_Sedan_Train", os.path.join(root, 'Cross-vehicle-setting/Cross-vehicle-type', f'Cam{i}', 'sedan_train.txt'))
            self.__setattr__(
                f"CVT_D{i}_Sedan_Train_A",
                os.path.join(
                    f'{root}Augment',
                    'Cross-vehicle-setting/Cross-vehicle-type',
                    f'Cam{i}',
                    'sedan_train_Augment.txt',
                ),
            )
            self.__setattr__(f"CVT_D{i}_Sedan_Val", os.path.join(root, 'Cross-vehicle-setting/Cross-vehicle-type', f'Cam{i}', 'sedan_val.txt'))
            self.__setattr__(f"CVT_D{i}_SUV_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-vehicle-type', f'Cam{i}', 'SUV.txt'))
            self.__setattr__(f"CVT_D{i}_Van_Test", os.path.join(root, 'Cross-vehicle-setting/Cross-vehicle-type', f'Cam{i}', 'Van.txt'))

    def check(self):
        for key, value in self.__dict__.items():
            if osp.exists(value):
                printPlus(f"Check Success: {key}: {value}", frontColor=32)
            else:
                printPlus(f"Check Failed: {key}: {value}", frontColor=31)
        
             
Driver100 = Driver100CrossDataset('100Driver')
Label = Driver100CrossLabel()
Label.check()

Driver100_Tradition_Setting = {
    "DataName": "SFD_augment",
    "modal": "rgb",
    "ImageTrainPath": f"{OSROOTDIR}/SFD_Augment",
    "TrainLabelPath": r"trainAcrossDatasets/groundtruth/SFD/trainLabel(224)_augment.txt",
    "ImageValPath": f"{OSROOTDIR}/SFD",
    "ValLabelPath": r"trainAcrossDatasets/groundtruth/SFD/testLabel(224).txt",
    "ImageTestPath": f"{OSROOTDIR}/SFD",
    "TestLabelPath": r"trainAcrossDatasets/groundtruth/SFD/testLabel(224).txt",
    "classes": 10,
    "class_names": None,
    "info": ["SFD augment dataset"],
}

Driver100_Cross_Camera_Setting_D1 = {
    "DataName": "Driver100_Cross_Camera_Setting_D1",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D1, Driver100.D1_A],
    "TrainLabelPath": [Label.CCS_D1_Train, Label.CCS_D1_Train_A],
    "ImageValPath": [Driver100.D1],
    "ValLabelPath": [Label.CCS_D1_Val],
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CCS_D1_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting Train:D1 Val:D1 Test:D1"],
}
Driver100_Cross_Camera_Setting_N1 = {
    "DataName": "Driver100_Cross_Camera_Setting_N1",
    "modal": "rgb",
    "ImageTestPath": Driver100.N1,
    "TestLabelPath": Label.CCS_N1_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting_N1 Train:D1 Val:D1 Test:D1"],
}

Driver100_Cross_Camera_Setting_D2 = {
    "DataName": "Driver100_Cross_Camera_Setting_D2",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D2, Driver100.D2_A],
    "TrainLabelPath": [Label.CCS_D2_Train, Label.CCS_D2_Train_A],
    "ImageValPath": [Driver100.D2],
    "ValLabelPath": [Label.CCS_D2_Val],
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CCS_D2_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting Train:D2 Val:D2 Test:D2"],
}
Driver100_Cross_Camera_Setting_N2 = {
    "DataName": "Driver100_Cross_Camera_Setting_N2",
    "modal": "rgb",
    "ImageTestPath": Driver100.N2,
    "TestLabelPath": Label.CCS_N2_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting_N2 Train:N2 Val:N2 Test:N2"],
}

Driver100_Cross_Camera_Setting_D3 = {
    "DataName": "Driver100_Cross_Camera_Setting_D3",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D3, Driver100.D3_A],
    "TrainLabelPath": [Label.CCS_D3_Train, Label.CCS_D3_Train_A],
    "ImageValPath": [Driver100.D3],
    "ValLabelPath": [Label.CCS_D3_Val],
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CCS_D3_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting Train:D3 Val:D3 Test:D3"],
}
Driver100_Cross_Camera_Setting_N3 = {
    "DataName": "Driver100_Cross_Camera_Setting_N3",
    "modal": "rgb",
    "ImageTestPath": Driver100.N3,
    "TestLabelPath": Label.CCS_N3_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting_N3 Train:N3 Val:N3 Test:N3"],
}

Driver100_Cross_Camera_Setting_D4 = {
    "DataName": "Driver100_Cross_Camera_Setting_D4",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D4, Driver100.D4_A],
    "TrainLabelPath": [Label.CCS_D4_Train, Label.CCS_D4_Train_A],
    "ImageValPath": [Driver100.D4],
    "ValLabelPath": [Label.CCS_D4_Val],
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CCS_D4_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting Train:D4 Val:D4 Test:D4"],
}
Driver100_Cross_Camera_Setting_N4 = {
    "DataName": "Driver100_Cross_Camera_Setting_N4",
    "modal": "rgb",
    "ImageTestPath": Driver100.N4,
    "TestLabelPath": Label.CCS_N4_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Camera_Setting_N4 Train:N4 Val:N4 Test:N4"],
}


Driver100_Cross_Individual_Vehicle_D1_MAZDA = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D1_MAZDA",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D1, Driver100.D1_A],
    "TrainLabelPath": [Label.CIV_D1_Mazda_Train, Label.CIV_D1_Mazda_Train_A],
    "ImageValPath": [Driver100.D1],
    "ValLabelPath": [Label.CIV_D1_Mazda_Test],
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CIV_D1_Mazda_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D1_MAZDA Train:Mazda_Train Val:Mazda_Val Test:Mazda_Test"],
}
Driver100_Cross_Individual_Vehicle_D1_Ankai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D1_Ankai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CIV_D1_Ankai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D1_Ankai_Test Train:Mazda_Train Val:Mazda_Test Test:Ankai_Test"],
}
Driver100_Cross_Individual_Vehicle_D1_Hyundai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D1_Hyundai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CIV_D1_Hyundai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D1_Hyundai_Test Train:Mazda_Train Val:Mazda_Test Test:Hyundai_Test"],
}
Driver100_Cross_Individual_Vehicle_D1_Lynk_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D1_Lynk_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CIV_D1_Lynk_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D1_Lynk_Test Train:Mazda_Train Val:Mazda_Test Test:Lynk_Test"],
}

Driver100_Cross_Individual_Vehicle_D2_MAZDA = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D2_MAZDA",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D2, Driver100.D2_A],
    "TrainLabelPath": [Label.CIV_D2_Mazda_Train, Label.CIV_D2_Mazda_Train_A],
    "ImageValPath": [Driver100.D2],
    "ValLabelPath": [Label.CIV_D2_Mazda_Test],
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CIV_D2_Mazda_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D2_MAZDA Train:Mazda_Train Val:Mazda_Val Test:Mazda_Test"],
}
Driver100_Cross_Individual_Vehicle_D2_Ankai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D2_Ankai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CIV_D2_Ankai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D2_Ankai_Test Train:Mazda_Train Val:Mazda_Test Test:Ankai_Test"],
}
Driver100_Cross_Individual_Vehicle_D2_Hyundai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D2_Hyundai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CIV_D2_Hyundai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D2_Hyundai_Test Train:Mazda_Train Val:Mazda_Test Test:Hyundai_Test"],
}
Driver100_Cross_Individual_Vehicle_D2_Lynk_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D2_Lynk_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CIV_D2_Lynk_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D2_Lynk_Test Train:Mazda_Train Val:Mazda_Test Test:Lynk_Test"],
}

Driver100_Cross_Individual_Vehicle_D3_MAZDA = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D3_MAZDA",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D3, Driver100.D3_A],
    "TrainLabelPath": [Label.CIV_D3_Mazda_Train, Label.CIV_D3_Mazda_Train_A],
    "ImageValPath": [Driver100.D3],
    "ValLabelPath": [Label.CIV_D3_Mazda_Test],
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CIV_D3_Mazda_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D3_MAZDA Train:Mazda_Train Val:Mazda_Val Test:Mazda_Test"],
}
Driver100_Cross_Individual_Vehicle_D3_Ankai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D3_Ankai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CIV_D3_Ankai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D3_Ankai_Test Train:Mazda_Train Val:Mazda_Test Test:Ankai_Test"],
}
Driver100_Cross_Individual_Vehicle_D3_Hyundai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D3_Hyundai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CIV_D3_Hyundai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D3_Hyundai_Test Train:Mazda_Train Val:Mazda_Test Test:Hyundai_Test"],
}
Driver100_Cross_Individual_Vehicle_D3_Lynk_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D3_Lynk_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CIV_D3_Lynk_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D3_Lynk_Test Train:Mazda_Train Val:Mazda_Test Test:Lynk_Test"],
}

Driver100_Cross_Individual_Vehicle_D4_MAZDA = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D4_MAZDA",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D4, Driver100.D4_A],
    "TrainLabelPath": [Label.CIV_D4_Mazda_Train, Label.CIV_D4_Mazda_Train_A],
    "ImageValPath": [Driver100.D4],
    "ValLabelPath": [Label.CIV_D4_Mazda_Test],
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CIV_D4_Mazda_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D4_MAZDA Train:Mazda_Train Val:Mazda_Val Test:Mazda_Test"],
}
Driver100_Cross_Individual_Vehicle_D4_Ankai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D4_Ankai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CIV_D4_Ankai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D4_Ankai_Test Train:Mazda_Train Val:Mazda_Test Test:Ankai_Test"],
}
Driver100_Cross_Individual_Vehicle_D4_Hyundai_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D4_Hyundai_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CIV_D4_Hyundai_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D4_Hyundai_Test Train:Mazda_Train Val:Mazda_Test Test:Hyundai_Test"],
}
Driver100_Cross_Individual_Vehicle_D4_Lynk_Test = {
    "DataName": "Driver100_Cross_Individual_Vehicle_D4_Lynk_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CIV_D4_Lynk_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Individual_Vehicle_D4_Lynk_Test Train:Mazda_Train Val:Mazda_Test Test:Lynk_Test"],
}


Driver100_Cross_Vehicle_Type_D1_Sedan = {
    "DataName": "Driver100_Cross_Vehicle_Type_D1_Sedan",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D1, Driver100.D1_A],
    "TrainLabelPath": [Label.CVT_D1_Sedan_Train, Label.CVT_D1_Sedan_Train_A],
    "ImageValPath": [Driver100.D1],
    "ValLabelPath": [Label.CVT_D1_Sedan_Val],
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CVT_D1_Sedan_Val,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D1_Sedan Train:Sedan_Train Val:Sedan_Val Test:Sedan_Test"],
}
Driver100_Cross_Vehicle_Type_D1_SUV_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D1_SUV_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CVT_D1_SUV_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D1_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:SUV_Test"],
}
Driver100_Cross_Vehicle_Type_D1_Van_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D1_Van_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D1,
    "TestLabelPath": Label.CVT_D1_Van_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D1_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:Van_Test"],
}

Driver100_Cross_Vehicle_Type_D2_Sedan = {
    "DataName": "Driver100_Cross_Vehicle_Type_D2_Sedan",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D2, Driver100.D2_A],
    "TrainLabelPath": [Label.CVT_D2_Sedan_Train, Label.CVT_D2_Sedan_Train_A],
    "ImageValPath": [Driver100.D2],
    "ValLabelPath": [Label.CVT_D2_Sedan_Val],
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CVT_D2_Sedan_Val,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D2_Sedan Train:Sedan_Train Val:Sedan_Val Test:Sedan_Test"],
}
Driver100_Cross_Vehicle_Type_D2_SUV_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D2_SUV_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CVT_D2_SUV_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D2_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:SUV_Test"],
}
Driver100_Cross_Vehicle_Type_D2_Van_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D2_Van_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D2,
    "TestLabelPath": Label.CVT_D2_Van_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D2_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:Van_Test"],
}

Driver100_Cross_Vehicle_Type_D3_Sedan = {
    "DataName": "Driver100_Cross_Vehicle_Type_D3_Sedan",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D3, Driver100.D3_A],
    "TrainLabelPath": [Label.CVT_D3_Sedan_Train, Label.CVT_D3_Sedan_Train_A],
    "ImageValPath": [Driver100.D3],
    "ValLabelPath": [Label.CVT_D3_Sedan_Val],
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CVT_D3_Sedan_Val,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D3_Sedan Train:Sedan_Train Val:Sedan_Val Test:Sedan_Test"],
}
Driver100_Cross_Vehicle_Type_D3_SUV_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D3_SUV_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CVT_D3_SUV_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D3_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:SUV_Test"],
}
Driver100_Cross_Vehicle_Type_D3_Van_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D3_Van_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D3,
    "TestLabelPath": Label.CVT_D3_Van_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D3_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:Van_Test"],
}

Driver100_Cross_Vehicle_Type_D4_Sedan = {
    "DataName": "Driver100_Cross_Vehicle_Type_D4_Sedan",
    "modal": "rgb",
    "ImageTrainPath": [Driver100.D4, Driver100.D4_A],
    "TrainLabelPath": [Label.CVT_D4_Sedan_Train, Label.CVT_D4_Sedan_Train_A],
    "ImageValPath": [Driver100.D4],
    "ValLabelPath": [Label.CVT_D4_Sedan_Val],
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CVT_D4_Sedan_Val,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D4_Sedan Train:Sedan_Train Val:Sedan_Val Test:Sedan_Test"],
}
Driver100_Cross_Vehicle_Type_D4_SUV_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D4_SUV_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CVT_D4_SUV_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D4_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:SUV_Test"],
}
Driver100_Cross_Vehicle_Type_D4_Van_Test = {
    "DataName": "Driver100_Cross_Vehicle_Type_D4_Van_Test",
    "modal": "rgb",
    "ImageTestPath": Driver100.D4,
    "TestLabelPath": Label.CVT_D4_Van_Test,
    "classes": 22,
    "class_names": None,
    "info": ["Driver100_Cross_Vehicle_Type_D4_SUV_Test Train:Sedan_Train Val:Sedan_Val Test:Van_Test"],
}