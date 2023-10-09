#! /bin/bash

# python -u train_fineturnModelScoreLossPlus.py --level 5 --level 10 --level 15 --level 20 --dataset SFD_augment_add_SFD
# python -u train_fineturnModelScoreLossPlus.py --level 5 --level 10 --level 15 --level 20 --dataset AUCDD_V1_and_SFD_augment
# python -u train_fineturnModelScoreLossPlus.py --level 5 --level 10 --level 15 --level 20 --dataset Driver100Mini_Tradition_Day_Cam4_and_SFD_augment
# python -u train_fineturnModelScoreLossPlus.py --level 5 --level 10 --level 15 --level 20 --dataset Driver100Mini_Tradition_Day_Cam4_and_AUCDDV1_augment

#--------------------------------Cross_Camera_Setting----------------------------
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Camera_Setting_D1
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Camera_Setting_D2
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Camera_Setting_D3
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Camera_Setting_D4

#--------------------------------Cross_Individual_Vehicle----------------------------
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Individual_Vehicle_D1_MAZDA
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Individual_Vehicle_D2_MAZDA
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Individual_Vehicle_D3_MAZDA
python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Individual_Vehicle_D4_MAZDA

#--------------------------------Cross_Vehicle_Type----------------------------
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Vehicle_Type_D1_Sedan
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Vehicle_Type_D2_Sedan
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Vehicle_Type_D3_Sedan
# python -u train_fineturnModelScoreLossPlus.py --level 15 --dataset Driver100_Cross_Vehicle_Type_D4_Sedan
