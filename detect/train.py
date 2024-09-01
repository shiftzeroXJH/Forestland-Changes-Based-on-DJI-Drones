import datetime
import os
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import xlwt
import keras
from keras.optimizers import adam_v2
adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

import numpy as np

# from unet.loss import bce_dice_loss, f_score
from nets.DSIFN import DSIFN

# from nets.DSIFN_training import Generator, LossHistory
from nets.training import Generator, LossHistory
from nets.Loss import CE, Focal_Loss, dice_loss_with_CE, dice_loss_with_Focal_Loss

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 定位cuda设备为GPU0（需要根据情况修稿）
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 最大可申请显存比例
# config.gpu_options.allow_growth = True  # 允许动态分配显存
# sess = tf.compat.v1.Session(config=config)



if __name__ == "__main__":
    #  训练好的权值保存在logs文件夹里面
    model_log_path = "logs/"

    dataset_path = "D:\GISproject\MyProject\out"
    mode = None

    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    dice_loss = True
    #   是否使用focal loss来防止正负样本不平衡
    focal_loss = True
    cls_weights = np.ones([3], np.float32)

    # 输入图片的大小
    input_size = [512, 512, 3]
    # 打开数据集的txt
    with open(os.path.join(dataset_path, "train.txt")) as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(os.path.join(dataset_path, "val.txt"), "r") as f:
        val_lines = f.readlines()

    #  训练数据数目
    train_num = len(train_lines)
    #  验证数据数目
    validation_num = len(val_lines)

    # 获取model
    model = DSIFN(input_size)
    # model.load_weights(r"D:\pyproject\unet\logs\cd_0517_01.h5", by_name=True, skip_mismatch=True)
    # 打印模型结构
    model.summary()

    model_path = 'logs/cd_0821_01.h5'

    #  回调函数
    model_checkpoint = ModelCheckpoint(model_path,
                                       monitor='loss',
                                       verbose=1,  # 日志显示模式:0->安静模式,1->进度条,2->每轮一行
                                       save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1)

    if focal_loss:
        if dice_loss:
            loss = dice_loss_with_Focal_Loss(cls_weights)
        else:
            loss = Focal_Loss(cls_weights)
    else:
        if dice_loss:
            loss = dice_loss_with_CE(cls_weights)
        else:
            loss = CE(cls_weights)

    if True:
        lr = 1e-4

        epochs = 50
        Batch_size = 2

        model.compile(
            # loss=loss,
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])
        # metrics=['binary_accuracy'])

        gen = Generator(input_size, dataset_path, Batch_size, train_lines, mode)
        # gen = gen.generate()
        gen_val = Generator(input_size, dataset_path, Batch_size, val_lines, mode)
        # gen_val = gen_val.generate()

        epoch_size = train_num // Batch_size
        epoch_size_val = validation_num // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(train_num,
                                                                                   validation_num,
                                                                                   Batch_size))

        # ---------------------------------------------------------------------- #
        # 原训练过程
        # model.fit_generator(gen,
        #                     steps_per_epoch=epoch_size,
        #                     validation_data=gen_val,
        #                     validation_steps=epoch_size_val,
        #                     epochs=Freeze_epoch,
        #                     initial_epoch=Init_epoch,
        #                     callbacks=[checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history])
        # ---------------------------------------------------------------------- #
        #  获取当前时间
        start_time = datetime.datetime.now()
        # 新训练过程
        history = model.fit(gen.generate(),
                            steps_per_epoch=epoch_size,
                            validation_data=gen_val.generate(),
                            validation_steps=epoch_size_val,
                            epochs=epochs,
                            # initial_epoch=Init_epoch,
                            callbacks=[model_checkpoint, early_stopping]
                            # [checkpoint_period, reduce_lr, early_stopping, tensorboard, loss_history]
                            )
        #  训练总时间
        end_time = datetime.datetime.now()
        log_time = "训练总时间: " + str((end_time - start_time).seconds / 60) + "m"
        print(log_time)
        with open('TrainTime.txt', 'w') as f:
            f.write(log_time)
        #  保存并绘制loss,acc
        acc = history.history['output_512_accuracy']
        val_acc = history.history['val_output_512_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        sheet = book.add_sheet('test', cell_overwrite_ok=True)
        for i in range(len(acc)):
            sheet.write(i, 0, str(acc[i]))
            sheet.write(i, 1, val_acc[i])
            sheet.write(i, 2, loss[i])
            sheet.write(i, 3, val_loss[i])
        book.save(r'AccAndLoss.xls')
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'r', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.savefig("accuracy.png", dpi=300)
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.savefig("loss.png", dpi=300)
