class Config :
    
    PATH_TRAIN_SET = '/mnt/mmlab2024nas/khanhnq/Dataset/ImageSets/'
    PATH_TEST_SET = '/mnt/mmlab2024nas/khanhnq/Dataset/Test_set/'
    NUM_CLASS_SEG = 4
    NUM_CLASS_CLS = 5
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 1

    # Defaults
    EARLY_STOP = 10
    LAMBDA1 = 1 # augument of segmentation loss
    LAMBDA2 = 0.2 # augument of classification loss
    LAMBDA3 = 0.4 # augument of auxiliary loss

    LEARNING_RATE = 1E-5
    WEIGHT_DECAY = 1E-4
    EPOCHS = 100

    # name of experiment name exp : ex1
    EXPERIMENT_NAME = 'ex1'
    '''Alter with your path'''
    PATH_SAVE_LOG = '/home/khanhnq/Experiment/Mask_RCNN/Experimence/' 
    PATH_SAVE_CKPT = '/mnt/mmlab2024nas/khanhnq/check_point_deeplabv3/'
    
    PATH_DATA_TRAIN = '/mnt/mmlab2024nas/khanhnq/Dataset/'
    PATH_DATA_TEST = '/mnt/mmlab2024nas/khanhnq/Dataset/Test_set/'
    # if using data augmentation
    PATH_DATA_AUGMENTATION = '/mnt/mmlab2024nas/khanhnq/Dataset/data_augment/'