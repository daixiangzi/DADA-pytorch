class Config(object):
        data_dir = './data/cifar-10-batches-py'#train data and test data dir
        gpu_id = '0,1' #gpu id
        train_batch_size=100
        test_batch_size=500 
        G_epochs = 200 # train G epochs
        epochs = 700
        count = 400
        save_img = './save_img_G'+str(G_epochs)+"_total"+str(epochs)+"_"+str(count)+"per_class/"
        
        lr = 0.0003 #learning rate
        fre_print=1 # print frequency
        seed = 1
        workers=4
        num_classes=10 #class num
        logs = './logs/'+str(count)+" per_class" #logs record dir
