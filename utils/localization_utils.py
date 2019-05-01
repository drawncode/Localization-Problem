from utils.model_1 import *

def eval_class(data):
    if data[5]=='knuckle':
        class_num = np.array([1,0,0])
    elif data[5]=='Palm':
        class_num = np.array([0,1,0])
    elif data[5]=='veins':
        class_num= np.array([0,0,1])
    return class_num

def reshape_insert(gt,base_path,X,Y_class,Y_regress,re_size):
    for i,data in enumerate(gt):
        try:
            img = cv2.imread(base_path+data[0])
            height,width = img.shape[:2]
            img = cv2.resize(img,re_size)
            X.append(img)
            w = (float(data[3])-float(data[1]))/width
            h = (float(data[4])-float(data[2]))/height
            c_x = (float(data[1]))/width+w/2.0
            c_y = (float(data[2]))/height+h/2.0
            Y_class.append(eval_class(data))
            Y_regress.append(np.array([c_x,c_y,w,h]))
        except:
            pass
    
def load_data(data_path):
    print("Loading the data........")
    X=[]
    Y_class=[]
    Y_regress  =[]
    for i in range(3):
        f = open(data_path[i]+"groundtruth.txt", "r")
        gt = []
        for line in f:
            line = line[:-1]
            gt.append(line.split(','))
        reshape_insert(gt,data_path[i],X,Y_class,Y_regress,(250,250))
    print("Data Loaded successfully")
    return X,Y_class,Y_regress


def reshape_insert_test(names,gt,base_path,X,X_orig,Y_class,Y_regress,re_size):
    for i,data in enumerate(gt):
        try:
            name = data[0]
#             print(names)
            names.append(name[5:])
            img = cv2.imread(base_path+name)
            height,width = img.shape[:2]
            X_orig.append(img)
            img = cv2.resize(img,re_size)
            X.append(img)
            w = (float(data[3])-float(data[1]))/width
            h = (float(data[4])-float(data[2]))/height
            c_x = (float(data[1]))/width+w/2.0
            c_y = (float(data[2]))/height+h/2.0
            Y_class.append(eval_class(data))
            Y_regress.append(np.array([c_x,c_y,w,h]))
        except:
            pass
    
def load_data_test(data_path):
    print("Loading the data........")
    X=[]
    names = []
    X_orig = []
    Y_class=[]
    Y_regress  =[]
    for i in range(3):
        f = open(data_path[i]+"groundtruth.txt", "r")
        gt = []
        for line in f:
            line = line[:-1]
            gt.append(line.split(','))
        reshape_insert_test(names,gt,data_path[i],X,X_orig,Y_class,Y_regress,(250,250))
    print("Data Loaded successfully")
    return names,X,X_orig,Y_class,Y_regress

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)