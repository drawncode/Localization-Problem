from utils.model_2 import *

def append_tool(Y,data):
    width = 1672
    height = 1572
    w = (float(data[3])-float(data[1]))/width
    h = (float(data[2])-float(data[0]))/height
    c_x = (float(data[1]))/width+w/2.0
    c_y = (float(data[0]))/height+h/2.0
    Y.append(np.array([c_x,c_y,w,h]))

def convert_and_append(images,bboxes,Y_1,Y_2,Y_3,Y_4,X):
    for i in range(len(images)):
        X.append(images[i])
        append_tool(Y_1,[int(bboxes[i].bounding_boxes[0].y1),int(bboxes[i].bounding_boxes[0].x1),int(bboxes[i].bounding_boxes[0].y2),int(bboxes[i].bounding_boxes[0].x2)])
        append_tool(Y_2,[int(bboxes[i].bounding_boxes[1].y1),int(bboxes[i].bounding_boxes[1].x1),int(bboxes[i].bounding_boxes[1].y2),int(bboxes[i].bounding_boxes[1].x2)])
        append_tool(Y_3,[int(bboxes[i].bounding_boxes[2].y1),int(bboxes[i].bounding_boxes[2].x1),int(bboxes[i].bounding_boxes[2].y2),int(bboxes[i].bounding_boxes[2].x2)])
        append_tool(Y_4,[int(bboxes[i].bounding_boxes[3].y1),int(bboxes[i].bounding_boxes[3].x1),int(bboxes[i].bounding_boxes[3].y2),int(bboxes[i].bounding_boxes[3].x2)])

def augment_tool(Y_1,Y_2,Y_3,Y_4,img):
    bbs = ia.BoundingBoxesOnImage([
    ia.BoundingBox(x1=Y_1[1], x2=Y_1[3], y1=Y_1[0], y2=Y_1[2]),
    ia.BoundingBox(x1=Y_2[1], x2=Y_2[3], y1=Y_2[0], y2=Y_2[2]),
    ia.BoundingBox(x1=Y_3[1], x2=Y_3[3], y1=Y_3[0], y2=Y_3[2]),
    ia.BoundingBox(x1=Y_4[1], x2=Y_4[3], y1=Y_4[0], y2=Y_4[2])], shape=img.shape)
    images,bboxes = augment(img,bbs)
    return images , bboxes
    
    
def augment(img,bbs):
    seq = []
    images = []
    bboxes = []
    eff = iaa.SomeOf((2,None),[
        iaa.Affine(rotate=(-30,30)), 
        iaa.Add((-40,40),per_channel=True),
        iaa.Fliplr(0.5),
#         iaa.Fliplud(0.5),
        iaa.Invert(0.25, per_channel=0.5),
        iaa.ContrastNormalization((0.5, 1.5)),
        iaa.Affine(scale=(0.8,1.2)),
        iaa.Affine(shear=(-16,16)),
        iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))])
    
    for y in np.arange(-0.1,0.2,0.2):
        for x in np.arange(-0.1,0.2,0.2):
            seq.append(iaa.Affine(translate_percent={"y": y, "x":x}, scale=0.8))

    for i in range(3):
        for pos in range(4):
            seq.append(iaa.Sequential([eff,seq[pos]]))
        seq.append(eff)
    seq_det = [x.to_deterministic() for x in seq]
    for aug in seq_det:
        image_aug = aug.augment_image(img)
        bbs_aug = aug.augment_bounding_boxes([bbs])[0]
        images.append(image_aug)
        bboxes.append(bbs_aug)
    return images,bboxes
    
def load_data(data_path):
    print("Loading the data........")
    X=[]
    Y_1=[]
    Y_2=[]
    Y_3=[]
    Y_4=[]
    data_files = os.listdir(data_path+"Ground_truth/")
    for file in data_files:
        img = cv2.imread(data_path+"Image/"+file[:-3]+"jpg")
        img = cv2.resize(img,(700,700))
        height,width = img.shape[:2]
        X.append(img)
        f = open(data_path+"Ground_truth/"+file,"r")
        gt = []
        for i in range(1,5):
            line = f.readline()[:-1]
            gt.append(list(map(int,line.split(','))))
        append_tool(Y_1,gt[0])
        append_tool(Y_2,gt[1])
        append_tool(Y_3,gt[2])
        append_tool(Y_4,gt[3])
        images , bboxes = augment_tool(gt[0],gt[1],gt[2],gt[3],img)
        convert_and_append(images,bboxes,Y_1,Y_2,Y_3,Y_4,X)
    print("Data Loaded successfully")
    return X,Y_1,Y_2,Y_3,Y_4

def load_data_test(data_path):
    print("Loading the data........")
    X=[]
    X_orig=[]
    names = []
    Y_1=[]
    Y_2=[]
    Y_3=[]
    Y_4=[]
    data_files = os.listdir(data_path+"Ground_truth/")
    for file in data_files:
        name = file[:-3]+"jpg"
        names.append(name)
#         print(data_path+"Image/"+file[:-3]+"jpg")
        img = cv2.imread(data_path+"Image/"+name)
        X_orig.append(img)
        img = cv2.resize(img,(700,700))
        height,width = img.shape[:2]
        X.append(img)
        f = open(data_path+"Ground_truth/"+file,"r")
        gt = []
        for i in range(1,5):
            line = f.readline()[:-1]
            gt.append(list(map(int,line.split(','))))
        append_tool(Y_1,gt[0])
        append_tool(Y_2,gt[1])
        append_tool(Y_3,gt[2])
        append_tool(Y_4,gt[3])
    print("Data Loaded successfully")
    return X,Y_1,Y_2,Y_3,Y_4,X_orig,names