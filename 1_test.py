
from utils.localization_utils import *

# path = input("Enter path to directory")
data_path = ["dataset/Knuckle/","dataset/Palm/","dataset/Vein/"]
names,X,X_orig,Y_class,Y_regress= load_data_test(data_path)

X=np.array(X)
X=X.astype('float32')/255.0
Y_class=np.array(Y_class)
Y_regress=np.array(Y_regress)
print("Total images loaded =",len(X))

network = load_model()

network.compile(optimizer = 'adam', loss = [categorical_crossentropy,dice_coef_loss], metrics = ['accuracy'])

network.load_weights("utils/data_1/weights/1_4.h5")

[Y_class_pred,Y_regress_pred] = network.predict(X)

f = open("output/1/iou.txt","w")
for ind,name in enumerate(names):
    img = X_orig[ind]
    height,width = img.shape[:2]
    Y_1 = Y_regress_pred[ind]
    Y_2 = Y_regress[ind]
    box_A = np.array([(Y_1[0]-Y_1[2]/2.0)*width,(Y_1[1]-Y_1[3]/2.0)*height,(Y_1[0]+Y_1[2]/2.0)*width,(Y_1[1]+Y_1[3]/2.0)*height])
    box_B = np.array([(Y_2[0]-Y_2[2]/2.0)*width,(Y_2[1]-Y_2[3]/2.0)*height,(Y_2[0]+Y_2[2]/2.0)*width,(Y_2[1]+Y_2[3]/2.0)*height])
    iou = bb_intersection_over_union(box_A,box_B)
    cv2.rectangle(img ,(int(box_B[0]),int(box_B[1])),(int(box_B[2]),int(box_B[3])),(0,255,0),3)
    cv2.rectangle(img ,(int(box_A[0]),int(box_A[1])),(int(box_A[2]),int(box_A[3])),(255,0,0),3)
    cv2.imwrite("output/1/images/"+name,img)
    f.write(name+":\t"+str(iou)+"\n")
f.close()

