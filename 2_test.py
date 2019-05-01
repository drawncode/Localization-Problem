from utils.fingerprint_utils import *

path = "dataset/Four_Slap_Fingerprint/" 
X,Y_1,Y_2,Y_3,Y_4,X_orig,names = load_data_test(path)

X = np.asarray(X)
X=X.astype('float32')/255.0
Y_1 = np.asarray(Y_1)
Y_2 = np.asarray(Y_2)
Y_3 = np.asarray(Y_3)
Y_4 = np.asarray(Y_4)
X,Y_1,Y_2,Y_3,Y_4 = shuffle(X,Y_1,Y_2,Y_3,Y_4,random_state=0)

network = load_model()
# network.summary()

network.compile(optimizer = 'adam', loss = mean_squared_error, metrics = ['mse','mape'])

network.load_weights("utils/data_2/weights/2_9.h5")

[Y_1_pred,Y_2_pred,Y_3_pred,Y_4_pred] = network.predict(X)

f = open("output/2/iou.txt","w")
for ind,name in enumerate(names):
    iou =0
    img = X_orig[ind]
    height,width = img.shape[:2]
    B_1 = Y_1_pred[ind]
    B_2 = Y_1[ind]
    # print(Y_1,Y_2)
    box_A = np.array([(B_1[0]-B_1[2]/2.0)*width,(B_1[1]-B_1[3]/2.0)*height,(B_1[0]+B_1[2]/2.0)*width,(B_1[1]+B_1[3]/2.0)*height])
    box_B = np.array([(B_2[0]-B_2[2]/2.0)*width,(B_2[1]-B_2[3]/2.0)*height,(B_2[0]+B_2[2]/2.0)*width,(B_2[1]+B_2[3]/2.0)*height])
    iou += bb_intersection_over_union(box_A,box_B)
    # print("iou =",iou*100,"%")
    cv2.rectangle(img ,(int(box_B[0]),int(box_B[1])),(int(box_B[2]),int(box_B[3])),(0,255,0),3)
    cv2.rectangle(img ,(int(box_A[0]),int(box_A[1])),(int(box_A[2]),int(box_A[3])),(255,0,0),3)
    B_1 = Y_2_pred[ind]
    B_2 = Y_2[ind]
    # print(Y_1,Y_2)
    box_A = np.array([(B_1[0]-B_1[2]/2.0)*width,(B_1[1]-B_1[3]/2.0)*height,(B_1[0]+B_1[2]/2.0)*width,(B_1[1]+B_1[3]/2.0)*height])
    box_B = np.array([(B_2[0]-B_2[2]/2.0)*width,(B_2[1]-B_2[3]/2.0)*height,(B_2[0]+B_2[2]/2.0)*width,(B_2[1]+B_2[3]/2.0)*height])
    iou += bb_intersection_over_union(box_A,box_B)
    # print("iou =",iou*100,"%")
    cv2.rectangle(img ,(int(box_B[0]),int(box_B[1])),(int(box_B[2]),int(box_B[3])),(0,255,0),3)
    cv2.rectangle(img ,(int(box_A[0]),int(box_A[1])),(int(box_A[2]),int(box_A[3])),(255,0,0),3)
    B_1 = Y_3_pred[ind]
    B_2 = Y_3[ind]
    # print(Y_1,Y_2)
    box_A = np.array([(B_1[0]-B_1[2]/2.0)*width,(B_1[1]-B_1[3]/2.0)*height,(B_1[0]+B_1[2]/2.0)*width,(B_1[1]+B_1[3]/2.0)*height])
    box_B = np.array([(B_2[0]-B_2[2]/2.0)*width,(B_2[1]-B_2[3]/2.0)*height,(B_2[0]+B_2[2]/2.0)*width,(B_2[1]+B_2[3]/2.0)*height])
    iou += bb_intersection_over_union(box_A,box_B)
    # print("iou =",iou*100,"%")
    cv2.rectangle(img ,(int(box_B[0]),int(box_B[1])),(int(box_B[2]),int(box_B[3])),(0,255,0),3)
    cv2.rectangle(img ,(int(box_A[0]),int(box_A[1])),(int(box_A[2]),int(box_A[3])),(255,0,0),3)
    B_1 = Y_4_pred[ind]
    B_2 = Y_4[ind]
    box_A = np.array([(B_1[0]-B_1[2]/2.0)*width,(B_1[1]-B_1[3]/2.0)*height,(B_1[0]+B_1[2]/2.0)*width,(B_1[1]+B_1[3]/2.0)*height])
    box_B = np.array([(B_2[0]-B_2[2]/2.0)*width,(B_2[1]-B_2[3]/2.0)*height,(B_2[0]+B_2[2]/2.0)*width,(B_2[1]+B_2[3]/2.0)*height])
    iou += bb_intersection_over_union(box_A,box_B)
    cv2.rectangle(img ,(int(box_B[0]),int(box_B[1])),(int(box_B[2]),int(box_B[3])),(0,255,0),3)
    cv2.rectangle(img ,(int(box_A[0]),int(box_A[1])),(int(box_A[2]),int(box_A[3])),(255,0,0),3)
    f.write(name+":\t"+str(iou/4*100))
    cv2.imwrite("output/2/images/8_"+name,img)
f.close()

