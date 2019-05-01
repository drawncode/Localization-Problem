from utils.fingerprint_utils import *

path = "dataset/Four_Slap_Fingerprint/" 
X,Y_1,Y_2,Y_3,Y_4 = load_data(path)

X = np.asarray(X)
X=X.astype('float32')/255.0
Y_1 = np.asarray(Y_1)
Y_2 = np.asarray(Y_2)
Y_3 = np.asarray(Y_3)
Y_4 = np.asarray(Y_4)
X,Y_1,Y_2,Y_3,Y_4 = shuffle(X,Y_1,Y_2,Y_3,Y_4,random_state=0)


print("Splitting data into test and train set")
X,Y_1,Y_2,Y_3,Y_4 = shuffle(X,Y_1,Y_2,Y_3,Y_4,random_state=0)
split = train_test_split(X,Y_1,Y_2,Y_3,Y_4,test_size=0.2, random_state=42)
(X_train,X_test,Y_1_train,Y_1_test,Y_2_train,Y_2_test,Y_3_train,Y_3_test,Y_4_train,Y_4_test) = split
X_train,Y_1_train,Y_2_train,Y_3_train,Y_4_train = shuffle(X_train,Y_1_train,Y_2_train,Y_3_train,Y_4_train,random_state = 0)
print("Train Set =", len(X_train), "Test Set =",len(X_test) )


print(X_train.shape,Y_1_train.shape,Y_2_train.shape,Y_3_train.shape,Y_4_train.shape)

network = load_model()

network.compile(optimizer = 'adam', loss = mean_squared_error, metrics = ['mse','mape'])


print("\n\nStarting the training")
for chunk in range(5):
    print("chunk",str(chunk))
    stats = network.fit(X_train,[Y_1_train,Y_2_train,Y_3_train,Y_4_train], epochs = 3, validation_data = (X_test,[Y_1_test,Y_2_test,Y_3_test,Y_4_test]), batch_size=16,verbose = 1,shuffle = True)
    print("\n\n Trained for chunk "+str(chunk)+", saving the weights\n")
    network.save_weights("utils/data_2/weights/2_"+str(chunk)+".h5")
print("Training completed successfully.")
