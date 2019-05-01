from utils.localization_utils import *

data_path = ["dataset/Knuckle/","dataset/Palm/","taset/data_1/dataset_arrays/Vein/"]
X,Y_class,Y_regress = load_data(data_path)

X=np.array(X)
X=X.astype('float32')/255.0
Y_class=np.array(Y_class)
Y_regress=np.array(Y_regress)
print("Total images loaded =",len(X))

print(X_train.shape, Y_class_train.shape, Y_regress_train.shape)

network = load_model()
# network.summary()

network.compile(optimizer = 'adam', loss = [categorical_crossentropy,mean_squared_error], metrics = ['accuracy','mse','mape'])

print("\n\nStarting the training")
for chunk in range(5):
    print("chunk",str(chunk))
    stats = network.fit(X_train,[Y_class_train,Y_regress_train], epochs = 3, validation_data = (X_test,[Y_class_test,Y_regress_test]), batch_size=16,verbose = 1,shuffle = True)
    print("\n\n Trained for chunk "+str(chunk)+", saving the weights\n")
    network.save_weights("utils/data_1/weights/1_"+str(chunk)+".h5")
    network.load_weights("utils/data_1/weights/1_"+str(chunk)+".h5")
print("Training completed successfully.")