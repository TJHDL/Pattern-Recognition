import svmutil

train_name = 'train.txt'
test_name = 'test.txt'
linear_name = 'linear.model'
gaussian_name = 'gaussian.model'

y,x = svmutil.svm_read_problem(train_name) #read the LibSVM type train data
y1,x1 = svmutil.svm_read_problem(test_name) #read the LibSVM type test data

model_1 = svmutil.svm_train(y,x,'-t 0 -c 100')  #-t choose kernel_type 0->linear
model_2 = svmutil.svm_train(y,x,'-t 2 -c 100')  #-t choose kernel_type 2->radial basis function(Gaussian)

svmutil.svm_save_model(linear_name,model_1) #save_model use .txt can get the SV
svmutil.svm_save_model(gaussian_name,model_2) #save_model use .txt can get the SV

p1_label, p1_acc, p1_val = svmutil.svm_predict(y1,x1,model_1)
p2_label, p2_acc, p2_val = svmutil.svm_predict(y1,x1,model_2)