clear
addpath ../function
load ../data/data_train_valid.mat

for i=1:size(data,1)
    data_new(i,1:4)=data(i,1:4);
    data_new(i,5:8)=data(i,1:4).^2;
    data_new(i,9)=data(i,1).*data(i,2);
    data_new(i,10)=data(i,1).*data(i,3);
    data_new(i,11)=data(i,1).*data(i,4);
    data_new(i,12)=data(i,2).*data(i,3);
    data_new(i,13)=data(i,2).*data(i,4);
    data_new(i,14)=data(i,3).*data(i,4);
    data_new(i,15)=data(i,5);
end
data=data_new;
input_num=size(data,2)-1;
train_x=data(1:25,1:input_num);
train_y=data(1:25,input_num+1);
valid_x=data(21:25,1:input_num);
valid_y=data(21:25,input_num+1);




iter_num = 30000;
step = 1000;
base_lr = 0.1;
input_size=input_num;
middle_size=input_num*2;
w=0.01;

repnum=40;
weight_0_s=[];
weight_1_s=[];
bias_0_s=[];
bias_1_s=[];
for i=1:repnum
    disp(i);
    [ weight_0,weight_1,bias_0,bias_1,train_mse_list,valid_mse_list] = train( train_x,train_y,valid_x,valid_y,iter_num,step,base_lr,w,input_size,middle_size);
    weight_0_s(:,:,i)=weight_0;
    weight_1_s(:,:,i)=weight_1;
    bias_0_s(:,:,i)=bias_0;
    bias_1_s(:,:,i)=bias_1;
end

load ../data/data_test.mat
data_new=[];
for i=1:size(data,1)
    data_new(i,1:4)=data(i,1:4);
    data_new(i,5:8)=data(i,1:4).^2;
    data_new(i,9)=data(i,1).*data(i,2);
    data_new(i,10)=data(i,1).*data(i,3);
    data_new(i,11)=data(i,1).*data(i,4);
    data_new(i,12)=data(i,2).*data(i,3);
    data_new(i,13)=data(i,2).*data(i,4);
    data_new(i,14)=data(i,3).*data(i,4);
    data_new(i,15)=data(i,5);
end
test_data=data_new;
test_data_num=size(data,1);
test_x=test_data(:,1:input_size);
test_y=test_data(:,input_size+1);
test_output_data_mean=test_y*0;
sen=zeros(size(weight_0,1),1);
for i=1:repnum
    [ test_output_data] = forward_back(test_x,weight_0,bias_0,weight_1,bias_1);
    test_output_data_mean=test_output_data_mean+test_output_data/repnum;
    sen= sen+sensitivity (weight_0,weight_1)/repnum;
end
t_mean_squre_error=sum((test_y-test_output_data_mean).^2)/test_data_num;
[a,b,r]=reg(test_output_data_mean,test_y);

fprintf('%f %f\n',r,t_mean_squre_error)