function [ weight_0,weight_1,bias_0,bias_1,train_mse_list,valid_mse_list] = train( train_x,train_y,valid_x,valid_y,iter_num,step,base_lr,w,input_size,middle_size )

lr=base_lr;

weight_0=randn(input_size,middle_size)*1;
bias_0=randn(1,middle_size)*0;
weight_1=randn(middle_size,1)*1;
bias_1=randn(1,1)*0;

train_mse_list=zeros(iter_num,1);
valid_mse_list=zeros(iter_num,1);
batch_size=length(train_y);
valid_data_num=length(valid_y);
for i=1:iter_num
    batch_train_x=train_x+randn(size(train_x,1),input_size)*0.01;
    batch_train_y=train_y;
    batch_valid_x=valid_x;
    batch_valid_y=valid_y;
    
    [ output_data,d_weight_0,d_bias_0,d_weight_1,d_bias_1] = forward_back(batch_train_x,weight_0,bias_0,weight_1,bias_1,batch_train_y);
    
    
    mean_squre_error=sum((train_y-output_data).^2)/batch_size;
    train_mse_list(i)=mean_squre_error;
    [ v_output_data] = forward_back(batch_valid_x,weight_0,bias_0,weight_1,bias_1);
    v_mean_squre_error=sum((batch_valid_y-v_output_data).^2)/valid_data_num;
    valid_mse_list(i)=v_mean_squre_error;
    
    
    weight_0=weight_0-lr*(d_weight_0+w*(weight_0));
    bias_0=bias_0-lr*d_bias_0;
    weight_1=weight_1-lr*(d_weight_1+w*(weight_1));
    bias_1=bias_1-lr*d_bias_1;
    
    if(mod(i,step)==0)
        lr=lr*0.5;
    end
end

end

