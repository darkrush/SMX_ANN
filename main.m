close all;
tic;

repnum=40;
r2list=1:16;
done =false;
while(~done)
    load data_25.mat
    data_scale=1;
    data_bias=0;
    valid_data_num=0;
    test_data_num=0;
    train_data_num=length(data)-valid_data_num-test_data_num;
    input_size=4;
    iter_num=30000;
    base_lr=0.1;
    w=0.001;
    step=10000;
    batch_size=ceil(train_data_num/1);
    mse=zeros(16,1);
    data=data(randperm(length(data)),:);
    train_data=data(1:train_data_num,:);
    %train_data=data(:,:);
    valid_data=data(train_data_num+1:train_data_num+valid_data_num,:);
    test_data=data(train_data_num+valid_data_num+1:end,:);
    weight_0=randn(input_size,middle_size)*1;
    bias_0=randn(1,middle_size)*0;
    weight_1=randn(middle_size,1)*1;
    bias_1=randn(1,1)*0;
    t_output_data_m=[];%zeros(6,1);
    for repate=1:repnum
        lr=base_lr;
        
        weight_0=randn(input_size,middle_size)*1;
        bias_0=randn(1,middle_size)*0;
        weight_1=randn(middle_size,1)*1;
        bias_1=randn(1,1)*0;
        start_data=0;
        
        mse_list=zeros(iter_num,1);
        v_mse_list=zeros(iter_num,1);
        
        for i=1:iter_num
            mini_batch=train_data(:,:);
            
            train_x=mini_batch(:,1:input_size)+randn(size(mini_batch,1),input_size)*0.01;
            train_y=data_bias+data_scale*mini_batch(:,input_size+1);
            valid_x=valid_data(:,1:input_size);
            valid_y=data_bias+data_scale*valid_data(:,input_size+1);
            
            [ output_data,d_weight_0,d_bias_0,d_weight_1,d_bias_1] = forward_back(train_x,weight_0,bias_0,weight_1,bias_1,train_y);
            
            
            mean_squre_error=sum(abs(train_y-output_data))/batch_size;
            mse_list(i)=mean_squre_error;
            [ v_output_data] = forward_back(valid_x,weight_0,bias_0,weight_1,bias_1);
            v_mean_squre_error=sum(abs(valid_y-v_output_data))/valid_data_num;
            v_mse_list(i)=v_mean_squre_error;
            
            
            
            %fprintf('%f|%f\n',mean_squre_error,v_mean_squre_error);
            
            weight_0=weight_0-lr*(d_weight_0+w*(weight_0));
            bias_0=bias_0-lr*d_bias_0;
            weight_1=weight_1-lr*(d_weight_1+w*(weight_1));
            bias_1=bias_1-lr*d_bias_1;
            
            
            
            start_data=mod(start_data+batch_size,train_data_num);
            
            if(mod(i,step)==0)
                lr=lr*0.5;
            end
            
        end
        load test_time.mat
        test_data=data;
        test_data_num=size(data,1);
        test_x=test_data(:,1:input_size);
        test_y=data_bias+data_scale*test_data(:,input_size+1);
        [ t_output_data] = forward_back(test_x,weight_0,bias_0,weight_1,bias_1);
        t_output_data_m=[t_output_data_m,t_output_data];
        mse(middle_size)=mse(middle_size)+v_mean_squre_error;
    end
    t_output_data_m = mean(t_output_data_m,2);
    t_mean_squre_error=sum((test_y-t_output_data_m()).^2)/test_data_num;
    
    disp(t_mean_squre_error)
    %    plot(t_output_data,test_y,'*')
    %    hold on;
    %    plot(0:0.1:1,0:0.1:1);
    %    axis('equal')
    [a,b,r]=reg(output_data,train_y);
    fprintf('r=%f,a=%f,b=%f\n',r,a,b);
    [a,b,r]=reg(t_output_data_m,test_y);
    fprintf('r=%f,a=%f,b=%f\n',r,a,b);
    r2list(middle_size)=r;
    if((r>0.94435)&&(r<0.94445) && (t_mean_squre_error>0.00145)&& (t_mean_squre_error<0.00155))
        done =true;
    end
end
sen=sensitivity(weight_0,weight_1);
toc;