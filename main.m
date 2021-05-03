clc;
clear all;
close all;

%% Parameters for training

load('hw2.mat');
filterlength = 5;
fold = 10;
L = 256;
M = 254;

number_of_samples_training = round(M * (fold - 1) / fold);
number_of_samples_test = M - number_of_samples_training;
%% 10 fold cross-validation

for fold_no=1:fold
%% Test Train Split      
    test_end = number_of_samples_test * fold_no;
    test_start = number_of_samples_test * (fold_no -1) + 1;
    
    total_ind = 1:M;
    ind = test_start:test_end;
    
    total_ind = setdiff(total_ind, ind);
    
    test_x_samples = x(:, ind);
    test_y_samples = y(:, ind);
    train_x_samples = x(:, total_ind);
    train_y_samples = y(:, total_ind);
    minimum_size_x_train= min(size(train_x_samples));
    
%% Mu Estimation

    mu_est = 0;
    for i = 1: number_of_samples_training
        mu_est = mu_est + train_x_samples(:, i);
    end
    mu_est = mu_est / number_of_samples_training;

%% Covariance Matrix Estimation

    cov_temp = 0;
    for i = 1:number_of_samples_training
        cov_temp =  cov_temp + (train_x_samples(:, i) - mu_est) * ...
            transpose(train_x_samples(:, i) - mu_est);
    end
    covariance_matrix = 0.0001*eye(L)+(cov_temp/number_of_samples_training);

%% Convolution Matrix Estimation

    for i=1: number_of_samples_training
        r = [train_x_samples(1, i)' zeros(1, filterlength)];
        c = [train_x_samples(:, i)'];
        X(:, :, i) = toeplitz(c, r);
    end

    conv_temp = 0;
    conv_temp_2 = 0;
    for i = 1: number_of_samples_training
        conv_temp = conv_temp + transpose(X(:, :, i)) * train_y_samples(:, i);
        conv_temp_2 = conv_temp_2 + transpose(X(:, :, i)) * X(:, :, i);
    end
    h_est = inv(conv_temp_2) * conv_temp;
    
    max_size_h = max(size(h_est));
    
    H_estimated = zeros(L,L);
    q = 1;
    for i = 1:L
        H_estimated(q: max_size_h+q-1, i) = h_est;
        q = q + 1;
    end
    H_estimated = H_estimated(1:L,:);




%% Variance Estimation  
    
    var_temp = 0;
    for i = 1:number_of_samples_training
        var_temp = var_temp + ...
            transpose(train_y_samples(:, i) - H_estimated * train_x_samples(:, i)) * ...
            (train_y_samples(:, i) - H_estimated * train_x_samples(:, i));
    end
    var_est = var_temp / ( number_of_samples_training * L);
    
%% Estimation of x    
        
    x_est =  inv(transpose(H_estimated) * H_estimated + var_est .* inv(covariance_matrix))*...
        (transpose(H_estimated) * test_y_samples + var_est * inv(covariance_matrix) * mu_est);
    
%% PSNR Calculation
        
    mse_temp = 0;
    for i=1:number_of_samples_test
        mse_temp = mse_temp+ power( norm(test_x_samples(:, i)-x_est(:, i)) ,2);
    end
    
    MSE{fold_no,1} = (mse_temp) / (L * number_of_samples_test);
    
    PSNR{fold_no,1}  = 20 * log10( max(test_x_samples(:)) / sqrt(MSE{fold_no,1}));
    
%% Figures    
    
    figure()
    sgtitle('Estimation of Input(x)');
    subplot(3,1,1);
    plot(1:L,x_est(:,2),'linewidth', 2);
    grid on;
    title('Estimated x');

    subplot(3,1,2);
    plot(1:L,test_x_samples(:,2),'linewidth', 2);
    grid on;
    title('Ground Truth');

    subplot(3,1,3);
    plot(1:L,test_y_samples(:,2),'linewidth', 2);
    grid on;
    title('Observation');
end