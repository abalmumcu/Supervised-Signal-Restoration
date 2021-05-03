clc;
clear all;
close all;

data = load('hw2.mat');

x = data.x;
y = data.y;
filterlength = 5;


K = 10;
M = 254;
Na_training_samples_len = round(M * (K - 1) / K);
Nb_test_samples_len = M - Na_training_samples_len;

for k=1:1:K
    test_start = Nb_test_samples_len * (k -1) + 1;
    test_end = Nb_test_samples_len * k;
    
    indices = test_start:1:test_end;
    total_indices = 1:1:M;
    
    total_indices = setdiff(total_indices, indices);
    
    test_x_samples = x(:, indices);
    test_y_samples = y(:, indices);
    
    train_x_samples = x(:, total_indices);
    train_y_samples = y(:, total_indices);
    

    mu_estimated = mu_estimate(Na_training_samples_len, train_x_samples);
    
    cov_matrix = covariance_matrix_estimate(train_x_samples, mu_estimated, Na_training_samples_len);
    
    % X = zeros(256, 5, min(size(train_x_samples)));
    for i=1: min(size(train_x_samples))
        r = [train_x_samples(1, i)' zeros(1, filterlength-1)];
        c = [train_x_samples(:, i)'];
        X(:, :, i) = toeplitz(c, r);
    end
    
    h_estimated = convolution_matrix_estimate(Na_training_samples_len, X, train_y_samples);
    
    
    H_estimated = zeros(256, 256);
    j = 1;
    for i = 1 : 256
        H_estimated(j:max(size(h_estimated)) + j -1, i) = h_estimated;
        j = j + 1;
    end
    H_estimated = H_estimated(1: 256, :);
    
    L = 256;
    var_estimated = variance_estimate(train_x_samples, train_y_samples, H_estimated, Na_training_samples_len, L);
    
    
    x_estimated = map_estimate(H_estimated, var_estimated, cov_matrix, test_y_samples, mu_estimated);
    
    
    PSNR{k,1} = psnr_calc(Nb_test_samples_len, test_x_samples, x_estimated);
    
    
    figure()
    sgtitle('Estimation');
    subplot(3,1,1);
    plot(1:256,x_estimated(:,1),'linewidth', 2);
    grid on;
    title('x_estimated');

    subplot(3,1,2);
    plot(1:256,test_x_samples(:,1),'linewidth', 2);
    grid on;
    title('GT-x');

    subplot(3,1,3);
    plot(1:256,test_y_samples(:,1),'linewidth', 2);
    grid on;
    title('observation');

    
    
end

function h_estimated = convolution_matrix_estimate(N, X, y)

temp_sum = 0;
for j = 1: N
    temp_sum = temp_sum + transpose(X(:, :, j)) * y(:, j);
end

temp_sum_2 = 0;
for i = 1: N
    temp_sum_2 = temp_sum_2 + transpose(X(:, :, i)) * X(:, :, i);
end

h_estimated = inv(temp_sum_2) * temp_sum;

end

function  mu_estimated = mu_estimate(N, x)

mu_estimated = 0;
for i = 1: 1: N
    mu_estimated = mu_estimated + x(:, i);
end
mu_estimated = mu_estimated / N;

end

function  cov_estimated = covariance_matrix_estimate(x, mu, Na)

temp_sum = 0;
for i = 1: 1: Na
    temp_sum = (x(:, i) - mu) * transpose(x(:, i) - mu);
end
temp_sum = temp_sum / Na;

cov_estimated = 0.0001 * eye(256) + temp_sum;

end

function var_estimated = variance_estimate(x, y, H, N, L)

temp_sum = 0;
for i = 1: 1: N
    temp_sum = temp_sum + transpose(y(:, i) - H * x(:, i)) * (y(:, i) - H * x(:, i));
end
var_estimated = temp_sum / ( N * L);

end

function x_estimated = map_estimate(H, variance, cov_matrix_E, y, mu)

x_estimated =  inv(transpose(H) * H + variance .* inv(cov_matrix_E)) *  (transpose(H) * y + variance * inv(cov_matrix_E) * mu);

end

function PSNR = psnr_calc(Nb, x, x_est)

mse = mse_calculate(Nb, x, x_est);

PSNR = 20 * log10( max(x(:)) / sqrt(mse));

end

function mse = mse_calculate(Nb, x, x_est)

temp = zeros(Nb, 1);
for i=1:1:Nb
    temp(i)= ( norm( x(:, i) - x_est(:, i) ) ) ^ 2;
end

mse = (sum(temp) / Nb) / (256 * Nb);

end






