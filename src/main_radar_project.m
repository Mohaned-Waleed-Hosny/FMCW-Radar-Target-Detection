%% 
% % % % % % % % % % % % ---------------- Scenarios Choice
clc, close all, clear all
fprintf("Program started.....\n");

scenario_id = 1;
radar_config;
fprintf("Tested Scenario is %i.\n", scenario_id);
% 1:Normal                        2:Low SNR                 3:Stationary
% 4:Direction Detection           5:Close Range             6:Close Velcoity         
% 7:High Velocity Difference      8:Unambiguous Range       9:Multiple Objects

use_noise = true;      % Set to True to add noisy target inputs

%% 
% % % % % % % % % % % % ---------------- Parameters
c =  3e8;               % Light Speed
T_chirp = 8e-6;         % 8 us
PRI = T_chirp;          % as Tchirp = 8 us (max is 9.8) 
f_c = 76.5e9;           % 76.5 GHz
f_min = -0.5e9;         % or 76 GHZ
f_max = 0.5e9;          % or 77 GHZ
bw = f_max - f_min;     % BandWidth is 1 GHz
R_min = 0.75;           % Min Range is 0.75 m
R_max = 250;            % Max Range is 250 m
RTT_max = 2 * R_max / c;
T_sample = 0.5e-9;      % Ts is 0.5 ns
f_sample = 1/T_sample;
T_window = RTT_max/4;
N_sample = round(T_chirp/T_sample);
N_p = 512;              % number of chirps
S = bw/T_chirp;         % Slope
time = 0;               % The total processing time of FFT

%% 
% % % % % % % % % % ---------------- Transmitted Signal

% % % Initializing one Chirp
% % % ======================

t = (0:N_sample-1).' * T_sample;                        % n*Ts
x_chirp = 1*exp(1j*2*pi*(t.* f_min + S * t.^2 / 2));    % One Chirp x_chirp

% % % Sending the whole Transmitted Signal
% % % ======================

x_tx = zeros(N_sample, N_p);
for m=1:N_p
    x_tx(:,m) = x_chirp; 
end

% % % Transmitted Signal Plots
% % % ======================

f_tx = f_min + S * t;       % Instantaneous frequency

figure(1);
subplot(2,1,1);             % Frequency of one chirp
plot(t*1e6, f_tx/1e9);
xlabel('Time (\mus)'); 
ylabel('One Chirp Frequency (GHz)');
title('Frequency of One FMCW Chirp'); 
grid on;

f_plot = repmat(f_tx, N_p, 1);
t_plot = (0:length(f_plot)-1).' * T_sample;
subplot(2,1,2);
plot(t_plot*1e6, f_plot/1e9);
xlabel('Time (\mus)');
ylabel('Frequency (GHz)');
title('Chirps Frequency vs Time (Repeated FMCW Chirps)');
grid on;

%% 
% % % % % % % % % % ---------------- Received Signal

y_rx = zeros(size(x_tx));        
for k = 1:length(R_targets)
    % Adding Targets using variables provided by radar_config
    t_delay_k = 2 * R_targets(k) / c;               % Delay of an object
    N_delay_k = round(t_delay_k / T_sample);        % Sampled Delay
    f_D_k = 2 * v_targets(k) * f_c / c;             % Doppler Frequency of an object
    
    % Delay Effect over one chirp --> Range
    target_signal = zeros(size(x_tx));
    if N_delay_k < N_sample
        target_signal(N_delay_k+1:end, :) = A_targets(k) * x_tx(1:end-N_delay_k, :);
    end
    % Doppler effect across chirps --> Velocity
    for m = 1:N_p
        target_signal(:, m) = target_signal(:, m) * exp(1j * 2 * pi * f_D_k * (m-1) * PRI);
    end
    y_rx = y_rx + target_signal;                    % Superposition
end

% % % Add white Gaussian noise
% % % ======================
if use_noise
    fprintf("Noise is ENABLED for this run.\n");
    y_rx = awgn(y_rx, SNR_target_db, 'measured');   % awgn = Additive white Gaussian noise
else
    fprintf("Noise is DISABLED for this run.\n");
end

% % % Received Signal Plots
% % % ======================
figure(2);
subplot(2,1,1);
plot(real(x_tx(:,1)), 'b');
xlabel('Time (\mus)');
ylabel('Magnitude');
title('Transmitted Signal'); 
grid on;

subplot(2,1,2);
plot(real(y_rx(:,1)), 'r');
xlabel('Time (\mus)');
ylabel('Magnitude');
title('Received Signal (Delayed)'); 
grid on;

% Verify Doppler Plots
figure(3);
subplot(2,1,1);
plot(real(y_rx(:,1)));
xlabel('Time (\mus)');
title('Fast-time: Range Effect (Single Chirp)');
grid on;

n_test = round(2 * R_targets(1) / c / T_sample) + 50;
subplot(2,1,2); 
plot(real(y_rx(n_test,:))); 
xlabel('Chirp Index');
title('Slow-time: Velcity / Doppler Effect (Across Chirps)'); 
grid on;


%% 
% % % % % % % % % % ----------------  Signal Processing
% % % Mixing Tx and Rx
% % % ======================
tic;
z_mix = x_tx.* conj(y_rx);

% % % Windowing = Removing Idle Periods
% % % ====================== 
n_start = round(RTT_max / T_sample);        % Start --> Sampled RTT_max
n_end   = N_sample;                         % End   --> Sampled T_chirp
z_windowed  = z_mix(n_start:n_end, :);
time = toc;

figure(4);
subplot(1,2,1); 
plot(real(z_windowed(:,1)));
xlabel('Fast-time samples (windowed)'); 
ylabel('Real [z]'); 
title('Windowed & Mixed Signal'); 
grid on;
subplot(1,2,2); 
plot(angle(z_windowed(:,1)));
xlabel('Fast-time samples (windowed)'); 
ylabel('Angle [z]'); 
title('Windowed & Mixed Signal'); 
grid on;

%% 
% % % FFT in the Fast-time Axis --> Range
% % % ======================
tic;
Nfft_r = 2^nextpow2(size(z_windowed,1));        % for number of points to be a power of 2
z_r = fft(z_windowed, Nfft_r, 1);               % FFT across rows
z_r = z_r(1:Nfft_r/2, :);
time = time + toc;

f_beat = (0:Nfft_r/2-1).' * (f_sample / Nfft_r);   % Beat Frequency
R_axis = (c * f_beat) / (2 * S);                   % Range Axis

figure(5);
range_data = abs(z_r(:,1));                     % first chirp only for range detection
plot(R_axis, range_data);
hold on;
xlabel('Range (m)'); 
ylabel('Magnitude'); 
title('Range Peaks');
xlim([0 R_max]); 
grid on;

% % % Automated Detection of peaks
% % % ====================== 

[pks, locs_r] = findpeaks(range_data, 'MinPeakHeight', max(range_data)*0.25, 'MinPeakDistance', 2);
plot(R_axis(locs_r), pks, 'ro');

for i = 1:length(pks)
    text(R_axis(locs_r(i)), pks(i)*1.05, sprintf('%.1f m', R_axis(locs_r(i))), ...
        'HorizontalAlignment', 'center');
end

% % % Range CA-CFAR
% % % ====================== 
r_train = 15;    
r_guard = 3;     
r_offset = 2.2;

sig_r = abs(z_r(:, 1)); 
N_r = length(sig_r);
r_threshold = zeros(N_r, 1);
r_detect = zeros(N_r, 1);

for i = (r_train + r_guard + 1) : (N_r - (r_train + r_guard))
    % Split noise cells into Left and Right windows
    noise_left  = sig_r(i-r_train-r_guard : i-r_guard-1);
    noise_right = sig_r(i+r_guard+1 : i+r_train+r_guard);
    
    % Greatest-Of Logic: Picks the noisier side to set a safer threshold
    r_threshold(i) = max(mean(noise_left), mean(noise_right)) * r_offset;
    
    if sig_r(i) > r_threshold(i)
        r_detect(i) = sig_r(i);
    end
end

figure(6);
plot(R_axis, sig_r, 'Color', 'w'); hold on;
plot(R_axis, r_threshold, 'r--');         
stem(R_axis, r_detect, 'g', 'Marker', 'none');             
xlabel('Range (m)'); 
xlim([0, R_max]);
ylabel('Magnitude'); 
title('CA-CFAR for Range');
legend('Signal', 'Threshold', 'Detections'); grid on;

%% 
% % % FFT in the Slow-time Axis --> Velocity
% % % ======================
tic;
Nfft_v = 512;               % Number of chirps
z_v = fftshift(fft(z_r, Nfft_v, 2), 2); 
time = time + toc;

fD_axis = (-Nfft_v/2 : Nfft_v/2-1) / (Nfft_v * PRI);
v_axis = -(c * fD_axis) / (2 * f_c);            % -ve means that the target is receding

% % % Velocity Profile
% % % ====================== 
figure(7);
vel_data = max(abs(z_v), [], 1); 
plot(v_axis, vel_data); 
xlabel('Velocity (m/s)'); 
ylabel('Magnitude'); 
title('Velocity Peaks');
xlim([-150 150]); 
grid on;
hold on;

[pks_v, locs_v] = findpeaks(vel_data, 'MinPeakHeight', max(vel_data)*0.25);
plot(v_axis(locs_v), pks_v, 'ro');

for i = 1:length(pks_v)
    v_val = v_axis(locs_v(i));
    
    if v_val > 0.5
        status = ' (Receding)';
    elseif v_val < -0.5
        status = ' (Approaching)';
    else
        status = ' (Stationary)';
    end
    
    label = [num2str(v_val, '%.1f'), ' m/s', status];
    text(v_val, pks_v(i)*1.05, label, 'HorizontalAlignment', 'center', ...
         'Color', 'r', 'FontWeight', 'bold');
end

% % % Velocity CA-CFAR
% % % ====================== 
v_train = 15;        
v_guard = 2;         
v_offset = 1.3;          

sig_v = max(abs(z_v), [], 1); 
N_v = length(sig_v);
v_threshold = zeros(1, N_v);
v_detect = zeros(1, N_v);

for i = (v_train + v_guard + 1) : (N_v - (v_train + v_guard))
    % Extract training cells
    v_noise = [sig_v(i-v_train-v_guard : i-v_guard-1), sig_v(i+v_guard+1 : i+v_train+v_guard)];
    v_threshold(i) = mean(v_noise) * v_offset;              % Compute threshold
    if sig_v(i) > v_threshold(i)
        v_detect(i) = sig_v(i);
    end
end

figure(8);
plot(v_axis, sig_v, 'Color', [0.5 0.5 0.5]); 
hold on;
plot(v_axis, v_threshold, 'r--');
stem(v_axis, v_detect, 'g', 'Marker', 'none');
xlabel('Velocity (m/s)'); 
xlim([-150 150]); 
ylabel('Magnitude');
title('CA-CFAR for Velocity');
legend('Signal', 'Threshold', 'Detections');
grid on;

%% 
% % % Range Velocity Map, in dB
% % % ====================== 
figure(9);
imagesc(v_axis, R_axis, 20*log10(abs(z_v))); 
xlabel('Velocity (m/s)'); 
ylabel('Range (m)');
title('Range–Velocity Map (dB Scale)'); 
colorbar; 
axis xy;

% % % Table
% % % ====================== 

% Range Table
true_R = R_targets(:);
matched_R = arrayfun(@(x) R_axis(locs_r(find(abs(R_axis(locs_r) - x) == min(abs(R_axis(locs_r) - x)), 1))), true_R);

T1 = table((1:length(true_R))', matched_R, true_R, abs(matched_R - true_R), ...
    'VariableNames', {'ID', 'Detected Range (m)', 'True Range (m)', 'Error (m)'});

% Velocity Table
true_V = v_targets(:);
matched_V = arrayfun(@(x) v_axis(locs_v(find(abs(v_axis(locs_v) - x) == min(abs(v_axis(locs_v) - x)), 1))), true_V);

status = repmat({'Stationary'}, length(true_V), 1);
status(matched_V > 0.5) = {'Receding'};
status(matched_V < -0.5) = {'Approaching'};

T2 = table((1:length(true_V))', matched_V, true_V, abs(matched_V - true_V), status, ...
    'VariableNames', {'ID', 'Detected Vel. (m/s)', 'True Vel. (m/s)', 'Error (m/s)', 'Status'});

% Console Display
fprintf("______________ FMCW Radar Detection Results" + ...
        "______________\n\n");
disp('Table 1: Range Performance'); disp(T1);
disp('Table 2: Velocity Performance'); disp(T2);
fprintf('FFT Processing Time: %.4f ms\n\n', time * 1e3);

fprintf('Program ended...\n');
