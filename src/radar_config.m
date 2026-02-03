%% 
% % % % % % % % % % ---------------- Radar Scenarios Configuration

switch scenario_id
    case 1 % Normal
        R_targets = [100, 220]; v_targets = [30, 110]; A_targets = [1, 0.8]; SNR_target_db = 15;
    case 2 % Low SNR
        R_targets = [100, 220]; v_targets = [30, 110]; A_targets = [1, 1]; SNR_target_db = 5;
    case 3 % Stationary
        R_targets = [100, 220]; v_targets = [0, 0]; A_targets = [1, 0.7]; SNR_target_db = 15;
    case 4 % Receding/Approaching Mix
        R_targets = [100, 220]; v_targets = [-50, 110]; A_targets = [1, 1]; SNR_target_db = 15;
    case 5 % Close Range
        R_targets = [120, 122]; v_targets = [30, 110]; A_targets = [1, 0.9]; SNR_target_db = 15;
    case 6 % Close Velocity
        R_targets = [100, 220]; v_targets = [30, 32]; A_targets = [1, 1]; SNR_target_db = 15;
    case 7 % High Velocity Difference
        R_targets = [100, 220]; v_targets = [-100, 120]; A_targets = [1, 1]; SNR_target_db = 15;
    case 8 % Unambiguous Range
        R_targets = [10, 240]; v_targets = [30, 90]; A_targets = [1, 0.5]; SNR_target_db = 15;
    case 9 % Multiple Objects
        R_targets = [10, 105, 75, 210]; v_targets = [12, -70, 0, 97]; A_targets = [1, 0.4, 0.8, 0.6]; SNR_target_db = 15;
end
