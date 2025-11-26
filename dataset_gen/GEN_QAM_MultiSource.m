function GEN_QAM_MultiSource()
    % Generate multi-source QAM dataset with different SNR
    for snr = 10:4:30
        % Dual-source dataset
        %GEN_QAM_DualSource_i(snr, 20, '16QAM', '64QAM');
        %GEN_QAM_DualSource_i(snr, 10, '64QAM', '64QAM');
        %GEN_QAM_DualSource_i(snr, 10, '64QAM', '128QAM');
        
        % % Tri-source dataset
        GEN_QAM_TriSource_i(snr, 1, '16QAM', '64QAM', '128QAM');
    end
end

function GEN_QAM_DualSource_i(snr, num_files, mod1_type, mod2_type)
    % Dual-source QAM signal generation function
    % snr: Signal-to-noise ratio (dB)
    % num_files: Number of files
    % mod1_type, mod2_type: Modulation types ('16QAM', '64QAM', '128QAM', '256QAM')
    
    %% Parameter validation
    if nargin < 1
        snr = 25;
    end
    if nargin < 2
        num_files = 20;
    end
    
    %% Carrier frequency configuration (normalized frequency offset 2.5e-5)
    fc_base = 20e6;
    freq_offset = fc_base * 2.5e-5;  % 500Hz
    fc_offsets = [-freq_offset, freq_offset];  % Dual-source: ±500Hz
    fc_array = fc_base + fc_offsets;
    
    %% Initial phase configuration
    initial_phases = [0, pi/4];  % Initial phases for two sources
    
    %% Basic parameters
    flo = 20e6;
    Fs_rf = 100e6;
    
    %% Root raised cosine filter
    Fs_sps = 20;
    alpha = 0.35;
    span = 20;
    filterCoeffs = rcosdesign(alpha, span, Fs_sps, 'sqrt');
    
    %% Dataset parameters
    samples_per_file = 500;
    frame_length = 4096;
    symbols_per_frame = 205;
    total_frames = num_files * samples_per_file;
    total_samples = total_frames * frame_length;
    total_symbols = total_frames * symbols_per_frame;
    
    %% Modulation parameter acquisition
    [M1, bits_per_symbol1] = get_qam_params(mod1_type);
    [M2, bits_per_symbol2] = get_qam_params(mod2_type);
    
    %% Ensure consistent bit count - use maximum bit count
    max_bits_per_symbol = max(bits_per_symbol1, bits_per_symbol2);
    total_bits = total_symbols * max_bits_per_symbol;
    
    %% Delay parameters (reduce delay)
    symbol_rate = 5e6;
    Tb = 1/symbol_rate;
    base_delay = 0.02 * Tb;  % Reduce base delay
    delay_samples = [0, round(base_delay * Fs_rf)];
    
    %% Low-pass filter design
    rolloff = 0.35;
    cutoff_freq = symbol_rate * (1+rolloff)/2;
    normalized_cutoff = cutoff_freq/(Fs_rf/2);
    h_lpf = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
    
    %% Generate global time axis
    t_global = (0:total_samples-1)' / Fs_rf;
    
    %% Generate bit streams and modulated signals
    rf_signals = zeros(length(t_global), 2);
    ideal_bb_signals = zeros(length(t_global), 2);
    
    % Generate bit data
    bit_data_1 = randi([0, 1], total_bits, 1, 'uint8');
    bit_data_2 = randi([0, 1], total_bits, 1, 'uint8');
    
    %% Source 1 modulation
    % Extract required bits
    bits_needed_1 = total_symbols * bits_per_symbol1;
    symbol_indices_1 = bi2de(reshape(bit_data_1(1:bits_needed_1), bits_per_symbol1, [])', 'left-msb');
    s_complex_1 = qammod(symbol_indices_1, M1, 'UnitAveragePower', true);
    
    % Upsampling and pulse shaping
    s_upsampled_1 = upsample(s_complex_1, Fs_sps);
    s_shaped_1 = conv(s_upsampled_1, filterCoeffs, 'same');
    
    % Add delay
    if delay_samples(1) > 0
        s_shaped_1 = [zeros(delay_samples(1), 1); s_shaped_1(1:end-delay_samples(1))];
    end
    
    % Ensure consistent length
    if length(s_shaped_1) > length(t_global)
        s_shaped_1 = s_shaped_1(1:length(t_global));
    elseif length(s_shaped_1) < length(t_global)
        s_shaped_1 = [s_shaped_1; zeros(length(t_global) - length(s_shaped_1), 1)];
    end
    
    % Upconversion
    carrier_1 = exp(1i*(2*pi*fc_array(1)*t_global + initial_phases(1)));
    rf_signals(:, 1) = real(s_shaped_1 .* carrier_1);
    
    %% Source 2 modulation
    bits_needed_2 = total_symbols * bits_per_symbol2;
    symbol_indices_2 = bi2de(reshape(bit_data_2(1:bits_needed_2), bits_per_symbol2, [])', 'left-msb');
    s_complex_2 = qammod(symbol_indices_2, M2, 'UnitAveragePower', true);
    
    % Upsampling and pulse shaping
    s_upsampled_2 = upsample(s_complex_2, Fs_sps);
    s_shaped_2 = conv(s_upsampled_2, filterCoeffs, 'same');
    
    % Add delay
    if delay_samples(2) > 0
        s_shaped_2 = [zeros(delay_samples(2), 1); s_shaped_2(1:end-delay_samples(2))];
    end
    
    % Ensure consistent length
    if length(s_shaped_2) > length(t_global)
        s_shaped_2 = s_shaped_2(1:length(t_global));
    elseif length(s_shaped_2) < length(t_global)
        s_shaped_2 = [s_shaped_2; zeros(length(t_global) - length(s_shaped_2), 1)];
    end
    
    % Upconversion
    carrier_2 = exp(1i*(2*pi*fc_array(2)*t_global + initial_phases(2)));
    rf_signals(:, 2) = real(s_shaped_2 .* carrier_2);
    
    %% Ideal demodulation (label signals)
    lo_i = cos(2*pi*flo*t_global);
    lo_q = sin(2*pi*flo*t_global);
    
    for src_idx = 1:2
        baseband_i = rf_signals(:, src_idx) .* lo_i;
        baseband_q = rf_signals(:, src_idx) .* (-lo_q);
        bb_i_filtered = conv(baseband_i, h_lpf, 'same');
        bb_q_filtered = conv(baseband_q, h_lpf, 'same');
        ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
    end
    
    %% Mix signals and add noise
    rf_combined = sum(rf_signals, 2);
    rf_combined_noisy = awgn(rf_combined, snr, 'measured');
    
    %% Downconvert to baseband
    baseband_i = rf_combined_noisy .* lo_i;
    baseband_q = rf_combined_noisy .* (-lo_q);
    bb_i_filtered = conv(baseband_i, h_lpf, 'same');
    bb_q_filtered = conv(baseband_q, h_lpf, 'same');
    mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
    
    %% Global normalization
    global_max_mixed = max(abs(mixed_baseband));
    global_max_ideal = max(max(abs(ideal_bb_signals)));
    
    mixed_baseband = mixed_baseband / global_max_mixed;
    ideal_bb_signals = ideal_bb_signals / global_max_ideal;
    
    %% Segment and save data
    save_dual_source_data(mixed_baseband, ideal_bb_signals, bit_data_1, bit_data_2, ...
                         samples_per_file, frame_length, num_files, snr, mod1_type, mod2_type, ...
                         bits_per_symbol1, bits_per_symbol2, symbols_per_frame);
    
    %% Display configuration information
    fprintf('\n=== %s+%s Dual-source Dataset Generation Complete ===\n', mod1_type, mod2_type);
    fprintf('SNR: %d dB\n', snr);
    fprintf('Carrier Frequencies: %.6f MHz, %.6f MHz\n', fc_array(1)/1e6, fc_array(2)/1e6);
    fprintf('Frequency Offsets: %+.0f Hz, %+.0f Hz\n', fc_offsets(1), fc_offsets(2));
    fprintf('Delays: %.1f ns, %.1f ns\n', delay_samples(1)/Fs_rf*1e9, delay_samples(2)/Fs_rf*1e9);
end

function GEN_QAM_TriSource_i(snr, num_files, mod1_type, mod2_type, mod3_type)
    % Tri-source QAM signal generation function
    
    %% Parameter validation
    if nargin < 1
        snr = 25;
    end
    if nargin < 2
        num_files = 20;
    end
    
    %% Carrier frequency configuration (symmetric distribution)
    fc_base = 20e6;
    freq_offset = fc_base * 2.5e-5;  % 500Hz
    fc_offsets = [-freq_offset, 0, freq_offset];  % Tri-source: -500, 0, 500Hz
    fc_array = fc_base + fc_offsets;
    
    %% Initial phase configuration
    initial_phases = [0, pi/3, 2*pi/3];  % Initial phases for three sources
    
    %% Basic parameters
    flo = 20e6;
    Fs_rf = 100e6;
    
    %% Root raised cosine filter
    Fs_sps = 20;
    alpha = 0.35;
    span = 20;
    filterCoeffs = rcosdesign(alpha, span, Fs_sps, 'sqrt');
    
    %% Dataset parameters
    samples_per_file = 500;
    frame_length = 4096;
    symbols_per_frame = 205;
    total_frames = num_files * samples_per_file;
    total_samples = total_frames * frame_length;
    total_symbols = total_frames * symbols_per_frame;
    
    %% Modulation parameter acquisition
    [M1, bits_per_symbol1] = get_qam_params(mod1_type);
    [M2, bits_per_symbol2] = get_qam_params(mod2_type);
    [M3, bits_per_symbol3] = get_qam_params(mod3_type);
    
    %% Ensure consistent bit count
    max_bits_per_symbol = max([bits_per_symbol1, bits_per_symbol2, bits_per_symbol3]);
    total_bits = total_symbols * max_bits_per_symbol;
    
    %% Delay parameters
    symbol_rate = 5e6;
    Tb = 1/symbol_rate;
    base_delay = 0.02 * Tb;
    delay_samples = [0, round(base_delay * Fs_rf), round(2*base_delay * Fs_rf)];
    
    %% Low-pass filter design
    rolloff = 0.35;
    cutoff_freq = symbol_rate * (1+rolloff)/2;
    normalized_cutoff = cutoff_freq/(Fs_rf/2);
    h_lpf = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
    
    %% Generate global time axis
    t_global = (0:total_samples-1)' / Fs_rf;
    
    %% Generate bit streams and modulated signals
    rf_signals = zeros(length(t_global), 3);
    ideal_bb_signals = zeros(length(t_global), 3);
    
    % Generate bit data
    bit_data_1 = randi([0, 1], total_bits, 1, 'uint8');
    bit_data_2 = randi([0, 1], total_bits, 1, 'uint8');
    bit_data_3 = randi([0, 1], total_bits, 1, 'uint8');
    
    % Modulation parameter arrays
    M_array = [M1, M2, M3];
    bits_array = [bits_per_symbol1, bits_per_symbol2, bits_per_symbol3];
    bit_data_array = {bit_data_1, bit_data_2, bit_data_3};
    
    %% Three-source modulation and upconversion
    for src_idx = 1:3
        % Modulation
        bits_needed = total_symbols * bits_array(src_idx);
        symbol_indices = bi2de(reshape(bit_data_array{src_idx}(1:bits_needed), bits_array(src_idx), [])', 'left-msb');
        s_complex = qammod(symbol_indices, M_array(src_idx), 'UnitAveragePower', true);
        
        % Upsampling and pulse shaping
        s_upsampled = upsample(s_complex, Fs_sps);
        s_shaped = conv(s_upsampled, filterCoeffs, 'same');
        
        % Add delay
        if delay_samples(src_idx) > 0
            s_shaped = [zeros(delay_samples(src_idx), 1); s_shaped(1:end-delay_samples(src_idx))];
        end
        
        % Ensure consistent length
        if length(s_shaped) > length(t_global)
            s_shaped = s_shaped(1:length(t_global));
        elseif length(s_shaped) < length(t_global)
            s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
        end
        
        % Upconversion
        carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
        rf_signals(:, src_idx) = real(s_shaped .* carrier);
    end
    
    %% Ideal demodulation
    lo_i = cos(2*pi*flo*t_global);
    lo_q = sin(2*pi*flo*t_global);
    
    for src_idx = 1:3
        baseband_i = rf_signals(:, src_idx) .* lo_i;
        baseband_q = rf_signals(:, src_idx) .* (-lo_q);
        bb_i_filtered = conv(baseband_i, h_lpf, 'same');
        bb_q_filtered = conv(baseband_q, h_lpf, 'same');
        ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
    end
    
    %% Mix signals and add noise
    rf_combined = sum(rf_signals, 2);
    rf_combined_noisy = awgn(rf_combined, snr, 'measured');
    
    %% Downconvert to baseband
    baseband_i = rf_combined_noisy .* lo_i;
    baseband_q = rf_combined_noisy .* (-lo_q);
    bb_i_filtered = conv(baseband_i, h_lpf, 'same');
    bb_q_filtered = conv(baseband_q, h_lpf, 'same');
    mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
    
    %% Global normalization
    global_max_mixed = max(abs(mixed_baseband));
    global_max_ideal = max(max(abs(ideal_bb_signals)));
    
    mixed_baseband = mixed_baseband / global_max_mixed;
    ideal_bb_signals = ideal_bb_signals / global_max_ideal;
    
    %% Save data
    save_tri_source_data(mixed_baseband, ideal_bb_signals, bit_data_1, bit_data_2, bit_data_3, ...
                        samples_per_file, frame_length, num_files, snr, mod1_type, mod2_type, mod3_type, ...
                        bits_per_symbol1, bits_per_symbol2, bits_per_symbol3, symbols_per_frame);
    
    %% Display configuration information
    fprintf('\n=== %s+%s+%s Tri-source Dataset Generation Complete ===\n', mod1_type, mod2_type, mod3_type);
    fprintf('SNR: %d dB\n', snr);
    fprintf('Carrier Frequencies: %.6f MHz, %.6f MHz, %.6f MHz\n', fc_array(1)/1e6, fc_array(2)/1e6, fc_array(3)/1e6);
    fprintf('Frequency Offsets: %+.0f Hz, %+.0f Hz, %+.0f Hz\n', fc_offsets(1), fc_offsets(2), fc_offsets(3));
end


%% Helper functions
function [M, bits_per_symbol] = get_qam_params(mod_type)
    % Get QAM modulation parameters
    switch mod_type
        case '16QAM'
            M = 16;
            bits_per_symbol = 4;
        case '64QAM'
            M = 64;
            bits_per_symbol = 6;
        case '128QAM'
            M = 128;
            bits_per_symbol = 7;
        case '256QAM'
            M = 256;
            bits_per_symbol = 8;
        otherwise
            error('Unsupported modulation type: %s', mod_type);
    end
end

function save_dual_source_data(mixed_baseband, ideal_bb_signals, bit_data_1, bit_data_2, ...
                              samples_per_file, frame_length, num_files, snr, mod1_type, mod2_type, ...
                              bits_per_symbol1, bits_per_symbol2, symbols_per_frame)
    % Save dual-source data
    total_samples = length(mixed_baseband);
    
    for file_idx = 1:num_files
        % Calculate frame range for current file
        start_frame = (file_idx-1) * samples_per_file + 1;
        end_frame = file_idx * samples_per_file;
        start_sample = (start_frame-1)*frame_length + 1;
        end_sample = min(end_frame * frame_length, total_samples);
        
        % Extract signal segments for current file
        file_mixed = mixed_baseband(start_sample:end_sample);
        file_ideal_all = ideal_bb_signals(start_sample:end_sample, :);
        
        % Extract bit data
        file_bits_1 = bit_data_1((start_frame-1)*symbols_per_frame*bits_per_symbol1+1:...
                                end_frame*symbols_per_frame*bits_per_symbol1);
        file_bits_2 = bit_data_2((start_frame-1)*symbols_per_frame*bits_per_symbol2+1:...
                                end_frame*symbols_per_frame*bits_per_symbol2);
        
        % Reshape to frame structure
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 4)); % Dual-source, 2 channels (I,Q) per source
        % Frame processing
        for frame_idx = 1:samples_per_file
            sample_start = (frame_idx-1)*frame_length + 1;
            sample_end = frame_idx * frame_length;
            if sample_end > length(file_mixed)
                break;
            end
            
            % Mixed signal frame
            frame_data = file_mixed(sample_start:sample_end);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);
            
            % Ideal signal frame
            for src_idx = 1:2
                frame_ideal = file_ideal_all(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        % Save data
        save_path_mixed = sprintf('./%s_%s_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                mod1_type, mod2_type, file_idx, snr);
        save_path_target = sprintf('./%s_%s_Dataset_target_%d_SNR=%ddB.mat', ...
                                 mod1_type, mod2_type, file_idx, snr);
        
        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');
        
        % Save bit data
        save_path_bit_1 = sprintf('./%s_BitData_%d_SNR=%ddB_Source1.mat', ...
                                mod1_type, file_idx, snr);
        save_path_bit_2 = sprintf('./%s_BitData_%d_SNR=%ddB_Source2.mat', ...
                                mod2_type, file_idx, snr);
        
        save(save_path_bit_1, 'file_bits_1', '-v7.3');
        save(save_path_bit_2, 'file_bits_2', '-v7.3');
        
        fprintf('Dual-source file %d/%d saved (%s+%s)\n', file_idx, num_files, mod1_type, mod2_type);
    end
end

function save_tri_source_data(mixed_baseband, ideal_bb_signals, bit_data_1, bit_data_2, bit_data_3, ...
                             samples_per_file, frame_length, num_files, snr, mod1_type, mod2_type, mod3_type, ...
                             bits_per_symbol1, bits_per_symbol2, bits_per_symbol3, symbols_per_frame)
    % Save tri-source data
    total_samples = length(mixed_baseband);
    
    for file_idx = 1:num_files
        % Calculate frame range for current file
        start_frame = (file_idx-1) * samples_per_file + 1;
        end_frame = file_idx * samples_per_file;
        start_sample = (start_frame-1)*frame_length + 1;
        end_sample = min(end_frame * frame_length, total_samples);
        
        % Extract signal segments for current file
        file_mixed = mixed_baseband(start_sample:end_sample);
        file_ideal_all = ideal_bb_signals(start_sample:end_sample, :);
        
        % Extract bit data
        file_bits_1 = bit_data_1((start_frame-1)*symbols_per_frame*bits_per_symbol1+1:...
                                end_frame*symbols_per_frame*bits_per_symbol1);
        file_bits_2 = bit_data_2((start_frame-1)*symbols_per_frame*bits_per_symbol2+1:...
                                end_frame*symbols_per_frame*bits_per_symbol2);
        file_bits_3 = bit_data_3((start_frame-1)*symbols_per_frame*bits_per_symbol3+1:...
                                end_frame*symbols_per_frame*bits_per_symbol3);
        
        % Reshape to frame structure
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 6)); % Tri-source, 2 channels (I,Q) per source
        
        % Frame processing
        for frame_idx = 1:samples_per_file
            sample_start = (frame_idx-1)*frame_length + 1;
            sample_end = frame_idx * frame_length;
            if sample_end > length(file_mixed)
                break;
            end
            
            % Mixed signal frame
            frame_data = file_mixed(sample_start:sample_end);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);
            
            % Ideal signal frame
            for src_idx = 1:3
                frame_ideal = file_ideal_all(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        % Save data
        save_path_mixed = sprintf('./%s_%s_%s_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                mod1_type, mod2_type, mod3_type, file_idx, snr);
        save_path_target = sprintf('./%s_%s_%s_Dataset_target_%d_SNR=%ddB.mat', ...
                                 mod1_type, mod2_type, mod3_type, file_idx, snr);
        
        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');
        
        % Save bit data
        save_path_bit_1 = sprintf('./%s_BitData_%d_SNR=%ddB_Source1.mat', ...
                                mod1_type, file_idx, snr);
        save_path_bit_2 = sprintf('./%s_BitData_%d_SNR=%ddB_Source2.mat', ...
                                mod2_type, file_idx, snr);
        save_path_bit_3 = sprintf('./%s_BitData_%d_SNR=%ddB_Source3.mat', ...
                                mod3_type, file_idx, snr);
        
        save(save_path_bit_1, 'file_bits_1', '-v7.3');
        save(save_path_bit_2, 'file_bits_2', '-v7.3');
        save(save_path_bit_3, 'file_bits_3', '-v7.3');
        
        fprintf('Tri-source file %d/%d saved (%s+%s+%s)\n', file_idx, num_files, mod1_type, mod2_type, mod3_type);
    end
end