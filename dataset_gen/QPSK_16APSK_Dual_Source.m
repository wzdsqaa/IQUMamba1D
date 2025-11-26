function GEN_QPSK_16APSK_DualSource()
    % Generate QPSK+16APSK dual-source dataset with different SNR
    for snr = -10:4:30
        GEN_QPSK_16APSK_DualSource_i(snr, 20)
    end
end

function GEN_QPSK_16APSK_DualSource_i(snr, num_files)
    % QPSK+16APSK dual-source signal generation function
    % snr: Signal-to-noise ratio (dB)
    % num_files: Number of files
    
    %% Parameter validation
    if nargin < 1
        snr = 25;  % Default 25dB
    end
    if nargin < 2
        num_files = 20;  % Default 20 files
    end
    
    %% ========== Added: Non-ideal characteristics parameters ==========
    impaired = false;  % Set to true to enable non-ideal characteristics
    % 1. Carrier frequency drift parameters
    enable_carrier_drift = impaired;
    carrier_drift_rate = 50;            % Hz/s, carrier frequency drift rate (linear drift)
    carrier_drift_random_walk_std = 5;  % Hz, random walk standard deviation
    carrier_drift_type = 'combined';    % 'linear', 'random_walk', 'sinusoidal', 'combined'
    
    % Sinusoidal frequency modulation parameters (simulate oscillator instability)
    carrier_fm_amplitude = 20;          % Hz, FM amplitude
    carrier_fm_frequency = 2;           % Hz, FM frequency
    
    % 2. Symbol clock jitter parameters
    enable_timing_jitter = impaired;
    timing_jitter_rms = 0.015;          % RMS jitter as fraction of symbol period (1.5%)
    timing_jitter_type = 'gaussian';    % 'gaussian', 'uniform', 'colored'
    
    % 3. Amplitude variation parameters
    enable_amplitude_variation = impaired;
    amplitude_variation_std = 0.03;     % Amplitude variation standard deviation (3%)
    amplitude_variation_bandwidth = 50; % Variation bandwidth (Hz)
    amplitude_fade_depth = 0.1;         % Slow fading depth (10%)
    amplitude_fade_rate = 0.5;          % Slow fading rate (Hz)
    
    fprintf('=== Non-ideal characteristics configuration ===\n');
    fprintf('1. Carrier frequency drift:\n');
    fprintf('   - Drift rate: %d Hz/s\n', carrier_drift_rate);
    fprintf('   - Random walk: std=%d Hz\n', carrier_drift_random_walk_std);
    fprintf('   - FM amplitude: %d Hz @ %d Hz\n', carrier_fm_amplitude, carrier_fm_frequency);
    fprintf('2. Symbol clock jitter:\n');
    fprintf('   - RMS jitter: %.2f%% symbol period\n', timing_jitter_rms*100);
    fprintf('3. Amplitude variation:\n');
    fprintf('   - Fast variation: std=%.1f%%, BW=%d Hz\n', amplitude_variation_std*100, amplitude_variation_bandwidth);
    fprintf('   - Slow fading: depth=%.1f%%, rate=%.2f Hz\n', amplitude_fade_depth*100, amplitude_fade_rate);
    fprintf('\n');
    
    %% ========================================================
    %% Generate complete signals file by file
    %% ========================================================
    %% Prepare non-ideal characteristics parameter structure
    drift_params = struct('type', carrier_drift_type, ...
                          'rate', carrier_drift_rate, ...
                          'random_walk_std', carrier_drift_random_walk_std, ...
                          'fm_amplitude', carrier_fm_amplitude, ...
                          'fm_frequency', carrier_fm_frequency);
    
    jitter_params = struct('enable', enable_timing_jitter, ...
                           'rms', timing_jitter_rms, ...
                           'type', timing_jitter_type);
    
    amp_params = struct('variation_std', amplitude_variation_std, ...
                        'variation_bandwidth', amplitude_variation_bandwidth, ...
                        'fade_depth', amplitude_fade_depth, ...
                        'fade_rate', amplitude_fade_rate);

    for file_idx = 1:num_files
        fprintf('Generating file %d/%d\n', file_idx, num_files);
        
        %% Carrier frequency configuration (random frequency offset within 0 to 700Hz for each file)
        fc_base = 20e6;  % Base carrier frequency 20MHz
        
        % Generate two random frequency offsets within 0 to 700Hz
        offset1 = rand() * 700;  % Random value between 0 and 700Hz
        offset2 = rand() * 700;  % Random value between 0 and 700Hz
        fc_offsets = [offset1, -offset1];  % Dual-source: two random frequency offsets within 0 to 700Hz
        fc_array = fc_base + fc_offsets;  % Actual carrier frequencies
        
        %% Initial phase configuration (each source uniformly distributed in 0 to π for each file)
        initial_phases = rand(2, 1) * pi;  % Two sources with random phases in 0 to π
        
        %% Basic parameters
        flo = 20e6;                 % Local oscillator frequency
        Fs_rf = 100e6;              % Sampling rate
        
        %% Root-raised cosine filter
        Fs_sps = 20;                % Samples per symbol
        alpha = 0.35;               % Roll-off factor
        span = 20;                   % Filter symbol span
        filterCoeffs = rcosdesign(alpha, span, Fs_sps, 'sqrt');
        
        %% Dataset parameters
        samples_per_file = 500;     % Number of frames per file
        frame_length = 4096;        % Number of points per frame
        symbols_per_frame = 205;    % Number of symbols per frame
        bits_per_symbol_QPSK = 2;   % QPSK bits per symbol
        bits_per_symbol_16APSK = 4; % 16APSK bits per symbol
        total_frames = samples_per_file;
        total_samples = total_frames * frame_length;
        total_symbols = total_frames * symbols_per_frame;
        
        %% Delay parameters (reduce delay to avoid affecting BER)
        symbol_rate = 5e6;          % Symbol rate
        Tb = 1/symbol_rate;         % Symbol period
        base_delay = 0.05 * Tb;     % Base delay
        delay_samples = [0, round(base_delay * Fs_rf)]; % [Source1 delay, Source2 delay]
        
        %% Low-pass filter design
        rolloff = 0.35;
        cutoff_freq = symbol_rate * (1+rolloff)/2;
        normalized_cutoff = cutoff_freq/(Fs_rf/2);
        h_lpf = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
        
        %% Constellation mapping table
        % QPSK constellation
        constellation_QPSK = [ 
            exp(1j * pi/4);       % 00
            exp(1j * 3*pi/4);     % 01
            exp(1j * 7*pi/4);     % 10
            exp(1j * 5*pi/4)      % 11
        ];
        
        % 16APSK constellation
        r1 = 1;       % Inner ring radius
        r2 = 2.85;    % Outer ring radius
        
        constellation_16APSK = zeros(16, 1);
        % Outer ring points (indices 0~11)
        constellation_16APSK(1)  = r2 * (cos(3*pi/12)  + 1i*sin(3*pi/12));   % 0000
        constellation_16APSK(2)  = r2 * (cos(21*pi/12) + 1i*sin(21*pi/12));  % 0001
        constellation_16APSK(3)  = r2 * (cos(9*pi/12)  + 1i*sin(9*pi/12));   % 0010
        constellation_16APSK(4)  = r2 * (cos(15*pi/12) + 1i*sin(15*pi/12));  % 0011
        constellation_16APSK(5)  = r2 * (cos(pi/12)    + 1i*sin(pi/12));     % 0100
        constellation_16APSK(6)  = r2 * (cos(23*pi/12) + 1i*sin(23*pi/12));  % 0101
        constellation_16APSK(7)  = r2 * (cos(11*pi/12) + 1i*sin(11*pi/12));  % 0110
        constellation_16APSK(8)  = r2 * (cos(13*pi/12) + 1i*sin(13*pi/12));  % 0111
        constellation_16APSK(9)  = r2 * (cos(5*pi/12)  + 1i*sin(5*pi/12));   % 1000
        constellation_16APSK(10) = r2 * (cos(19*pi/12) + 1i*sin(19*pi/12));  % 1001
        constellation_16APSK(11) = r2 * (cos(7*pi/12)  + 1i*sin(7*pi/12));   % 1010
        constellation_16APSK(12) = r2 * (cos(17*pi/12) + 1i*sin(17*pi/12));  % 1011
        % Inner ring points (indices 12~15)
        constellation_16APSK(13) = r1 * (cos(pi/4)     + 1i*sin(pi/4));      % 1100
        constellation_16APSK(14) = r1 * (cos(7*pi/4)   + 1i*sin(7*pi/4));    % 1101
        constellation_16APSK(15) = r1 * (cos(3*pi/4)   + 1i*sin(3*pi/4));    % 1110
        constellation_16APSK(16) = r1 * (cos(5*pi/4)   + 1i*sin(5*pi/4));    % 1111
        
        %% Power normalization (reference original logic)
        QPSK_power = mean(abs(constellation_QPSK).^2);
        APSK_power_raw = (4*abs(r1)^2 + 12*abs(r2)^2)/16;
        scaling_factor = sqrt(QPSK_power / APSK_power_raw);
        constellation_16APSK = scaling_factor * constellation_16APSK;
        
        %% Generate time axis for current file
        t_global = (0:total_samples-1)' / Fs_rf;
        
        %% 1. Generate bit streams and modulation signals (dual-source)
        total_bits_QPSK = total_symbols * bits_per_symbol_QPSK;
        total_bits_16APSK = total_symbols * bits_per_symbol_16APSK;
        
        % Store dual-source signals
        rf_signals = zeros(length(t_global), 2);
        ideal_bb_signals = zeros(length(t_global), 2);
        
        % Generate bit data
        bit_data_QPSK = randi([0, 1], total_bits_QPSK, 1, 'uint8');
        bit_data_16APSK = randi([0, 1], total_bits_16APSK, 1, 'uint8');
        bit_data_all = {bit_data_QPSK, bit_data_16APSK};
        
        %% Source 1: QPSK modulation
        symbol_indices_1 = bi2de(reshape(bit_data_QPSK, bits_per_symbol_QPSK, [])', 'left-msb') + 1;
        s_complex_1 = constellation_QPSK(symbol_indices_1);
        
        % Upsampling and pulse shaping
        s_upsampled_1 = upsample(s_complex_1, Fs_sps);
        s_shaped_1 = conv(s_upsampled_1, filterCoeffs, 'same');
        
        %% ========== Key modification: Save ideal signal for target ==========
        s_shaped_1_ideal = s_shaped_1;
        
        %% Apply symbol clock jitter
        if enable_timing_jitter
            s_shaped_1 = apply_timing_jitter(s_shaped_1, jitter_params, length(symbol_indices_1), Fs_sps);
            % Realign length
            if length(s_shaped_1) > length(t_global)
                s_shaped_1 = s_shaped_1(1:length(t_global));
            elseif length(s_shaped_1) < length(t_global)
                s_shaped_1 = [s_shaped_1; zeros(length(t_global) - length(s_shaped_1), 1)];
            end
            s_shaped_1_ideal = s_shaped_1_ideal(1:length(s_shaped_1));
        else
            % If no jitter, ensure ideal signal length matches global time axis
            if length(s_shaped_1_ideal) > length(t_global)
                s_shaped_1_ideal = s_shaped_1_ideal(1:length(t_global));
            elseif length(s_shaped_1_ideal) < length(t_global)
                s_shaped_1_ideal = [s_shaped_1_ideal; zeros(length(t_global) - length(s_shaped_1_ideal), 1)];
            end
        end
        
        % Add delay
        if delay_samples(1) > 0
            s_shaped_1 = [zeros(delay_samples(1), 1); s_shaped_1(1:end-delay_samples(1))];
        end
        
        % Ensure length consistency
        if length(s_shaped_1) > length(t_global)
            s_shaped_1 = s_shaped_1(1:length(t_global));
        elseif length(s_shaped_1) < length(t_global)
            s_shaped_1 = [s_shaped_1; zeros(length(t_global) - length(s_shaped_1), 1)];
        end
        
        %% Source 2: 16APSK modulation
        symbol_indices_2 = bi2de(reshape(bit_data_16APSK, bits_per_symbol_16APSK, [])', 'left-msb') + 1;
        s_complex_2 = constellation_16APSK(symbol_indices_2);
        
        % Upsampling and pulse shaping
        s_upsampled_2 = upsample(s_complex_2, Fs_sps);
        s_shaped_2 = conv(s_upsampled_2, filterCoeffs, 'same');
        
        %% ========== Key modification: Save ideal signal for target ==========
        s_shaped_2_ideal = s_shaped_2;
        
        %% Apply symbol clock jitter
        if enable_timing_jitter
            s_shaped_2 = apply_timing_jitter(s_shaped_2, jitter_params, length(symbol_indices_2), Fs_sps);
            % Realign length
            if length(s_shaped_2) > length(t_global)
                s_shaped_2 = s_shaped_2(1:length(t_global));
            elseif length(s_shaped_2) < length(t_global)
                s_shaped_2 = [s_shaped_2; zeros(length(t_global) - length(s_shaped_2), 1)];
            end
            s_shaped_2_ideal = s_shaped_2_ideal(1:length(s_shaped_2));
        else
            % If no jitter, ensure ideal signal length matches global time axis
            if length(s_shaped_2_ideal) > length(t_global)
                s_shaped_2_ideal = s_shaped_2_ideal(1:length(t_global));
            elseif length(s_shaped_2_ideal) < length(t_global)
                s_shaped_2_ideal = [s_shaped_2_ideal; zeros(length(t_global) - length(s_shaped_2_ideal), 1)];
            end
        end
        
        % Add delay
        if delay_samples(2) > 0
            s_shaped_2 = [zeros(delay_samples(2), 1); s_shaped_2(1:end-delay_samples(2))];
        end
        
        % Ensure length consistency
        if length(s_shaped_2) > length(t_global)
            s_shaped_2 = s_shaped_2(1:length(t_global));
        elseif length(s_shaped_2) < length(t_global)
            s_shaped_2 = [s_shaped_2; zeros(length(t_global) - length(s_shaped_2), 1)];
        end
        
        %% Apply carrier frequency drift and amplitude variation
        if enable_carrier_drift
            phase_drift_1 = generate_carrier_drift(t_global, fc_array(1), drift_params, file_idx*100 + 1);
            phase_drift_2 = generate_carrier_drift(t_global, fc_array(2), drift_params, file_idx*100 + 2);
            carrier_1 = exp(1i*(2*pi*fc_array(1)*t_global + phase_drift_1 + initial_phases(1)));
            carrier_2 = exp(1i*(2*pi*fc_array(2)*t_global + phase_drift_2 + initial_phases(2)));
        else
            carrier_1 = exp(1i*(2*pi*fc_array(1)*t_global + initial_phases(1)));
            carrier_2 = exp(1i*(2*pi*fc_array(2)*t_global + initial_phases(2)));
        end
        
        rf_signal_1 = real(s_shaped_1 .* carrier_1);
        rf_signal_2 = real(s_shaped_2 .* carrier_2);
        
        %% Apply amplitude variation
        if enable_amplitude_variation
            amp_envelope_1 = generate_amplitude_variation(t_global, amp_params, file_idx*200 + 1);
            amp_envelope_2 = generate_amplitude_variation(t_global, amp_params, file_idx*200 + 2);
            rf_signal_1 = rf_signal_1 .* amp_envelope_1;
            rf_signal_2 = rf_signal_2 .* amp_envelope_2;
        end
        
        rf_signals(:, 1) = rf_signal_1;
        rf_signals(:, 2) = rf_signal_2;
        
        %% Ideal demodulation (label signal) - Use ideal signal
        lo_i = cos(2*pi*flo*t_global);
        lo_q = sin(2*pi*flo*t_global);
        
        for src_idx = 1:2
            if enable_carrier_drift
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            else
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            if src_idx == 1
                % Ensure ideal signal length matches time axis
                if length(s_shaped_1_ideal) ~= length(t_global)
                    if length(s_shaped_1_ideal) > length(t_global)
                        s_shaped_1_ideal_local = s_shaped_1_ideal(1:length(t_global));
                    else
                        s_shaped_1_ideal_local = [s_shaped_1_ideal; zeros(length(t_global) - length(s_shaped_1_ideal), 1)];
                    end
                else
                    s_shaped_1_ideal_local = s_shaped_1_ideal;
                end
                rf_signal_ideal = real(s_shaped_1_ideal_local .* carrier_ideal);
            else
                % Ensure ideal signal length matches time axis
                if length(s_shaped_2_ideal) ~= length(t_global)
                    if length(s_shaped_2_ideal) > length(t_global)
                        s_shaped_2_ideal_local = s_shaped_2_ideal(1:length(t_global));
                    else
                        s_shaped_2_ideal_local = [s_shaped_2_ideal; zeros(length(t_global) - length(s_shaped_2_ideal), 1)];
                    end
                else
                    s_shaped_2_ideal_local = s_shaped_2_ideal;
                end
                rf_signal_ideal = real(s_shaped_2_ideal_local .* carrier_ideal);
            end
            
            baseband_i = rf_signal_ideal .* lo_i;
            baseband_q = rf_signal_ideal .* (-lo_q);
            bb_i_filtered = conv(baseband_i, h_lpf, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf, 'same');
            ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
        end
        
        %% Mix signals and add noise
        rf_combined = sum(rf_signals, 2);
        rf_combined_noisy = awgn(rf_combined, snr, 'measured');
        
        %% Down-convert to baseband
        baseband_i = rf_combined_noisy .* lo_i;
        baseband_q = rf_combined_noisy .* (-lo_q);
        bb_i_filtered = conv(baseband_i, h_lpf, 'same');
        bb_q_filtered = conv(baseband_q, h_lpf, 'same');
        mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
        
        %% ========================================================
        %% File-level normalization
        %% ========================================================
        
        % Calculate file-level maximum amplitude
        file_max_mixed = max(abs(mixed_baseband));
        file_max_ideal = max(max(abs(ideal_bb_signals)));
        
        % Apply file-level normalization
        mixed_baseband = mixed_baseband / file_max_mixed;
        ideal_bb_signals = ideal_bb_signals / file_max_ideal;
        
        %% Split long signal into frames and save
        %% ========================================================
        
        % Reshape to frame structure
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 4)); % Dual-source, 2 channels (I,Q) per source
        
        % Frame processing
        for frame_idx = 1:samples_per_file
            % Calculate current frame position
            sample_start = (frame_idx-1)*frame_length + 1;
            sample_end = frame_idx * frame_length;
            if sample_end > length(mixed_baseband)
                break;
            end
            
            %% Mixed signal frame (directly use file normalized values)
            frame_data = mixed_baseband(sample_start:sample_end);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);
            
            %% Ideal signal frame (directly use file normalized values)
            for src_idx = 1:2
                frame_ideal = ideal_bb_signals(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        %% Save data
        save_path_mixed = sprintf('./QPSK_16APSK_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                file_idx, snr);
        save_path_target = sprintf('./QPSK_16APSK_Dataset_target_%d_SNR=%ddB.mat', ...
                                 file_idx, snr);
        
        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');
        
        % Save bit data for both sources separately
        save_path_bit_QPSK = sprintf('./QPSK_BitData_%d_SNR=%ddB_Source1.mat', ...
                                    file_idx, snr);
        save_path_bit_16APSK = sprintf('./16APSK_BitData_%d_SNR=%ddB_Source2.mat', ...
                                      file_idx, snr);
        
        save(save_path_bit_QPSK, 'bit_data_QPSK', '-v7.3');
        save(save_path_bit_16APSK, 'bit_data_16APSK', '-v7.3');
        
        fprintf('File %d/%d saved (QPSK+16APSK dual-source)\n', file_idx, num_files);
        fprintf('  Frequency offset: %.1f Hz, %.1f Hz\n', fc_offsets(1), fc_offsets(2));
        fprintf('  Initial phase: %.3f π, %.3f π\n', initial_phases(1)/pi, initial_phases(2)/pi);
    end
    
    %% Display configuration information
    fprintf('\n=== QPSK+16APSK dual-source dataset generation completed ===\n');
    fprintf('Modulation: QPSK + 16APSK\n');
    fprintf('SNR: %d dB\n', snr);
    fprintf('Frequency offset range: 0 to 700 Hz\n');
    fprintf('Number of files: %d\n', num_files);
    
    
    disp('All files generated');
end

%% ========== Added: Generate carrier frequency drift function ==========
function phase_drift = generate_carrier_drift(t, fc_base, drift_params, file_seed)
    % Generate time-varying carrier frequency drift (as phase change)
    % Input:
    %   t - Time vector
    %   fc_base - Base carrier frequency
    %   drift_params - Drift parameter structure
    % Output:
    %   phase_drift - Phase drift (radians)
    
    rng(file_seed * 7777); % Ensure reproducibility
    Fs = 1/mean(diff(t));
    N = length(t);
    
    phase_drift = zeros(size(t));
    
    % 1. Linear drift component
    if contains(drift_params.type, 'linear') || contains(drift_params.type, 'combined')
        linear_drift = drift_params.rate * t;  % Hz * s = Hz
        phase_drift = phase_drift + 2*pi * cumsum(linear_drift) / Fs;
    end
    
    % 2. Random walk component (simulate slow random drift like temperature changes)
    if contains(drift_params.type, 'random_walk') || contains(drift_params.type, 'combined')
        random_steps = randn(N, 1) * drift_params.random_walk_std;
        % Low-pass filter to make changes smoother
        [b_smooth, a_smooth] = butter(2, 10/(Fs/2));  % 10Hz cutoff
        random_walk = filter(b_smooth, a_smooth, random_steps);
        phase_drift = phase_drift + 2*pi * cumsum(random_walk) / Fs;
    end
    
    % 3. Sinusoidal frequency modulation component (simulate periodic oscillator instability)
    if contains(drift_params.type, 'sinusoidal') || contains(drift_params.type, 'combined')
        fm_signal = drift_params.fm_amplitude * sin(2*pi * drift_params.fm_frequency * t);
        % Integrate frequency offset to get phase
        phase_drift = phase_drift + 2*pi * cumsum(fm_signal) / Fs;
    end
    
    phase_drift = phase_drift(:);  % Ensure column vector
end

%% ========== Added: Generate symbol clock jitter function ==========
function jittered_signal = apply_timing_jitter(signal, jitter_params, num_symbols, sps)
    % Apply symbol clock jitter to signal
    % Input:
    %   signal - Input signal
    %   jitter_params - Jitter parameters
    %   num_symbols - Number of symbols
    %   sps - Samples per symbol
    
    if ~jitter_params.enable
        jittered_signal = signal;
        return;
    end
    
    % Generate clock jitter for each symbol (unit: samples)
    if strcmp(jitter_params.type, 'gaussian')
        % Gaussian white noise jitter
        jitter_samples = randn(num_symbols, 1) * jitter_params.rms * sps;
    elseif strcmp(jitter_params.type, 'uniform')
        % Uniform distribution jitter
        jitter_samples = (rand(num_symbols, 1) - 0.5) * 2 * jitter_params.rms * sps * sqrt(3);
    else % 'colored'
        % Colored noise jitter (more realistic, adjacent symbol jitter is correlated)
        white_jitter = randn(num_symbols, 1);
        % First-order low-pass filter
        alpha_jitter = 0.3;
        jitter_samples = filter(alpha_jitter, [1, -(1-alpha_jitter)], white_jitter);
        jitter_samples = jitter_samples * jitter_params.rms * sps / std(jitter_samples);
    end
    
    % Use fractional delay filter to implement time-varying delay
    jittered_signal = zeros(size(signal));
    
    for sym_idx = 1:num_symbols
        % Current symbol sample range
        start_idx = (sym_idx-1)*sps + 1;
        end_idx = min(sym_idx*sps, length(signal));
        
        if end_idx > length(signal)
            break;
        end
        
        % Extract current segment signal, add boundary check
        segment_start = max(1, start_idx-10);
        segment_end = min(length(signal), end_idx+10);
        
        % Ensure segment length is sufficient
        if segment_end >= segment_start
            segment = signal(segment_start:segment_end);
        else
            % If index is invalid, copy original signal segment
            jittered_signal(start_idx:end_idx) = signal(start_idx:end_idx);
            continue;
        end
        
        % Apply fractional delay (using Lagrange interpolation)
        delay_frac = jitter_samples(sym_idx);
        delay_int = floor(delay_frac);
        delay_frac_part = delay_frac - delay_int;
        
        % Third-order Lagrange interpolation
        if abs(delay_frac_part) > 0.001 && length(segment) >= 4
            segment_delayed = lagrange_interp(segment, delay_frac_part);
        else
            segment_delayed = segment;
        end
        
        % Integer delay
        if delay_int ~= 0
            if delay_int > 0
                % Positive delay: pad with zeros at front
                if length(segment_delayed) > abs(delay_int)
                    segment_delayed = [zeros(abs(delay_int), 1); segment_delayed(1:end-abs(delay_int))];
                else
                    % If delay is too large, directly copy
                    segment_delayed = [zeros(abs(delay_int), 1); segment_delayed];
                end
            else
                % Negative delay: pad with zeros at back
                if length(segment_delayed) > abs(delay_int)
                    segment_delayed = [segment_delayed(abs(delay_int)+1:end); zeros(abs(delay_int), 1)];
                else
                    % If delay is too large, directly copy
                    segment_delayed = [segment_delayed; zeros(abs(delay_int), 1)];
                end
            end
        end
        
        % Extract valid part, add boundary check
        valid_start = max(1, 11);
        valid_end = min(length(segment_delayed), valid_start + (end_idx - start_idx));
        
        if valid_end <= length(segment_delayed) && valid_start <= length(segment_delayed)
            % Ensure index is valid
            actual_length = min(end_idx - start_idx + 1, valid_end - valid_start + 1);
            if actual_length > 0
                jittered_signal(start_idx:start_idx+actual_length-1) = segment_delayed(valid_start:valid_start+actual_length-1);
            end
        else
            % If interpolation fails, use original signal
            jittered_signal(start_idx:end_idx) = signal(start_idx:end_idx);
        end
    end
end

%% ========== Added: Lagrange interpolation function ==========
function y_interp = lagrange_interp(y, delay_frac)
    % Third-order Lagrange fractional delay interpolation
    N = length(y);
    y_interp = zeros(N, 1);
    
    % Check if input length is sufficient
    if N < 4
        y_interp = y;
        return;
    end
    
    for n = 3:N-2
        % Use 4 points for third-order interpolation
        d = delay_frac;
        y_interp(n) = y(n-1) * (-d)*(d-1)*(d-2)/6 + ...
                      y(n)   * (d+1)*(d-1)*(d-2)/2 + ...
                      y(n+1) * (d+1)*d*(d-2)/(-2) + ...
                      y(n+2) * (d+1)*d*(d-1)/6;
    end
    
    % Boundary processing - only fill boundaries when original signal is long enough
    if N >= 4
        y_interp(1:2) = y(1:2);
        y_interp(N-1:N) = y(N-1:N);
    else
        y_interp = y;
    end
end

%% ========== Added: Generate amplitude variation function ==========
function amplitude_envelope = generate_amplitude_variation(t, amp_params, file_seed)
    % Generate time-varying amplitude envelope
    % Input:
    %   t - Time vector
    %   amp_params - Amplitude variation parameters
    % Output:
    %   amplitude_envelope - Normalized amplitude envelope
    
    rng(file_seed * 8888);
    Fs = 1/mean(diff(t));
    N = length(t);
    
    amplitude_envelope = ones(N, 1);
    
    % 1. Fast random variation (simulate AGC, amplifier nonlinearity, etc.)
    if amp_params.variation_std > 0
        % Generate band-limited Gaussian noise
        white_noise = randn(N, 1);
        % Design low-pass filter to limit bandwidth
        [b_lp, a_lp] = butter(4, amp_params.variation_bandwidth/(Fs/2));
        fast_variation = filter(b_lp, a_lp, white_noise);
        fast_variation = fast_variation / std(fast_variation) * amp_params.variation_std;
        amplitude_envelope = amplitude_envelope + fast_variation;
    end
    
    % 2. Slow fading (simulate multipath, blocking, etc.)
    if amp_params.fade_depth > 0
        % Sinusoidal fading
        slow_fade = amp_params.fade_depth * sin(2*pi * amp_params.fade_rate * t + rand()*2*pi);
        amplitude_envelope = amplitude_envelope .* (1 + slow_fade);
    end
    
    % Ensure amplitude is positive and reasonable
    amplitude_envelope = max(0.5, min(1.5, amplitude_envelope));
    amplitude_envelope = amplitude_envelope(:);  % Column vector
end