function GEN_8PSK_MultiSource()
    for snr = -10:4:30
        GEN_8PSK_MultiSource_i(2, snr, 1)
    end

    % for snr = -10:4:30
    %     GEN_8PSK_MultiSource_i(3, snr, 20)
    % end

    % for snr = -10:4:30
    %     GEN_8PSK_MultiSource_i(3, snr, 20)
    % end
        
end

function GEN_8PSK_MultiSource_i(num_sources, snr, num_files)
    % Multi-source 8PSK signal generation function
    % num_sources: Number of sources (2, 3, 4)
    % snr: Signal-to-noise ratio (dB)
    % num_files: Number of files
    
    %% Parameter validation
    if nargin < 1
        num_sources = 2;  % Default dual-source
    end
    if nargin < 2
        snr = 25;  % Default 25dB
    end
    
    if ~ismember(num_sources, [2, 3, 4])
        error('Number of sources must be 2, 3, or 4');
    end
    
    %% ========== Added: Non-ideal characteristics parameters ==========
    impaired = true;  % Set to true to enable non-ideal characteristics
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
    
    %% Basic parameters
    flo = 20e6;                 % Local oscillator frequency
    Fs_rf = 100e6;              % Sampling rate
    
    %% Root-raised cosine filter
    Fs_sps = 20;                % Samples per symbol
    alpha = 0.35;               % Roll-off factor
    span = 20;                   % Filter symbol span
    filterCoeffs = rcosdesign(alpha, span, Fs_sps, 'sqrt');
    
    %% Dataset parameters
    %num_files = 20;              % Number of files
    samples_per_file = 500;     % Number of frames per file
    frame_length = 4096;        % Number of points per frame
    symbols_per_frame = 205;    % Number of symbols per frame
    bits_per_symbol = 3;        % Number of bits per symbol
    total_frames = samples_per_file;
    total_samples = total_frames * frame_length;  % Total sampling points per file
    total_symbols = total_frames * symbols_per_frame; % Total symbols per file
    
    %% Delay parameters (reduce delay to avoid affecting BER)
    symbol_rate = 5e6;          % Symbol rate
    Tb = 1/symbol_rate;         % Symbol period
    % Reduce delay to 0.05*Tb to reduce impact on BER
    base_delay = 0.05 * Tb;
    delay_samples = round((0:num_sources-1) * base_delay * Fs_rf);
    
    %% Low-pass filter design
    rolloff = 0.35;
    cutoff_freq = symbol_rate * (1+rolloff)/2;
    normalized_cutoff = cutoff_freq/(Fs_rf/2);
    h_lpf = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
    
    %% Constellation mapping table
    constellation = [exp(1j * pi/8);       % dec 0 (000)
                 exp(1j * 3*pi/8);     % dec 1 (001)
                 exp(1j * 5*pi/8);     % dec 3 (011)
                 exp(1j * 7*pi/8);     % dec 2 (010)
                 exp(1j * 9*pi/8);     % dec 6 (110)
                 exp(1j * 11*pi/8);   % dec 7 (111)
                 exp(1j * 13*pi/8);    % dec 5 (101)
                 exp(1j * 15*pi/8)];    % dec 4 (100)   
    % Mapping table: natural binary -> constellation index (Gray code mapping)
    gray_map_array = [1, 2, 4, 3, 8, 7, 5, 6]; % Index mapping [n=0,1,2,3,4,5,6,7]
    
    %% ========== Added: Generate carrier frequency drift function ==========
    function phase_drift = generate_carrier_drift(t, fc_base, drift_params, file_seed)
        % Generate time-varying carrier frequency drift (as phase variation)
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
        
        % 3. Sinusoidal FM component (simulate periodic oscillator instability)
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
            % Current symbol's sampling point range
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
            
            % Apply fractional delay (use Lagrange interpolation)
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
                    % Positive delay: pad zeros in front
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed(1:end-abs(delay_int))];
                    else
                        % If delay is too large, just copy
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed];
                    end
                else
                    % Negative delay: pad zeros at end
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [segment_delayed(abs(delay_int)+1:end); zeros(abs(delay_int), 1)];
                    else
                        % If delay is too large, just copy
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
        
        % 1. Fast random variation (simulate AGC, PA nonlinearity, etc.)
        if amp_params.variation_std > 0
            % Generate band-limited Gaussian noise
            white_noise = randn(N, 1);
            % Design low-pass filter to limit bandwidth
            [b_lp, a_lp] = butter(4, amp_params.variation_bandwidth/(Fs/2));
            fast_variation = filter(b_lp, a_lp, white_noise);
            fast_variation = fast_variation / std(fast_variation) * amp_params.variation_std;
            amplitude_envelope = amplitude_envelope + fast_variation;
        end
        
        % 2. Slow fading (simulate multipath, obstruction, etc.)
        if amp_params.fade_depth > 0
            % Sinusoidal fading
            slow_fade = amp_params.fade_depth * sin(2*pi * amp_params.fade_rate * t + rand()*2*pi);
            amplitude_envelope = amplitude_envelope .* (1 + slow_fade);
        end
        
        % Ensure amplitude is positive and reasonable
        amplitude_envelope = max(0.5, min(1.5, amplitude_envelope));
        amplitude_envelope = amplitude_envelope(:);  % Column vector
    end
    
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
    
    %% ========================================================
    %% Generate complete signals file by file
    %% ========================================================
    
    for file_idx = 1:num_files
        fprintf('Generating file %d/%d\n', file_idx, num_files);
        
        %% Carrier frequency configuration (random frequency offset within ±700Hz for each file)
        fc_base = 20e6;  % Base carrier frequency 20MHz
        
        switch num_sources
            case 2
                % Generate two symmetric frequency offsets within ±700Hz
                offset_mag = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset_mag, offset_mag];
            case 3  
                % Generate one random value between -700 and 700Hz, plus 0Hz and symmetric value
                offset_mag = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset_mag, 0, offset_mag];
            case 4
                % Generate two random frequency offsets, maintaining symmetry
                offset1 = rand() * 700;  % Random value between 0 and 700Hz
                offset2 = rand() * 700;  % Random value between 0 and 700Hz
                fc_offsets = [-offset1, -offset2, offset2, offset1];
        end
        
        % Calculate actual carrier frequencies
        fc_array = fc_base + fc_offsets;
        
        %% ========== Modified: Initial phase for each file is uniformly distributed in 0 to π ==========
        initial_phases = rand(num_sources, 1) * pi;  % Random phase for each source in 0 to π
        
        % Generate global time axis (continuous time starting from 0)
        t_global = (0:total_samples-1)' / Fs_rf;
        
        %% 1. Generate bit stream and modulation signals (multi-source)
        total_bits = total_symbols * bits_per_symbol;
        
        % Store signals for each source
        rf_signals = zeros(length(t_global), num_sources);
        ideal_bb_signals = zeros(length(t_global), num_sources);
        bit_data_all = cell(num_sources, 1);
        
        for src_idx = 1:num_sources
            %% Generate bit stream
            bit_data_all{src_idx} = randi([0, 1], total_bits, 1, 'uint8');
            
            %% Modulate to 8PSK symbols
            symbol_indices = bi2de(reshape(bit_data_all{src_idx}, bits_per_symbol, [])', 'left-msb') + 1;
            % Use mapping table to convert to correct constellation index
            symbol_indices = gray_map_array(symbol_indices);
            s_complex = constellation(symbol_indices);
            
            %% Upsampling and pulse shaping
            s_upsampled = upsample(s_complex, Fs_sps);
            s_shaped = conv(s_upsampled, filterCoeffs, 'same');
            
            %% ========== Key modification: Save ideal signal for target ==========
            s_shaped_ideal = s_shaped;
            
            %% Apply symbol clock jitter
            if enable_timing_jitter
                num_symbols_total = length(symbol_indices);
                s_shaped = apply_timing_jitter(s_shaped, jitter_params, num_symbols_total, Fs_sps);
                % Re-align length
                if length(s_shaped) > length(t_global)
                    s_shaped = s_shaped(1:length(t_global));
                elseif length(s_shaped) < length(t_global)
                    s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
                end
                s_shaped_ideal = s_shaped_ideal(1:length(s_shaped));
            else
                % If no jitter, ensure ideal signal length matches global time axis
                if length(s_shaped_ideal) > length(t_global)
                    s_shaped_ideal = s_shaped_ideal(1:length(t_global));
                elseif length(s_shaped_ideal) < length(t_global)
                    s_shaped_ideal = [s_shaped_ideal; zeros(length(t_global) - length(s_shaped_ideal), 1)];
                end
            end
            
            %% Add delay
            if delay_samples(src_idx) > 0
                s_shaped = [zeros(delay_samples(src_idx), 1); s_shaped(1:end-delay_samples(src_idx))];
            end
            
            %% Ensure length consistency
            if length(s_shaped) > length(t_global)
                s_shaped = s_shaped(1:length(t_global));
            elseif length(s_shaped) < length(t_global)
                s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
            end
            
            %% Apply carrier frequency drift and amplitude variation
            if enable_carrier_drift
                phase_drift = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift + initial_phases(src_idx)));
            else
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            rf_signal = real(s_shaped .* carrier);
            
            %% Apply amplitude variation
            if enable_amplitude_variation
                amp_envelope = generate_amplitude_variation(t_global, amp_params, file_idx*200 + src_idx);
                rf_signal = rf_signal .* amp_envelope;
            end
            
            rf_signals(:, src_idx) = rf_signal;
            
            %% Generate ideal demodulation signal (label signal) - use ideal signal
            if enable_carrier_drift
                % If there is carrier drift, ideal signal should also have same drift
                phase_drift_ideal = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift_ideal + initial_phases(src_idx)));
            else
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            % Ensure ideal signal length matches time axis
            if length(s_shaped_ideal) ~= length(t_global)
                if length(s_shaped_ideal) > length(t_global)
                    s_shaped_ideal = s_shaped_ideal(1:length(t_global));
                else
                    s_shaped_ideal = [s_shaped_ideal; zeros(length(t_global) - length(s_shaped_ideal), 1)];
                end
            end
            
            rf_signal_ideal = real(s_shaped_ideal .* carrier_ideal);
            
            baseband_i = rf_signal_ideal .* cos(2*pi*flo*t_global);
            baseband_q = rf_signal_ideal .* (-sin(2*pi*flo*t_global));
            bb_i_filtered = conv(baseband_i, h_lpf, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf, 'same');
            ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
        end
        
        %% Mix signals and add noise
        rf_combined = sum(rf_signals, 2);
        rf_combined_noisy = awgn(rf_combined, snr, 'measured');
        
        %% Down-convert to baseband
        % Generate LO signal
        lo_i = cos(2*pi*flo*t_global);
        lo_q = sin(2*pi*flo*t_global);
        
        % Mix signal down-convert
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
        ideal_frames = single(zeros(samples_per_file, frame_length, 2*num_sources)); % 2 channels (I,Q) per source
        
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
            for src_idx = 1:num_sources
                frame_ideal = ideal_bb_signals(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        %% Save data
        save_path_mixed = sprintf('D:/My/SCBSS/2.0/Data_gen/dataset_8192/8PSK/%dSource_8PSK_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                num_sources, file_idx, snr);
        save_path_target = sprintf('D:/My/SCBSS/2.0/Data_gen/dataset_8192/8PSK/%dSource_8PSK_Dataset_target_%d_SNR=%ddB.mat', ...
                                 num_sources, file_idx, snr);
        
        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');
        
        % Save bit data (each source saved separately)
        for src_idx = 1:num_sources
            save_path_bit = sprintf('D:/My/SCBSS/2.0/Data_gen/dataset_8192/8PSK/bits/%dSource_8PSK_BitData_%d_SNR=%ddB_Source%d.mat', ...
                                  num_sources, file_idx, snr, src_idx);
            file_bits = bit_data_all{src_idx};
            save(save_path_bit, 'file_bits', '-v7.3');
        end
        
        fprintf('File %d/%d saved (%d sources)\n', file_idx, num_files, num_sources);
        fprintf('  Frequency offset: ');
        for i = 1:num_sources
            fprintf('%.1f Hz ', fc_offsets(i));
        end
        fprintf('\n');
        fprintf('  Initial phase: ');
        for i = 1:num_sources
            fprintf('%.3f rad ', initial_phases(i));
        end
        fprintf('\n');
    end
    
    %% Display configuration information
    fprintf('\n=== Dataset generation completed ===\n');
    fprintf('Number of sources: %d\n', num_sources);
    fprintf('SNR: %d dB\n', snr);
    fprintf('Frequency offset range: ±700 Hz\n');
    fprintf('Number of files: %d\n', num_files);
    
    disp('All files generated');
end