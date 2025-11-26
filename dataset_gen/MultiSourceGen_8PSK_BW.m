function MultiSourceGen_8PSK_BW()
    for snr = -10:4:30
        MultiSourceGen_i(2, snr, 20)
    end  
end


function MultiSourceGen_i(num_sources, snr, num_files)
    %% Multi-source 8PSK signal generation function (supports 2, 3, or 4 sources)
    % num_sources: number of sources (2, 3, 4)
    % snr: signal-to-noise ratio (dB)
    
    %% Parameter validation
    if nargin < 1
        num_sources = 2;  % default: dual-source
    end
    if nargin < 2
        snr = 24;  % default: 24 dB
    end
    
    if ~ismember(num_sources, [2, 3, 4])
        error('Number of sources must be 2, 3, or 4');
    end
    
    %% ========== New: Non-ideal characteristic parameters ==========
    impaired = false;  % Set to true to enable non-ideal effects
    % 1. Carrier frequency drift parameters
    enable_carrier_drift = impaired;
    carrier_drift_rate = 50;            % Hz/s, linear drift rate
    carrier_drift_random_walk_std = 5;  % Hz, standard deviation of random walk
    carrier_drift_type = 'combined';    % 'linear', 'random_walk', 'sinusoidal', 'combined'
    
    % Sinusoidal FM parameters (to simulate oscillator instability)
    carrier_fm_amplitude = 20;          % Hz, FM amplitude
    carrier_fm_frequency = 2;           % Hz, FM frequency
    
    % 2. Symbol timing jitter parameters
    enable_timing_jitter = impaired;
    timing_jitter_rms = 0.015;          % RMS jitter as fraction of symbol period (1.5%)
    timing_jitter_type = 'gaussian';    % 'gaussian', 'uniform', 'colored'
    
    % 3. Amplitude variation parameters
    enable_amplitude_variation = impaired;
    amplitude_variation_std = 0.03;     % std of amplitude variation (3%)
    amplitude_variation_bandwidth = 50; % bandwidth of fast variation (Hz)
    amplitude_fade_depth = 0.1;         % slow fading depth (10%)
    amplitude_fade_rate = 0.5;          % slow fading rate (Hz)
    
    fprintf('=== Non-ideal Characteristics Configuration ===\n');
    fprintf('1. Carrier Frequency Drift:\n');
    fprintf('   - Drift rate: %d Hz/s\n', carrier_drift_rate);
    fprintf('   - Random walk std: %d Hz\n', carrier_drift_random_walk_std);
    fprintf('   - FM modulation: %d Hz amplitude @ %d Hz\n', carrier_fm_amplitude, carrier_fm_frequency);
    fprintf('2. Symbol Timing Jitter:\n');
    fprintf('   - RMS jitter: %.2f%% of symbol period\n', timing_jitter_rms*100);
    fprintf('3. Amplitude Variation:\n');
    fprintf('   - Fast variation: std=%.1f%%, BW=%d Hz\n', amplitude_variation_std*100, amplitude_variation_bandwidth);
    fprintf('   - Slow fading: depth=%.1f%%, rate=%.2f Hz\n', amplitude_fade_depth*100, amplitude_fade_rate);
    fprintf('\n');
    
    %% ========================================================
    %% Generate full signals file-by-file
    %% ========================================================
    
    for file_idx = 1:num_files
        fprintf('Generating file %d/%d\n', file_idx, num_files);
        
        %% Basic parameter configuration
        fc_base = 10.000125e6;       % Base carrier frequency (LO reference)
        flo = 10.000125e6;           % Local oscillator frequency
        Fs_rf = 50e6;                % Sampling rate
        
        %% Carrier frequency configuration
        % Dual-source: symmetric offsets ±offset, where offset ∈ [0, 700] Hz
        if num_sources == 2
            offset_mag = rand() * 700;  % random value between 0 and 700 Hz
            fc_offsets = [-offset_mag, offset_mag];  % symmetric offsets
        else
            % Keep original offset design for 3/4 sources
            switch num_sources
                case 3  
                    % Three sources: -750Hz, 0Hz, +750Hz
                    fc_offsets = [-750, 0, 750];
                case 4
                    % Four sources: -1125Hz, -375Hz, +375Hz, +1125Hz  
                    fc_offsets = [-1125, -375, 375, 1125];
            end
        end
        
        % Compute actual carrier frequencies
        fc_array = fc_base + fc_offsets;
        
        %% Symbol rate configuration (different for each source)
        switch num_sources
            case 2
                symbol_rates = [5e6, 2.5e6];           % 5 MHz, 2.5 MHz
                Fs_sps_array = [10, 20];               % samples per symbol
                symbols_per_frame = [410, 205];        % symbols per frame
            case 3
                symbol_rates = [5e6, 3.5e6, 2e6];     % 5 MHz, 3.5 MHz, 2 MHz (linearly decreasing)
                Fs_sps_array = [10, 14, 25];          % samples per symbol
                symbols_per_frame = [410, 287, 164];   % symbols per frame, proportional to symbol rates
            case 4
                symbol_rates = [5e6, 4e6, 3e6, 2e6];  % 5 MHz, 4 MHz, 3 MHz, 2 MHz (linearly decreasing)
                Fs_sps_array = [10, 12, 17, 25];      % samples per symbol
                symbols_per_frame = [410, 328, 246, 164]; % symbols per frame, proportional to symbol rates
        end
        
        %% Initial phase configuration (each source: uniform in [0, π])
        initial_phases = rand(num_sources, 1) * pi;  % random phases in [0, π]
        
        %% Root-raised-cosine (RRC) filters (design per source)
        alpha = 0.35;                % roll-off factor
        span = 20;                   % filter span in symbols
        filterCoeffs = cell(num_sources, 1);
        for i = 1:num_sources
            filterCoeffs{i} = rcosdesign(alpha, span, Fs_sps_array(i), 'sqrt');
        end
        
        %% Dataset parameters
        samples_per_file = 250;      % number of frames per file
        frame_length = 4096;         % samples per frame
        bits_per_symbol = 3;         % bits per symbol (8-PSK)
        total_frames = samples_per_file;
        total_samples = total_frames * frame_length;
        
        % Compute total symbols and bits
        total_symbols = zeros(num_sources, 1);
        total_bits = zeros(num_sources, 1);
        for i = 1:num_sources
            total_symbols(i) = total_frames * symbols_per_frame(i);
            total_bits(i) = total_symbols(i) * bits_per_symbol;
        end
        
        %% Delay configuration (based on minimum symbol period)
        min_symbol_rate = min(symbol_rates);
        Tb_min = 1/min_symbol_rate;
        base_delay = 0.3 * Tb_min;
        delay_samples = zeros(num_sources, 1);
        for i = 1:num_sources
            delay_samples(i) = round((i-1) * base_delay * Fs_rf);
        end
        
        %% Low-pass filter design (for each source + mixed signal)
        rolloff = 0.35;
        h_lpf = cell(num_sources + 1, 1);  % +1 for mixed signal
        
        % Mixed signal: use highest bandwidth
        max_symbol_rate = max(symbol_rates);
        cutoff_freq_mixed = max_symbol_rate * (1+rolloff)/2;
        normalized_cutoff_mixed = cutoff_freq_mixed/(Fs_rf/2);
        h_lpf{1} = fir1(127, normalized_cutoff_mixed, 'low', kaiser(128, 5));
        
        % Per-source LPF
        for i = 1:num_sources
            cutoff_freq = symbol_rates(i) * (1+rolloff)/2;
            normalized_cutoff = cutoff_freq/(Fs_rf/2);
            h_lpf{i+1} = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));
        end
        
        %% Constellation mapping table (8-PSK)
        constellation = single([...
            0.924 + 1i*0.383;   % 000
            0.383 + 1i*0.924;   % 001
            -0.383 + 1i*0.924;  % 011
            -0.924 + 1i*0.383;  % 010
            -0.924 - 1i*0.383;  % 110
            -0.383 - 1i*0.924;  % 111
            0.383 - 1i*0.924;   % 101
            0.924 - 1i*0.383    % 100
        ]);
        % Gray coding mapping
        gray_map_array = [1, 2, 4, 3, 8, 7, 5, 6];
        
        %% Prepare non-ideal parameter structures
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
        
        %% Generate global time axis for current file
        t_global = (0:total_samples-1)' / Fs_rf;
        
        %% Generate individual source signals
        rf_signals = zeros(length(t_global), num_sources);
        ideal_bb_signals = zeros(length(t_global), num_sources);
        bit_data_all = cell(num_sources, 1);
        
        for src_idx = 1:num_sources
            fprintf('Generating signal %d/%d (symbol rate: %.2f MHz)...\n', src_idx, num_sources, symbol_rates(src_idx)/1e6);
            
            %% Generate bit stream
            bit_data_all{src_idx} = randi([0, 1], total_bits(src_idx), 1, 'uint8');
            
            %% Modulate to 8-PSK symbols
            symbol_indices = bi2de(reshape(bit_data_all{src_idx}, bits_per_symbol, [])', 'left-msb') + 1;
            symbol_indices = gray_map_array(symbol_indices);
            s_complex = constellation(symbol_indices);
            
            %% Upsample and pulse-shaping
            s_upsampled = upsample(s_complex, Fs_sps_array(src_idx));
            s_shaped = conv(s_upsampled, filterCoeffs{src_idx}, 'same');
            
            %% ========== Key modification: preserve ideal signal for ground truth ==========
            s_shaped_ideal = s_shaped;
            
            %% Apply symbol timing jitter
            if enable_timing_jitter
                s_shaped = apply_timing_jitter(s_shaped, jitter_params, length(symbol_indices), Fs_sps_array(src_idx));
                % Re-align lengths
                if length(s_shaped) > length(t_global)
                    s_shaped = s_shaped(1:length(t_global));
                elseif length(s_shaped) < length(t_global)
                    s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
                end
                s_shaped_ideal = s_shaped_ideal(1:length(s_shaped));
            end
            
            %% Add delay
            if delay_samples(src_idx) > 0
                s_shaped = [zeros(delay_samples(src_idx), 1); s_shaped(1:end-delay_samples(src_idx))];
            end
            
            %% Ensure consistent length
            if length(s_shaped) > length(t_global)
                s_shaped = s_shaped(1:length(t_global));
            elseif length(s_shaped) < length(t_global)
                s_shaped = [s_shaped; zeros(length(t_global) - length(s_shaped), 1)];
            end
            
            %% Apply carrier drift and amplitude variation
            if enable_carrier_drift
                phase_drift = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift + initial_phases(src_idx)));
            else
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end
            
            % Ensure carrier has the same length as s_shaped
            if length(carrier) > length(s_shaped)
                carrier = carrier(1:length(s_shaped));
            elseif length(carrier) < length(s_shaped)
                carrier = [carrier; carrier(end)*ones(length(s_shaped)-length(carrier), 1)];
            end
            
            rf_signal = real(s_shaped .* carrier);
            
            %% Apply amplitude variation
            if enable_amplitude_variation
                amp_envelope = generate_amplitude_variation(t_global, amp_params, file_idx*200 + src_idx);
                % Ensure amplitude envelope has the same length as rf_signal
                if length(amp_envelope) > length(rf_signal)
                    amp_envelope = amp_envelope(1:length(rf_signal));
                elseif length(amp_envelope) < length(rf_signal)
                    amp_envelope = [amp_envelope; amp_envelope(end)*ones(length(rf_signal)-length(amp_envelope), 1)];
                end
                rf_signal = rf_signal .* amp_envelope;
            end
            
            % Ensure rf_signal has the correct length for the output array
            if length(rf_signal) > size(rf_signals, 1)
                rf_signal = rf_signal(1:size(rf_signals, 1));
            elseif length(rf_signal) < size(rf_signals, 1)
                rf_signal = [rf_signal; zeros(size(rf_signals, 1)-length(rf_signal), 1)];
            end
            
            rf_signals(:, src_idx) = rf_signal;
            
            %% Ideal demodulation (for target labels) — using ideal signal
            carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            
            % Ensure s_shaped_ideal and carrier_ideal have matching lengths
            if length(s_shaped_ideal) > length(carrier_ideal)
                s_shaped_ideal = s_shaped_ideal(1:length(carrier_ideal));
            elseif length(s_shaped_ideal) < length(carrier_ideal)
                s_shaped_ideal = [s_shaped_ideal; zeros(length(carrier_ideal)-length(s_shaped_ideal), 1)];
            end
            
            rf_signal_ideal = real(s_shaped_ideal .* carrier_ideal);
            
            % Ensure rf_signal_ideal has the correct length
            if length(rf_signal_ideal) > size(ideal_bb_signals, 1)
                rf_signal_ideal = rf_signal_ideal(1:size(ideal_bb_signals, 1));
            elseif length(rf_signal_ideal) < size(ideal_bb_signals, 1)
                rf_signal_ideal = [rf_signal_ideal; zeros(size(ideal_bb_signals, 1)-length(rf_signal_ideal), 1)];
            end
            
            baseband_i = rf_signal_ideal .* cos(2*pi*flo*t_global(1:length(rf_signal_ideal)));
            baseband_q = rf_signal_ideal .* (-sin(2*pi*flo*t_global(1:length(rf_signal_ideal))));
            bb_i_filtered = conv(baseband_i, h_lpf{src_idx+1}, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf{src_idx+1}, 'same');
            
            % Ensure the ideal baseband signal has the correct size
            if length(bb_i_filtered) > size(ideal_bb_signals, 1)
                ideal_bb_signals(:, src_idx) = complex(bb_i_filtered(1:size(ideal_bb_signals, 1)), bb_q_filtered(1:size(ideal_bb_signals, 1)));
            else
                ideal_bb_signals(1:length(bb_i_filtered), src_idx) = complex(bb_i_filtered, bb_q_filtered);
            end
        end
        
        %% Combine signals and add noise
        rf_combined = sum(rf_signals, 2);
        rf_combined_noisy = awgn(rf_combined, snr, 'measured');
        
        %% Down-convert to baseband
        lo_i = cos(2*pi*flo*t_global);
        lo_q = sin(2*pi*flo*t_global);
        
        baseband_i = rf_combined_noisy .* lo_i;
        baseband_q = rf_combined_noisy .* (-lo_q);
        bb_i_filtered = conv(baseband_i, h_lpf{1}, 'same');
        bb_q_filtered = conv(baseband_q, h_lpf{1}, 'same');
        mixed_baseband = complex(bb_i_filtered, bb_q_filtered);
        
        %% ========================================================
        %% File-level normalization
        %% ========================================================
        
        file_max_mixed = max(abs(mixed_baseband));
        file_max_ideal = max(max(abs(ideal_bb_signals)));
        
        mixed_baseband = mixed_baseband / file_max_mixed;
        ideal_bb_signals = ideal_bb_signals / file_max_ideal;
        
        %% ========================================================
        %% Split signals into frames and save
        %% ========================================================
        
        % Reshape into frame format
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 2*num_sources));
        
        % Frame processing
        for frame_idx = 1:samples_per_file
            sample_start = (frame_idx-1)*frame_length + 1;
            sample_end = frame_idx * frame_length;
            if sample_end > length(mixed_baseband)
                break;
            end
            
            %% Mixed signal frame
            frame_data = mixed_baseband(sample_start:sample_end);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);
            
            %% Ideal signal frames
            for src_idx = 1:num_sources
                frame_ideal = ideal_bb_signals(sample_start:sample_end, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);  % I channel
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);    % Q channel
            end
        end
        
        %% Save data
        save_path_mixed = sprintf('./%dSource_8PSK_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                num_sources, file_idx, snr);
        save_path_target = sprintf('./%dSource_8PSK_Dataset_target_%d_SNR=%ddB.mat', ...
                                 num_sources, file_idx, snr);
        
        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');
        
        % Save bit data (per source)
        for src_idx = 1:num_sources
            save_path_bit = sprintf('./%dSource_8PSK_BitData_%d_SNR=%ddB_Source%d.mat', ...
                                  num_sources, file_idx, snr, src_idx);
            file_bits = bit_data_all{src_idx};
            save(save_path_bit, 'file_bits', '-v7.3');
        end
        
        fprintf('File %d/%d saved (%d sources, SNR=%ddB)\n', file_idx, num_files, num_sources, snr);
        fprintf('  Frequency offsets: %.1f Hz, %.1f Hz\n', fc_offsets(1), fc_offsets(2));
        fprintf('  Initial phases: %.3f π, %.3f π\n', initial_phases(1)/pi, initial_phases(2)/pi);
    end
    
    %% Display configuration summary
    fprintf('\n=== Dataset Generation Complete ===\n');
    fprintf('Number of sources: %d\n', num_sources);
    fprintf('SNR: %d dB\n', snr);
    fprintf('Carrier frequencies: ');
    for i = 1:num_sources
        fprintf('%.6f MHz ', fc_array(i)/1e6);
    end
    fprintf('\n');
    fprintf('Symbol rates: ');
    for i = 1:num_sources
        fprintf('%.2f MHz ', symbol_rates(i)/1e6);
    end
    fprintf('\n');
    fprintf('Frequency offsets: ');
    for i = 1:num_sources
        fprintf('%+.0f Hz ', fc_offsets(i));
    end
    fprintf('\n');
    fprintf('Symbols per frame: ');
    for i = 1:num_sources
        fprintf('%d ', symbols_per_frame(i));
    end
    fprintf('\n');
    fprintf('Bits per frame: ');
    for i = 1:num_sources
        fprintf('%d ', symbols_per_frame(i)*bits_per_symbol);
    end
    fprintf('\n');
    fprintf('Delays: ');
    for i = 1:num_sources
        fprintf('%.1f ns ', delay_samples(i)/Fs_rf*1e9);
    end
    fprintf('\n');

    
    disp('All files generated successfully');
    
    %% ========== Local function definitions (must be at end!) ==========
    function phase_drift = generate_carrier_drift(t, fc_base, drift_params, file_seed)
        % Generate time-varying carrier frequency drift (as phase deviation)
        % Inputs:
        %   t           — time vector
        %   fc_base     — base carrier frequency
        %   drift_params— drift configuration struct
        %   file_seed   — random seed for reproducibility
        % Output:
        %   phase_drift — phase drift in radians
        
        rng(file_seed * 7777); % ensure reproducibility
        Fs = 1/mean(diff(t));
        N = length(t);
        
        phase_drift = zeros(size(t));
        
        % 1. Linear drift component
        if contains(drift_params.type, 'linear') || contains(drift_params.type, 'combined')
            linear_drift = drift_params.rate * t;  % Hz * s = Hz deviation
            phase_drift = phase_drift + 2*pi * cumsum(linear_drift) / Fs;
        end
        
        % 2. Random walk component (e.g., due to temperature drift)
        if contains(drift_params.type, 'random_walk') || contains(drift_params.type, 'combined')
            random_steps = randn(N, 1) * drift_params.random_walk_std;
            % Smooth via low-pass filtering
            [b_smooth, a_smooth] = butter(2, 10/(Fs/2));  % 10-Hz cutoff
            random_walk = filter(b_smooth, a_smooth, random_steps);
            phase_drift = phase_drift + 2*pi * cumsum(random_walk) / Fs;
        end
        
        % 3. Sinusoidal FM component (periodic oscillator instability)
        if contains(drift_params.type, 'sinusoidal') || contains(drift_params.type, 'combined')
            fm_signal = drift_params.fm_amplitude * sin(2*pi * drift_params.fm_frequency * t);
            % Integrate frequency deviation to get phase
            phase_drift = phase_drift + 2*pi * cumsum(fm_signal) / Fs;
        end
        
        phase_drift = phase_drift(:);  % ensure column vector
    end
    
    function jittered_signal = apply_timing_jitter(signal, jitter_params, num_symbols, sps)
        % Apply symbol clock jitter to signal
        % Inputs:
        %   signal          — input complex baseband signal
        %   jitter_params   — jitter configuration struct
        %   num_symbols     — number of symbols
        %   sps             — samples per symbol
        
        if ~jitter_params.enable
            jittered_signal = signal;
            return;
        end
        
        % Generate jitter per symbol (in samples)
        if strcmp(jitter_params.type, 'gaussian')
            % Gaussian white jitter
            jitter_samples = randn(num_symbols, 1) * jitter_params.rms * sps;
        elseif strcmp(jitter_params.type, 'uniform')
            % Uniform jitter
            jitter_samples = (rand(num_symbols, 1) - 0.5) * 2 * jitter_params.rms * sps * sqrt(3);
        else % 'colored'
            % Colored jitter (more realistic, correlated between symbols)
            white_jitter = randn(num_symbols, 1);
            % First-order low-pass filter
            alpha_jitter = 0.3;
            jitter_samples = filter(alpha_jitter, [1, -(1-alpha_jitter)], white_jitter);
            jitter_samples = jitter_samples * jitter_params.rms * sps / std(jitter_samples);
        end
        
        % Use fractional delay filters to apply time-varying delay
        jittered_signal = zeros(size(signal));
        
        for sym_idx = 1:num_symbols
            % Sampling indices for current symbol
            start_idx = (sym_idx-1)*sps + 1;
            end_idx = min(sym_idx*sps, length(signal));
            
            if end_idx > length(signal)
                break;
            end
            
            % Extract signal segment with margin, with bounds checking
            segment_start = max(1, start_idx-10);
            segment_end = min(length(signal), end_idx+10);
            
            if segment_end >= segment_start
                segment = signal(segment_start:segment_end);
            else
                % Fallback: copy original segment if extraction fails
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
            
            % Apply integer delay
            if delay_int ~= 0
                if delay_int > 0
                    % Positive delay: prepend zeros
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed(1:end-abs(delay_int))];
                    else
                        segment_delayed = [zeros(abs(delay_int), 1); segment_delayed];
                    end
                else
                    % Negative delay: append zeros
                    if length(segment_delayed) > abs(delay_int)
                        segment_delayed = [segment_delayed(abs(delay_int)+1:end); zeros(abs(delay_int), 1)];
                    else
                        segment_delayed = [segment_delayed; zeros(abs(delay_int), 1)];
                    end
                end
            end
            
            % Extract valid output region with bounds safety
            valid_start = max(1, 11);
            valid_end = min(length(segment_delayed), valid_start + (end_idx - start_idx));
            
            if valid_end <= length(segment_delayed) && valid_start <= length(segment_delayed)
                actual_length = min(end_idx - start_idx + 1, valid_end - valid_start + 1);
                if actual_length > 0
                    jittered_signal(start_idx:start_idx+actual_length-1) = segment_delayed(valid_start:valid_start+actual_length-1);
                end
            else
                % Fallback to original if interpolation fails
                jittered_signal(start_idx:end_idx) = signal(start_idx:end_idx);
            end
        end
    end
    
    function y_interp = lagrange_interp(y, delay_frac)
        % Third-order Lagrange fractional delay interpolation
        N = length(y);
        y_interp = zeros(N, 1);
        
        % Safety check
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
        
        % Boundary handling
        if N >= 4
            y_interp(1:2) = y(1:2);
            y_interp(N-1:N) = y(N-1:N);
        else
            y_interp = y;
        end
    end
    
    function amplitude_envelope = generate_amplitude_variation(t, amp_params, file_seed)
        % Generate time-varying amplitude envelope
        % Inputs:
        %   t           — time vector
        %   amp_params  — amplitude variation parameters
        %   file_seed   — random seed for reproducibility
        % Output:
        %   amplitude_envelope — normalized (≈1) amplitude scaling factor
        
        rng(file_seed * 8888);
        Fs = 1/mean(diff(t));
        N = length(t);
        
        amplitude_envelope = ones(N, 1);
        
        % 1. Fast random variation (e.g., AGC, PA nonlinearity)
        if amp_params.variation_std > 0
            white_noise = randn(N, 1);
            % Band-limited via low-pass filter
            [b_lp, a_lp] = butter(4, amp_params.variation_bandwidth/(Fs/2));
            fast_variation = filter(b_lp, a_lp, white_noise);
            fast_variation = fast_variation / std(fast_variation) * amp_params.variation_std;
            amplitude_envelope = amplitude_envelope + fast_variation;
        end
        
        % 2. Slow fading (e.g., multipath, shadowing)
        if amp_params.fade_depth > 0
            slow_fade = amp_params.fade_depth * sin(2*pi * amp_params.fade_rate * t + rand()*2*pi);
            amplitude_envelope = amplitude_envelope .* (1 + slow_fade);
        end
        
        % Clamp to reasonable positive range
        amplitude_envelope = max(0.5, min(1.5, amplitude_envelope));
        amplitude_envelope = amplitude_envelope(:);  % ensure column vector
    end
end

%% Batch generation function
function BatchGenerate()
    % Batch generate datasets for various source counts and SNRs
    source_nums = [2, 3, 4];
    snr_values = [20, 24, 28];
    
    for src_num = source_nums
        for snr_val = snr_values
            fprintf('\nStarting generation for %d-source, SNR=%ddB dataset...\n', src_num, snr_val);
            MultiSourceGen_i(src_num, snr_val, 40);  % assuming 40 files (original had num_files=40 commented out)
        end
    end
    
    fprintf('\nAll datasets generated successfully!\n');
end