function GEN_8PSK_MultiSource()
    for snr = -10:4:30
        GEN_8PSK_MultiSource_i(2, snr, 10)
    end
end

function GEN_8PSK_MultiSource_i(num_sources, snr, num_files)
    % Multi-source 8PSK signal generation function
    % num_sources: number of sources (2, 3, or 4)
    % snr: signal-to-noise ratio (dB)
    % num_files: number of files to generate

    %% Parameter validation
    if nargin < 1
        num_sources = 2;  % default: dual sources
    end
    if nargin < 2
        snr = 25;  % default: 25 dB
    end

    if ~ismember(num_sources, [2, 3, 4])
        error('Number of sources must be 2, 3, or 4');
    end

    %% ========== New: Burst signal parameters ==========
    enable_burst = true;           % enable burst-mode signals
    burst_probability = 0.7;       % probability of signal presence (0–1); 0.7 means 70% active, 30% silent
    min_burst_length = 1;          % minimum burst duration (in frames)
    max_burst_length = 5;          % maximum burst duration (in frames)

    fprintf('=== Burst Signal Configuration ===\n');
    fprintf('Burst mode enabled: %s\n', mat2str(enable_burst));
    fprintf('Signal presence probability: %.2f\n', burst_probability);
    fprintf('Minimum burst length: %d frames\n', min_burst_length);
    fprintf('Maximum burst length: %d frames\n', max_burst_length);
    fprintf('\n');

    %% ========== New: Impairment parameters ==========
    impaired = false;  % set to true to enable realistic impairments

    % 1. Carrier frequency drift parameters
    enable_carrier_drift = impaired;
    carrier_drift_rate = 50;            % Hz/s, linear drift rate
    carrier_drift_random_walk_std = 5;  % Hz, standard deviation of random walk component
    carrier_drift_type = 'combined';    % options: 'linear', 'random_walk', 'sinusoidal', 'combined'

    % Sinusoidal FM parameters (model oscillator instability)
    carrier_fm_amplitude = 20;          % Hz, FM modulation amplitude
    carrier_fm_frequency = 2;           % Hz, FM modulation frequency

    % 2. Symbol timing jitter parameters
    enable_timing_jitter = impaired;
    timing_jitter_rms = 0.015;          % RMS jitter as fraction of symbol period (1.5%)
    timing_jitter_type = 'gaussian';    % options: 'gaussian', 'uniform', 'colored'

    % 3. Amplitude variation parameters
    enable_amplitude_variation = impaired;
    amplitude_variation_std = 0.03;     % std of fast amplitude variation (3%)
    amplitude_variation_bandwidth = 50; % bandwidth of fast variation (Hz)
    amplitude_fade_depth = 0.1;         % depth of slow fading (10%)
    amplitude_fade_rate = 0.5;          % rate of slow fading (Hz)

    fprintf('=== Impairment Configuration ===\n');
    fprintf('1. Carrier Frequency Drift:\n');
    fprintf('   - Linear drift rate: %d Hz/s\n', carrier_drift_rate);
    fprintf('   - Random walk std: %d Hz\n', carrier_drift_random_walk_std);
    fprintf('   - Sinusoidal FM: %.0f Hz @ %.1f Hz\n', carrier_fm_amplitude, carrier_fm_frequency);
    fprintf('2. Symbol Timing Jitter:\n');
    fprintf('   - RMS jitter: %.2f%% of symbol period\n', timing_jitter_rms*100);
    fprintf('3. Amplitude Variation:\n');
    fprintf('   - Fast variation: std=%.1f%%, BW=%d Hz\n', amplitude_variation_std*100, amplitude_variation_bandwidth);
    fprintf('   - Slow fading: depth=%.1f%%, rate=%.2f Hz\n', amplitude_fade_depth*100, amplitude_fade_rate);
    fprintf('\n');

    %% Basic parameters
    flo = 20e6;                 % local oscillator frequency
    Fs_rf = 100e6;              % RF sampling rate

    %% Root-raised-cosine filter
    Fs_sps = 20;                % samples per symbol
    alpha = 0.35;               % roll-off factor
    span = 20;                  % filter span (in symbols)
    filterCoeffs = rcosdesign(alpha, span, Fs_sps, 'sqrt');

    %% Dataset parameters
    samples_per_file = 500;     % number of frames per file
    frame_length = 4096;        % samples per frame
    symbols_per_frame = 205;    % symbols per frame
    bits_per_symbol = 3;        % bits per 8PSK symbol
    total_frames = samples_per_file;
    total_samples = total_frames * frame_length;   % total samples per file
    total_symbols = total_frames * symbols_per_frame; % total symbols per file

    %% Delay parameters (reduced to minimize BER impact)
    symbol_rate = 5e6;          % symbol rate
    Tb = 1/symbol_rate;         % symbol period
    % Reduce relative delay to 0.05*Tb to minimize ISI/BER degradation
    base_delay = 0.05 * Tb;
    delay_samples = round((0:num_sources-1) * base_delay * Fs_rf);

    %% Low-pass filter design (for downconversion)
    rolloff = 0.35;
    cutoff_freq = symbol_rate * (1+rolloff)/2;
    normalized_cutoff = cutoff_freq/(Fs_rf/2);
    h_lpf = fir1(127, normalized_cutoff, 'low', kaiser(128, 5));

    %% Constellation mapping
    constellation = [exp(1j * pi/8);       % dec 0 (000)
                     exp(1j * 3*pi/8);     % dec 1 (001)
                     exp(1j * 5*pi/8);     % dec 3 (011)
                     exp(1j * 7*pi/8);     % dec 2 (010)
                     exp(1j * 9*pi/8);     % dec 6 (110)
                     exp(1j * 11*pi/8);    % dec 7 (111)
                     exp(1j * 13*pi/8);    % dec 5 (101)
                     exp(1j * 15*pi/8)];   % dec 4 (100)
    % Gray code mapping: natural binary index → constellation index
    gray_map_array = [1, 2, 4, 3, 8, 7, 5, 6]; % indices for n = 0:7

    %% ========== New: Carrier drift generator ==========
    function phase_drift = generate_carrier_drift(t, fc_base, drift_params, file_seed)
        % Generate time-varying carrier phase drift (integrated frequency offset)
        % Inputs:
        %   t - time vector
        %   fc_base - base carrier frequency
        %   drift_params - struct with drift parameters
        %   file_seed - random seed for reproducibility
        % Output:
        %   phase_drift - phase offset (radians)

        rng(file_seed * 7777); % ensure reproducibility
        Fs = 1/mean(diff(t));
        N = length(t);

        phase_drift = zeros(size(t));

        % 1. Linear drift component
        if contains(drift_params.type, 'linear') || contains(drift_params.type, 'combined')
            linear_drift = drift_params.rate * t;  % instantaneous frequency offset (Hz)
            phase_drift = phase_drift + 2*pi * cumsum(linear_drift) / Fs;
        end

        % 2. Random walk component (slow stochastic drift)
        if contains(drift_params.type, 'random_walk') || contains(drift_params.type, 'combined')
            random_steps = randn(N, 1) * drift_params.random_walk_std;
            % Low-pass filter to smooth random steps
            [b_smooth, a_smooth] = butter(2, 10/(Fs/2));  % 10 Hz cutoff
            random_walk = filter(b_smooth, a_smooth, random_steps);
            phase_drift = phase_drift + 2*pi * cumsum(random_walk) / Fs;
        end

        % 3. Sinusoidal FM component (oscillator instability)
        if contains(drift_params.type, 'sinusoidal') || contains(drift_params.type, 'combined')
            fm_signal = drift_params.fm_amplitude * sin(2*pi * drift_params.fm_frequency * t);
            phase_drift = phase_drift + 2*pi * cumsum(fm_signal) / Fs;
        end

        phase_drift = phase_drift(:);  % enforce column vector
    end

    %% ========== New: Timing jitter application ==========
    function jittered_signal = apply_timing_jitter(signal, jitter_params, num_symbols, sps, t_global)
        % Apply symbol-level timing jitter to baseband signal
        % Inputs:
        %   signal - input waveform (length = num_symbols * sps)
        %   jitter_params - timing jitter configuration
        %   num_symbols - total number of symbols
        %   sps - samples per symbol
        %   t_global - global time vector (for consistency)

        if ~jitter_params.enable
            jittered_signal = signal;
            return;
        end

        % Generate per-symbol jitter (in sample units)
        if strcmp(jitter_params.type, 'gaussian')
            jitter_samples = randn(num_symbols, 1) * jitter_params.rms * sps;
        elseif strcmp(jitter_params.type, 'uniform')
            jitter_samples = (rand(num_symbols, 1) - 0.5) * 2 * jitter_params.rms * sps * sqrt(3);
        else % 'colored'
            white_jitter = randn(num_symbols, 1);
            % First-order low-pass for colored noise
            alpha_jitter = 0.3;
            jitter_samples = filter(alpha_jitter, [1, -(1-alpha_jitter)], white_jitter);
            jitter_samples = jitter_samples * jitter_params.rms * sps / std(jitter_samples);
        end

        % Apply time-varying delay per symbol
        jittered_signal = zeros(size(t_global));
        signal_len = length(signal);

        for sym_idx = 1:num_symbols
            sym_start = (sym_idx-1) * sps + 1;
            sym_end = min(sym_idx * sps, signal_len);
            if sym_end > signal_len, break; end

            delay_samples_int = round(jitter_samples(sym_idx));

            if delay_samples_int ~= 0
                delayed_start = sym_start + delay_samples_int;
                delayed_end = sym_end + delay_samples_int;

                if delayed_start > 0 && delayed_end <= length(t_global)
                    jittered_signal(delayed_start:delayed_end) = signal(sym_start:sym_end);
                elseif delayed_start <= 0 && delayed_end > 0
                    jittered_signal(1:delayed_end) = signal(sym_start:sym_end + delayed_start - 1);
                elseif delayed_start <= length(t_global)
                    valid_start = max(1, delayed_start);
                    valid_end = min(length(t_global), delayed_end);
                    if valid_start <= valid_end
                        sig_start = sym_start + (valid_start - delayed_start);
                        sig_end = sym_start + (valid_end - delayed_start);
                        if sig_start <= signal_len && sig_end <= signal_len
                            jittered_signal(valid_start:valid_end) = signal(sig_start:sig_end);
                        end
                    end
                end
            else
                if sym_start <= length(t_global) && sym_end <= length(t_global)
                    jittered_signal(sym_start:sym_end) = signal(sym_start:sym_end);
                end
            end
        end
    end

    %% ========== New: Amplitude variation generator ==========
    function amplitude_envelope = generate_amplitude_variation(t, amp_params, file_seed)
        % Generate time-varying amplitude envelope
        % Inputs:
        %   t - time vector
        %   amp_params - amplitude variation parameters
        %   file_seed - random seed
        % Output:
        %   amplitude_envelope - normalized scale factor (real, >0)

        rng(file_seed * 8888);
        Fs = 1/mean(diff(t));
        N = length(t);

        amplitude_envelope = ones(N, 1);

        % 1. Fast stochastic variation (e.g., AGC, PA nonlinearity)
        if amp_params.variation_std > 0
            white_noise = randn(N, 1);
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

        % Clamp to reasonable range and ensure column vector
        amplitude_envelope = max(0.5, min(1.5, amplitude_envelope));
        amplitude_envelope = amplitude_envelope(:);
    end

    %% ========== New: Burst pattern generator ==========
    function burst_pattern = generate_burst_pattern(num_frames, prob, min_len, max_len, file_seed)
        % Generate on/off burst pattern per frame
        % Inputs:
        %   num_frames - total number of frames
        %   prob - probability of "on" state
        %   min_len, max_len - burst duration bounds
        %   file_seed - seed for reproducibility
        % Output:
        %   burst_pattern - logical vector (true = signal present)

        rng(file_seed * 9999);

        burst_pattern = false(num_frames, 1);
        idx = 1;

        while idx <= num_frames
            is_signal = (rand() < prob);

            if is_signal
                burst_len = min(max_len, max(min_len, round(rand() * (max_len - min_len + 1) + min_len)));
                burst_len = min(burst_len, num_frames - idx + 1);
                burst_pattern(idx:min(idx + burst_len - 1, num_frames)) = true;
                idx = idx + burst_len;
            else
                silence_len = min(max_len, max(min_len, round(rand() * (max_len - min_len + 1) + min_len)));
                silence_len = min(silence_len, num_frames - idx + 1);
                idx = idx + silence_len;
            end
        end

        % Ensure at least one active segment
        if all(~burst_pattern)
            burst_pattern(1) = true;
        end
    end

    %% ========== New: Burst mode application ==========
    function [mixed_with_burst, target_with_burst] = apply_burst_mode(mixed_signal, target_signal, burst_pattern)
        % Zero-out signal during silent frames while preserving noise floor
        % Inputs:
        %   mixed_signal - [frames × samples × 2] (I/Q)
        %   target_signal - [frames × samples × 2*num_sources]
        %   burst_pattern - [frames × 1] logical
        % Outputs:
        %   mixed_with_burst, target_with_burst - modified tensors

        [num_frames, frame_samples, ~] = size(mixed_signal);
        [~, ~, num_target_channels] = size(target_signal);

        mixed_with_burst = mixed_signal;
        target_with_burst = target_signal;

        for frame_idx = 1:num_frames
            if ~burst_pattern(frame_idx)  % silent frame
                % Estimate current frame noise power
                frame_noise_power_i = var(mixed_signal(frame_idx, :, 1), 'omitnan');
                frame_noise_power_q = var(mixed_signal(frame_idx, :, 2), 'omitnan');
                frame_noise_power = (frame_noise_power_i + frame_noise_power_q) / 2;

                % Regenerate Gaussian noise with same power
                noise_i = sqrt(max(frame_noise_power/2, 1e-10)) * randn(frame_samples, 1);
                noise_q = sqrt(max(frame_noise_power/2, 1e-10)) * randn(frame_samples, 1);

                mixed_with_burst(frame_idx, :, 1) = noise_i;
                mixed_with_burst(frame_idx, :, 2) = noise_q;

                % Zero out target (no signal)
                target_with_burst(frame_idx, :, :) = 0;
            end
        end
    end

    %% Assemble impairment parameter structs
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
    %% Generate signals file-by-file
    %% ========================================================

    for file_idx = 1:num_files
        fprintf('Generating file %d/%d\n', file_idx, num_files);

        %% Carrier frequency configuration (random offsets within ±700 Hz)
        fc_base = 20e6;  % nominal carrier at 20 MHz

        switch num_sources
            case 2
                offset_mag = rand() * 700;
                fc_offsets = [-offset_mag, offset_mag];
            case 3
                offset_mag = rand() * 700;
                fc_offsets = [-offset_mag, 0, offset_mag];
            case 4
                offset1 = rand() * 700;
                offset2 = rand() * 700;
                fc_offsets = [-offset1, -offset2, offset2, offset1];
        end

        fc_array = fc_base + fc_offsets;

        %% Initial phase: uniform in [0, π] for each source
        initial_phases = rand(num_sources, 1) * pi;

        % Global continuous time axis
        t_global = (0:total_samples-1)' / Fs_rf;

        %% Generate multi-source signals
        total_bits = total_symbols * bits_per_symbol;

        rf_signals = zeros(length(t_global), num_sources);
        ideal_bb_signals = zeros(length(t_global), num_sources);
        bit_data_all = cell(num_sources, 1);

        for src_idx = 1:num_sources
            %% Generate random bit stream
            bit_data_all{src_idx} = randi([0, 1], total_bits, 1, 'uint8');

            %% Map to 8PSK symbols (Gray-coded)
            symbol_indices = bi2de(reshape(bit_data_all{src_idx}, bits_per_symbol, [])', 'left-msb') + 1;
            symbol_indices = gray_map_array(symbol_indices);
            s_complex = constellation(symbol_indices);

            %% Upsample and pulse-shape
            s_upsampled = upsample(s_complex, Fs_sps);
            s_shaped = conv(s_upsampled, filterCoeffs, 'same');
            s_shaped = s_shaped(1:length(t_global));  % truncate/zero-pad to match time axis

            %% Preserve ideal waveform for target labeling
            s_shaped_ideal = s_shaped;

            %% Apply timing jitter (if enabled)
            if enable_timing_jitter
                num_symbols_total = length(symbol_indices);
                s_shaped = apply_timing_jitter(s_shaped, jitter_params, num_symbols_total, Fs_sps, t_global);
                s_shaped = s_shaped(1:length(t_global));

                % Apply identical jitter to ideal signal (for consistency in labeling)
                s_shaped_ideal = apply_timing_jitter(s_shaped_ideal, jitter_params, num_symbols_total, Fs_sps, t_global);
                s_shaped_ideal = s_shaped_ideal(1:length(t_global));
            end

            %% Apply source-specific delay
            if delay_samples(src_idx) > 0
                s_shaped = [zeros(delay_samples(src_idx), 1); s_shaped(1:end-delay_samples(src_idx))];
                s_shaped_ideal = [zeros(delay_samples(src_idx), 1); s_shaped_ideal(1:end-delay_samples(src_idx))];
            end
            s_shaped = s_shaped(1:length(t_global));
            s_shaped_ideal = s_shaped_ideal(1:length(t_global));

            %% Modulate to RF with impairments
            if enable_carrier_drift
                phase_drift = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift + initial_phases(src_idx)));
            else
                carrier = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end

            rf_signal = real(s_shaped .* carrier);

            if enable_amplitude_variation
                amp_envelope = generate_amplitude_variation(t_global, amp_params, file_idx*200 + src_idx);
                rf_signal = rf_signal .* amp_envelope;
            end

            rf_signals(:, src_idx) = rf_signal;

            %% Generate ideal baseband signal (for supervised labels)
            % Use *ideal* shaped signal + *same* carrier (with drift if enabled) for coherent reference
            if enable_carrier_drift
                phase_drift_ideal = generate_carrier_drift(t_global, fc_array(src_idx), drift_params, file_idx*100 + src_idx);
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + phase_drift_ideal + initial_phases(src_idx)));
            else
                carrier_ideal = exp(1i*(2*pi*fc_array(src_idx)*t_global + initial_phases(src_idx)));
            end

            rf_signal_ideal = real(s_shaped_ideal .* carrier_ideal);

            baseband_i = rf_signal_ideal .* cos(2*pi*flo*t_global);
            baseband_q = rf_signal_ideal .* (-sin(2*pi*flo*t_global));
            bb_i_filtered = conv(baseband_i, h_lpf, 'same');
            bb_q_filtered = conv(baseband_q, h_lpf, 'same');
            ideal_bb_signals(:, src_idx) = complex(bb_i_filtered, bb_q_filtered);
        end

        %% Combine and add noise
        rf_combined = sum(rf_signals, 2);
        rf_combined_noisy = awgn(rf_combined, snr, 'measured');

        %% Downconvert to baseband
        lo_i = cos(2*pi*flo*t_global);
        lo_q = sin(2*pi*flo*t_global);

        baseband_i = rf_combined_noisy .* lo_i;
        baseband_q = rf_combined_noisy .* (-lo_q);
        bb_i_filtered = conv(baseband_i, h_lpf, 'same');
        bb_q_filtered = conv(baseband_q, h_lpf, 'same');
        mixed_baseband = complex(bb_i_filtered, bb_q_filtered);

        %% File-level normalization
        file_max_mixed = max(abs(mixed_baseband));
        file_max_ideal = max(max(abs(ideal_bb_signals)));

        mixed_baseband = mixed_baseband / file_max_mixed;
        ideal_bb_signals = ideal_bb_signals / file_max_ideal;

        %% Reshape to frames
        mixed_frames = single(zeros(samples_per_file, frame_length, 2));
        ideal_frames = single(zeros(samples_per_file, frame_length, 2*num_sources));

        for frame_idx = 1:samples_per_file
            start_idx = (frame_idx-1)*frame_length + 1;
            end_idx = frame_idx * frame_length;
            if end_idx > length(mixed_baseband), break; end

            frame_data = mixed_baseband(start_idx:end_idx);
            mixed_frames(frame_idx, :, 1) = real(frame_data);
            mixed_frames(frame_idx, :, 2) = imag(frame_data);

            for src_idx = 1:num_sources
                frame_ideal = ideal_bb_signals(start_idx:end_idx, src_idx);
                ideal_frames(frame_idx, :, 2*src_idx-1) = real(frame_ideal);
                ideal_frames(frame_idx, :, 2*src_idx) = imag(frame_ideal);
            end
        end

        %% ========== Apply burst mode ==========
        if enable_burst
            burst_pattern = generate_burst_pattern(samples_per_file, burst_probability, ...
                                                   min_burst_length, max_burst_length, file_idx);
            [mixed_frames, ideal_frames] = apply_burst_mode(mixed_frames, ideal_frames, burst_pattern);

            signal_frames = sum(burst_pattern);
            silence_frames = sum(~burst_pattern);
            fprintf('  Burst pattern: %d active bursts, %d silent bursts, signal=%d frames, silent=%d frames\n', ...
                    sum(diff([false; burst_pattern; false]) == 1), ...
                    sum(diff([true; ~burst_pattern; true]) == 1), ...
                    signal_frames, silence_frames);
        end

        %% Save datasets
        save_path_mixed = sprintf('./%dSource_8PSK_Dataset_mixed_%d_SNR=%ddB.mat', ...
                                num_sources, file_idx, snr);
        save_path_target = sprintf('./%dSource_8PSK_Dataset_target_%d_SNR=%ddB.mat', ...
                                 num_sources, file_idx, snr);

        save(save_path_mixed, 'mixed_frames', '-v7.3');
        save(save_path_target, 'ideal_frames', '-v7.3');

        % Save bit sequences
        for src_idx = 1:num_sources
            save_path_bit = sprintf('./%dSource_8PSK_BitData_%d_SNR=%ddB_Source%d.mat', ...
                                  num_sources, file_idx, snr, src_idx);
            file_bits = bit_data_all{src_idx};
            save(save_path_bit, 'file_bits', '-v7.3');
        end

        fprintf('File %d/%d saved (%d sources)\n', file_idx, num_files, num_sources);
        fprintf('  Frequency offsets (Hz): ');
        fprintf('%.1f ', fc_offsets);
        fprintf('\n  Initial phases (rad): ');
        fprintf('%.3f ', initial_phases);
        fprintf('\n');
    end

    %% Final summary
    fprintf('\n=== Dataset Generation Completed ===\n');
    fprintf('Number of sources: %d\n', num_sources);
    fprintf('SNR: %d dB\n', snr);
    fprintf('Frequency offset range: ±700 Hz\n');
    fprintf('Files generated: %d\n', num_files);
    fprintf('Burst mode: %s\n', mat2str(enable_burst));
    if enable_burst
        fprintf('  - Signal probability: %.2f\n', burst_probability);
        fprintf('  - Burst length: %d–%d frames\n', min_burst_length, max_burst_length);
    end

    disp('All files successfully generated.');
end