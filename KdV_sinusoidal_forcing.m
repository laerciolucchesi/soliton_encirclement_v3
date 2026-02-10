% Parâmetros do domínio
L = 100; N = 512;
x = linspace(-L, L, N+1); x(end) = []; 
dt = 0.005; 
t_max = 60;
t_vec = 0:dt:t_max;

% Parâmetros do "Gerador de Ondas" (Forçamento Localizado)
A = 1.0;            
omega = 0.8;        
x_source = 0;       
sigma = 2.0;        
forcing = @(t) A * sin(omega * t) * exp(-(x - x_source).^2 / (2 * sigma^2));

% Operadores de Fourier (Fator de Integração)
k = (pi/L) * [0:N/2-1 0 -N/2+1:-1];
L_op = 1i * k.^3;           
E = exp(L_op * dt); E2 = exp(L_op * dt/2);     

v = fft(zeros(1, N));

% --- CONFIGURAÇÃO DE VÍDEO E DADOS ---
video_filename = 'kdv_sinusoidal_forcing.mp4';
data_filename  = 'kdv_sinusoidal_forcing.mat';
v_obj = VideoWriter(video_filename, 'MPEG-4');
v_obj.FrameRate = 20; % Sincronizado com o passo visual (0.05s por frame)
open(v_obj);

% Pré-alocação (salvando a cada 10 passos)
n_saved_steps = floor(length(t_vec)/10);
U_history = zeros(n_saved_steps, N);
t_history = zeros(n_saved_steps, 1);
save_idx = 1;

% Figura para Animação
fig = figure('Color', 'w', 'Position', [100, 100, 900, 450]);
h_plot = plot(x, real(ifft(v)), 'LineWidth', 2, 'Color', [0.3 0.2 0.8]);
axis([min(x) max(x) -2 6]); grid on;
xlabel('Position (x)'); ylabel('Amplitude u(x,t)');
title('Forced KdV: Real-Time Visualization and Recording');

h_time = text(L*0.5, 5.5, 'Time: 0.00 s', 'FontSize', 12, 'FontWeight', 'bold', 'BackgroundColor', 'w');

% --- INÍCIO DA SIMULAÇÃO ---
wall_clock = tic;

for n = 1:length(t_vec)
    t_sim = t_vec(n);
    
    % RHS: Não-linearidade + Forçamento
    rhs = @(v_in, t_in) -3i * k .* fft(real(ifft(v_in)).^2) + fft(forcing(t_in));
    
    % RK4 + Fator de Integração
    k1 = rhs(v, t_sim);
    k2 = rhs(E2.*v + (dt/2)*k1, t_sim + dt/2);
    k3 = rhs(E2.*v + (dt/2)*k2, t_sim + dt/2);
    k4 = rhs(E.*v + dt*E2.*k3, t_sim + dt);
    v = E.*v + (dt/6) * (E.*k1 + 2*E2.*k2 + 2*E2.*k3 + k4);
    
    % Visualização, Gravação e Dados (a cada 10 passos)
    if mod(n, 10) == 0
        while toc(wall_clock) < t_sim
            % Sincronia real-time
        end
        
        u_vals = real(ifft(v));
        
        % Verifica Estabilidade
        if any(isnan(u_vals)) || any(abs(u_vals) > 50)
            error('Instabilidade detectada!');
        end
        
        % 1. Update Visual
        set(h_plot, 'YData', u_vals);
        set(h_time, 'String', sprintf('Time: %.2f s', t_sim));
        drawnow limitrate;
        
        % 2. Grava Frame
        writeVideo(v_obj, getframe(fig));
        
        % 3. Armazena Dados
        if save_idx <= n_saved_steps
            U_history(save_idx, :) = u_vals;
            t_history(save_idx) = t_sim;
            save_idx = save_idx + 1;
        end
    end
    if ~ishandle(fig), break; end
end

% --- FINALIZAÇÃO ---
close(v_obj);
save(data_filename, 'x', 't_history', 'U_history', 'A', 'omega');
fprintf('Simulação Finalizada!\nVídeo: %s\nDados: %s\n', video_filename, data_filename);