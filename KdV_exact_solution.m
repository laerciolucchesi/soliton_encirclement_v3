% Parâmetros do domínio
L = 100; N = 512;
x = linspace(-L, L, N+1); x(end) = []; 
dt = 0.01; 
t_max = 60;        
t_vec = 0:dt:t_max;

% Parâmetros do Soliton
c = 4;              
x0 = 0;           
u0 = (c/2) * (sech(sqrt(c)/2 * (x - x0))).^2;

% Operadores de Fourier (Fator de Integração)
k = (pi/L) * [0:N/2-1 0 -N/2+1:-1];
L_op = 1i * k.^3;           
E = exp(L_op * dt); E2 = exp(L_op * dt/2);     
v = fft(u0);

% --- CONFIGURAÇÃO DE VÍDEO E DADOS ---
video_filename = 'kdv_exact_solution.mp4';
data_filename  = 'kdv_exact_solution.mat';
v_obj = VideoWriter(video_filename, 'MPEG-4');
v_obj.FrameRate = 25; % 25 quadros por segundo
open(v_obj);

% Pré-alocação da matriz de dados (Memória eficiente)
% Salvaremos apenas os frames visualizados (a cada 4 passos)
n_saved_steps = floor(length(t_vec)/4);
U_history = zeros(n_saved_steps, N);
t_history = zeros(n_saved_steps, 1);
save_idx = 1;

% Configuração da Figura
fig = figure('Color', 'w', 'Position', [100, 100, 900, 450]);
h_plot = plot(x, u0, 'LineWidth', 2.5, 'Color', [0 0.45 0.74]);
axis([min(x) max(x) -0.5 c+1]); grid on;
xlabel('Position (x)'); ylabel('Amplitude u(x,t)');
title('KdV Exact Solution: Real-Time Recording & Data Export');

h_time = text(L*0.5, c+0.5, 'Time: 0.00 s', 'FontSize', 14, 'FontWeight', 'bold', ...
    'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5]);

% --- INÍCIO DA EXECUÇÃO ---
wall_clock = tic; 

for n = 1:length(t_vec)
    t_sim = t_vec(n);
    
    % Cálculo Numérico (RK4 + ETD)
    rhs = @(v_in) -3i * k .* fft(real(ifft(v_in)).^2);
    k1 = rhs(v);
    k2 = rhs(E2.*v + (dt/2)*k1);
    k3 = rhs(E2.*v + (dt/2)*k2);
    k4 = rhs(E.*v + dt*E2.*k3);
    v = E.*v + (dt/6) * (E.*k1 + 2*E2.*k2 + 2*E2.*k3 + k4);
    
    % Visualização, Gravação e Armazenamento (a cada 4 passos)
    if mod(n, 4) == 0
        % Sincronização em Tempo Real
        while toc(wall_clock) < t_sim
            % Aguarda
        end
        
        u_current = real(ifft(v));
        
        % 1. Atualiza Visualização
        set(h_plot, 'YData', u_current);
        set(h_time, 'String', sprintf('Time: %.2f s', t_sim));
        drawnow limitrate;
        
        % 2. Grava Frame de Vídeo
        writeVideo(v_obj, getframe(fig));
        
        % 3. Armazena Dados Numericamente
        if save_idx <= n_saved_steps
            U_history(save_idx, :) = u_current;
            t_history(save_idx) = t_sim;
            save_idx = save_idx + 1;
        end
    end
    
    if ~ishandle(fig), break; end
end

% --- FINALIZAÇÃO ---
close(v_obj); % Fecha o vídeo
save(data_filename, 'x', 't_history', 'U_history'); % Salva os dados
fprintf('Processo concluído!\nVídeo: %s\nDados: %s\n', video_filename, data_filename);