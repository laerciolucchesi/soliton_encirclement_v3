# Ideas and Research Directions (Encirclement Project)

Notes after meeting with Markus and Bruno - 08/Jan/2026

1) Related literature for tangential spacing control: identify work related to soliton-inspired tangential control, including distributed control, synchronization, traveling waves, coupled chains, and soliton-inspired systems beyond encirclement.

2) State of the art in encirclement: collect and organize the main approaches, assumptions, communication models, global/local information requirements, and performance metrics.

3) Soliton control that reproduces KdV dynamics: investigate formulations whose collective evolution approximates the Korteweg-de Vries (KdV) equation and evaluate stability and robustness benefits.

4) Robustness to sensing noise: add noise to local measurements (position, angle, distance to target and neighbors) and evaluate robustness.

5) Communication impairments: evaluate performance under limited range, variable delays, and intermittent packet loss.

6) Sensitivity to control period: study how different update periods affect stability, convergence, and formation quality.

7) Alternative nonlinearities: replace the cubic term with other odd nonlinear functions (e.g., tan(u) or smooth saturations) and compare stability and transient behavior.

8) Optional feedforward term: assess whether a feedforward component improves target tracking or disturbance rejection.

9) Impact of agent failures: study temporary or permanent failures and the effect on formation stability and global metrics.

10) Dynamic agent entry/exit: evaluate behavior when agents join or leave the swarm dynamically.

11) Scalability with agent count: analyze how performance metrics and convergence vary with the number of agents.

12) Typical dynamic parameters in the literature: gather typical limits for different drone classes (racing, surveillance/cinematography, cargo) to set realistic simulations.

13) Heterogeneous agent dynamics: study behavior with different speed/acceleration limits, actuation delays, or motion models.

14) Extension to other formations: adapt the framework to other shapes (e.g., V formation) and test whether local coupling still works.

15) External disturbances: include wind gusts or persistent currents and evaluate disturbance rejection.

16) Controller design rules and stability analysis: propose systematic tuning rules linking control parameters, dynamics, and communication conditions.

17) 3D encirclement: extend the framework to 3D formations and study sensing/communication requirements.

18) GPS-denied encirclement: evaluate feasibility using only relative measurements exchanged over communication links.
