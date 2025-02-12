clear; clc;
%% end effector trajectory

% psm1
x_range = [-0.068, -0.028];
y_range = [0.221, 0.251];
z_range = [-0.226, -0.216];

% psm3
% x_range = [0.042, 0.070];
% y_range = [-0.030, 0.025];
% z_range = [-0.524, -0.507];

npts = 15;
seed = 20241207;
rng(seed,"twister");
x_samples = (x_range(2)-x_range(1)) .* rand(npts, 1) + x_range(1);
y_samples = (y_range(2)-y_range(1)) .* rand(npts, 1) + y_range(1);
z_samples = (z_range(2)-z_range(1)) .* rand(npts, 1) + z_range(1);
xyz = [x_samples'; y_samples'; z_samples'];

figure(1);
plot3(xyz(1,:),xyz(2,:),xyz(3,:),'b*','LineWidth',2);
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on
hold on

c = cscvn(xyz(:,[1:end 1]));

hold on
% fnplt(c,'r',2);
[points, t] = fnplt(c,'r',2);

tpts = [0 5];
tvec = 0:0.005:5;
[q, qd, qdd, pp] = bsplinepolytraj(points,tpts,tvec);
plot3(q(1,:), q(2,:), q(3,:), 'r', 'LineWidth', 2)

%% jaw open angle trajectory
initial_jawangle_js = rand(1);
angle = 0.3 * cos((2*pi)*tvec-initial_jawangle_js) + 0.2;
figure(2);
plot(tvec, angle, 'r.');
grid on
title("jaw angle")

%% last joint trajectory
initial_lastjoint_js = rand(1);
last_joint = 0.6 * sin((2*pi)*tvec-initial_lastjoint_js);
figure(3);
plot(tvec, last_joint, 'r.');
grid on
title("last joint")

%% wrist joint trajectory
initial_wrist_js = 0.3;
wrist_joint = 0.3 * cos((4*pi)*tvec-rand(1)) + initial_wrist_js;
figure(4);
plot(tvec, wrist_joint, 'r.');
grid on
title("wrist")

%% shaft joint trajectory
initial_shaft_js = -2.8; % 1.1;%
shaft_joint = 0.4 * sin((4*pi)*tvec-rand(1)) + initial_shaft_js;
figure(5);
plot(tvec, shaft_joint, 'r.');
grid on
title("shaft")

%% save .mat file
save('trajectory_000103_psm1.mat', "q", "angle", "last_joint", "wrist_joint", "shaft_joint");

