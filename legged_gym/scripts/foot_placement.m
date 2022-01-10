clear; 
clc;
h = 0.40;
T = 0.33;
p = [0.866279, -0.629717];
[A, B, K, kappa, tprime] = get_ABK(h, T, p);



fprintf("kp: %f, kd: %f kappa: %f tprime: %f \n", K(1), K(2), kappa, tprime);

function [A, B, K, kappa, tprime] = get_ABK(h, T, p)

    g = 9.8;
    omega = sqrt(g/h);
    omega_T = omega*T;
    A = [cosh(omega_T),(1.0/omega)*sinh(omega_T); 
         omega*sinh(omega_T), cosh(omega_T)];
    
    B = [1-cosh(omega_T);-omega*sinh(omega_T)];
    C = eye(2);
    D = [0;0];
    K = place(A, -B, p);
    close_eigen = eig(A+B*K);
    kappa = K(1)-1;
    tprime = acoth(K(2)*omega)/omega;
end
