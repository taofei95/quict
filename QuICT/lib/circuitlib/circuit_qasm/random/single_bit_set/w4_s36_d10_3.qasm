OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
t q[1];
x q[2];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[1];
t q[1];
rx(1.5707963267948966) q[1];
s q[0];
u2(1.5707963267948966, 1.5707963267948966) q[1];
t q[0];
sdg q[3];
rz(1.5707963267948966) q[0];
sdg q[3];
sdg q[0];
u3(0, 0, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
tdg q[3];
rz(1.5707963267948966) q[2];
x q[2];
x q[3];
t q[3];
h q[0];
tdg q[0];
sdg q[1];
rz(1.5707963267948966) q[0];
y q[2];
t q[3];
y q[1];
u1(1.5707963267948966) q[2];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[0];
h q[2];
sdg q[0];
tdg q[3];
