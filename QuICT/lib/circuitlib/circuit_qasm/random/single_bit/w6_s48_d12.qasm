OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
u3(0, 0, 1.5707963267948966) q[5];
tdg q[1];
s q[2];
t q[5];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[0];
y q[0];
x q[3];
x q[0];
sdg q[5];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[5];
s q[1];
t q[3];
rx(1.5707963267948966) q[4];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[3];
x q[1];
h q[5];
t q[1];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
sdg q[1];
h q[3];
sdg q[3];
rz(1.5707963267948966) q[3];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[3];
y q[5];
u3(0, 0, 1.5707963267948966) q[5];
s q[0];
y q[2];
tdg q[2];
s q[5];
rx(1.5707963267948966) q[0];
sdg q[5];
ry(1.5707963267948966) q[5];
s q[0];
u1(1.5707963267948966) q[1];
sdg q[0];
u1(1.5707963267948966) q[1];
sdg q[1];
h q[4];
z q[3];
sdg q[1];
u1(1.5707963267948966) q[1];
ry(1.5707963267948966) q[3];
s q[1];
