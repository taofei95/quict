OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[7];
h q[4];
z q[5];
rz(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[6];
t q[4];
s q[2];
z q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[4];
sdg q[2];
rz(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
z q[4];
rx(1.5707963267948966) q[6];
z q[2];
tdg q[4];
sdg q[5];
rz(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[2];
x q[2];
tdg q[1];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
s q[6];
rx(1.5707963267948966) q[0];
y q[0];
rx(1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[0];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[5];
sdg q[4];
tdg q[5];
z q[1];
y q[4];
z q[6];
ry(1.5707963267948966) q[6];
h q[6];
h q[4];
tdg q[1];
t q[3];
y q[5];
rz(1.5707963267948966) q[5];
sdg q[2];
z q[6];
