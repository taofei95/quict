OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
x q[7];
tdg q[5];
u1(1.5707963267948966) q[7];
tdg q[5];
rx(1.5707963267948966) q[1];
s q[6];
u1(1.5707963267948966) q[2];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[10];
rz(1.5707963267948966) q[1];
z q[8];
rz(1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[0];
s q[8];
tdg q[2];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[8];
t q[4];
h q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
h q[5];
z q[3];
z q[0];
s q[2];
h q[1];
tdg q[8];
rz(1.5707963267948966) q[2];
z q[9];
u1(1.5707963267948966) q[3];
sdg q[4];
ry(1.5707963267948966) q[0];
x q[10];
sdg q[6];
u3(0, 0, 1.5707963267948966) q[8];
t q[2];
u3(0, 0, 1.5707963267948966) q[3];
sdg q[5];
tdg q[7];
u1(1.5707963267948966) q[8];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[5];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[1];
z q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[7];
z q[8];
u3(0, 0, 1.5707963267948966) q[9];
t q[1];
z q[1];
t q[4];
u3(0, 0, 1.5707963267948966) q[1];
y q[0];
