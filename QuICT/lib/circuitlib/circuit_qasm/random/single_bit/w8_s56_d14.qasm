OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rx(1.5707963267948966) q[4];
y q[6];
s q[4];
ry(1.5707963267948966) q[1];
z q[2];
tdg q[7];
u1(1.5707963267948966) q[5];
x q[5];
tdg q[1];
y q[0];
h q[7];
z q[4];
ry(1.5707963267948966) q[0];
z q[0];
rz(1.5707963267948966) q[7];
sdg q[3];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
z q[0];
u3(0, 0, 1.5707963267948966) q[2];
z q[7];
u2(1.5707963267948966, 1.5707963267948966) q[6];
y q[2];
u3(0, 0, 1.5707963267948966) q[6];
h q[3];
t q[5];
h q[3];
rz(1.5707963267948966) q[3];
h q[4];
rz(1.5707963267948966) q[3];
t q[4];
y q[5];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[6];
sdg q[5];
u1(1.5707963267948966) q[3];
z q[2];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
h q[5];
y q[5];
y q[1];
ry(1.5707963267948966) q[5];
x q[5];
t q[3];
h q[0];
sdg q[6];
rx(1.5707963267948966) q[2];
y q[5];
rx(1.5707963267948966) q[5];
tdg q[5];
u1(1.5707963267948966) q[5];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[5];