OPENQASM 2.0;
include "qelib1.inc";
qreg q[11];
creg c[11];
u1(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[0];
h q[9];
h q[6];
z q[0];
y q[7];
u1(1.5707963267948966) q[9];
s q[9];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[10];
sdg q[10];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
x q[10];
s q[3];
ry(1.5707963267948966) q[3];
z q[4];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[9];
x q[8];
tdg q[7];
tdg q[0];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[6];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[5];
h q[2];
h q[9];
h q[5];
s q[9];
s q[9];
rx(1.5707963267948966) q[8];
tdg q[8];
rz(1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
sdg q[8];
z q[6];
y q[7];
rx(1.5707963267948966) q[7];
y q[5];
h q[10];
s q[5];
ry(1.5707963267948966) q[7];
y q[3];
u1(1.5707963267948966) q[6];
h q[0];
z q[0];
tdg q[2];
y q[7];
u1(1.5707963267948966) q[0];
z q[3];
y q[10];
t q[10];
rz(1.5707963267948966) q[2];
h q[5];
rz(1.5707963267948966) q[4];
u1(1.5707963267948966) q[1];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[8];
h q[2];
y q[0];
x q[9];
