OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
x q[0];
u1(1.5707963267948966) q[12];
h q[1];
u2(1.5707963267948966, 1.5707963267948966) q[15];
y q[8];
rz(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[2];
z q[10];
sdg q[2];
h q[6];
u3(0, 0, 1.5707963267948966) q[6];
h q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[4];
x q[1];
ry(1.5707963267948966) q[11];
t q[12];
h q[10];
x q[3];
t q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rx(1.5707963267948966) q[11];
z q[13];
ry(1.5707963267948966) q[12];
x q[11];
z q[0];
u2(1.5707963267948966, 1.5707963267948966) q[9];
y q[6];
y q[8];
x q[14];
u1(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[7];
z q[12];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[1];
s q[11];
h q[14];
rx(1.5707963267948966) q[4];
x q[8];
rx(1.5707963267948966) q[8];
s q[0];
u3(0, 0, 1.5707963267948966) q[13];
rz(1.5707963267948966) q[6];
h q[12];
rz(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[7];
sdg q[3];
y q[12];
rx(1.5707963267948966) q[12];
z q[6];
sdg q[12];
rz(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[3];
t q[9];
u1(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[3];
h q[10];
ry(1.5707963267948966) q[6];
t q[8];
u2(1.5707963267948966, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[9];
h q[2];
t q[13];
rz(1.5707963267948966) q[11];
sdg q[8];
sdg q[0];
u2(1.5707963267948966, 1.5707963267948966) q[8];
s q[4];
z q[3];
rz(1.5707963267948966) q[14];
u1(1.5707963267948966) q[13];
tdg q[13];
y q[15];
t q[9];
