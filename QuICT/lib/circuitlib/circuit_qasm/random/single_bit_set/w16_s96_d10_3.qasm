OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
z q[13];
t q[4];
y q[8];
tdg q[9];
h q[13];
z q[2];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[15];
t q[3];
u1(1.5707963267948966) q[4];
sdg q[13];
rz(1.5707963267948966) q[15];
sdg q[12];
rz(1.5707963267948966) q[12];
rx(1.5707963267948966) q[8];
u1(1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[15];
sdg q[1];
u1(1.5707963267948966) q[7];
x q[14];
rz(1.5707963267948966) q[6];
tdg q[15];
h q[1];
u3(0, 0, 1.5707963267948966) q[11];
y q[8];
h q[8];
y q[14];
u2(1.5707963267948966, 1.5707963267948966) q[1];
rz(1.5707963267948966) q[4];
h q[7];
rz(1.5707963267948966) q[6];
tdg q[7];
tdg q[7];
u2(1.5707963267948966, 1.5707963267948966) q[14];
h q[13];
rx(1.5707963267948966) q[2];
z q[4];
s q[13];
u3(0, 0, 1.5707963267948966) q[1];
sdg q[4];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
s q[15];
z q[8];
h q[8];
ry(1.5707963267948966) q[3];
rz(1.5707963267948966) q[15];
x q[13];
x q[0];
ry(1.5707963267948966) q[1];
s q[1];
sdg q[7];
y q[3];
u2(1.5707963267948966, 1.5707963267948966) q[4];
z q[4];
h q[10];
rx(1.5707963267948966) q[1];
s q[9];
u3(0, 0, 1.5707963267948966) q[9];
u1(1.5707963267948966) q[10];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[9];
x q[8];
u2(1.5707963267948966, 1.5707963267948966) q[3];
z q[1];
s q[11];
sdg q[8];
rx(1.5707963267948966) q[6];
x q[13];
rz(1.5707963267948966) q[14];
sdg q[14];
h q[10];
h q[10];
s q[13];
s q[9];
s q[15];
z q[5];
t q[9];
u2(1.5707963267948966, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[0];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[15];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[5];
t q[4];
rx(1.5707963267948966) q[2];
u3(0, 0, 1.5707963267948966) q[15];
tdg q[10];
ry(1.5707963267948966) q[14];
z q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[14];
h q[15];
