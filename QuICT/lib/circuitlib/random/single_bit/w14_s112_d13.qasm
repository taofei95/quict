OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
x q[3];
sdg q[7];
tdg q[0];
tdg q[6];
sdg q[12];
s q[1];
sdg q[8];
h q[2];
rz(1.5707963267948966) q[11];
y q[0];
x q[9];
z q[0];
t q[5];
s q[6];
u3(0, 0, 1.5707963267948966) q[6];
rx(1.5707963267948966) q[10];
tdg q[12];
u1(1.5707963267948966) q[6];
h q[13];
z q[11];
u2(1.5707963267948966, 1.5707963267948966) q[11];
h q[4];
h q[9];
z q[7];
h q[1];
rx(1.5707963267948966) q[0];
tdg q[2];
t q[10];
u2(1.5707963267948966, 1.5707963267948966) q[6];
h q[13];
u3(0, 0, 1.5707963267948966) q[8];
tdg q[10];
z q[7];
u1(1.5707963267948966) q[12];
h q[4];
t q[6];
ry(1.5707963267948966) q[10];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[8];
h q[12];
y q[3];
h q[7];
u2(1.5707963267948966, 1.5707963267948966) q[7];
rz(1.5707963267948966) q[2];
h q[7];
tdg q[1];
u3(0, 0, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[7];
u1(1.5707963267948966) q[2];
s q[13];
z q[1];
x q[12];
h q[4];
sdg q[2];
sdg q[11];
y q[0];
x q[1];
rz(1.5707963267948966) q[7];
x q[0];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[12];
sdg q[8];
tdg q[11];
x q[8];
x q[8];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[1];
h q[6];
u3(0, 0, 1.5707963267948966) q[3];
t q[5];
t q[9];
h q[7];
ry(1.5707963267948966) q[11];
sdg q[12];
rx(1.5707963267948966) q[12];
z q[5];
rx(1.5707963267948966) q[1];
s q[12];
s q[10];
rx(1.5707963267948966) q[3];
sdg q[2];
h q[9];
rz(1.5707963267948966) q[0];
y q[7];
h q[1];
z q[13];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[8];
x q[13];
t q[13];
u1(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
u1(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
y q[9];
s q[10];
tdg q[7];
u3(0, 0, 1.5707963267948966) q[8];
sdg q[8];
s q[7];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
y q[5];
z q[0];
s q[13];
rx(1.5707963267948966) q[9];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[6];