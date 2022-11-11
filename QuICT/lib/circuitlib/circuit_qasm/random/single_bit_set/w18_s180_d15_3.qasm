OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
t q[1];
x q[14];
y q[5];
z q[12];
rx(1.5707963267948966) q[5];
x q[3];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[1];
h q[4];
y q[2];
z q[6];
z q[13];
y q[10];
s q[8];
u2(1.5707963267948966, 1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
h q[2];
t q[17];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[15];
sdg q[13];
rz(1.5707963267948966) q[9];
x q[15];
h q[13];
u3(0, 0, 1.5707963267948966) q[0];
h q[4];
t q[9];
rx(1.5707963267948966) q[2];
z q[4];
s q[15];
y q[1];
u2(1.5707963267948966, 1.5707963267948966) q[14];
h q[8];
h q[16];
t q[15];
rz(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[12];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[13];
z q[1];
rx(1.5707963267948966) q[15];
rz(1.5707963267948966) q[4];
t q[11];
ry(1.5707963267948966) q[6];
t q[12];
ry(1.5707963267948966) q[1];
sdg q[15];
x q[14];
z q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[3];
y q[1];
x q[2];
z q[8];
sdg q[16];
rz(1.5707963267948966) q[1];
sdg q[14];
tdg q[3];
u2(1.5707963267948966, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[17];
z q[0];
rx(1.5707963267948966) q[0];
tdg q[16];
u1(1.5707963267948966) q[7];
t q[16];
tdg q[6];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[0];
t q[16];
tdg q[5];
s q[4];
ry(1.5707963267948966) q[9];
y q[6];
s q[7];
s q[12];
x q[15];
h q[5];
tdg q[10];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[0];
sdg q[14];
s q[2];
u3(0, 0, 1.5707963267948966) q[4];
rx(1.5707963267948966) q[6];
h q[14];
z q[15];
x q[13];
t q[7];
rz(1.5707963267948966) q[10];
x q[11];
rz(1.5707963267948966) q[8];
tdg q[1];
rx(1.5707963267948966) q[4];
x q[2];
rx(1.5707963267948966) q[6];
h q[12];
h q[15];
rx(1.5707963267948966) q[9];
tdg q[17];
rz(1.5707963267948966) q[1];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[1];
x q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
s q[2];
y q[5];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[10];
rz(1.5707963267948966) q[8];
y q[9];
u2(1.5707963267948966, 1.5707963267948966) q[12];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[0];
s q[14];
h q[0];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[6];
s q[15];
s q[2];
z q[6];
s q[4];
u3(0, 0, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[16];
y q[11];
sdg q[9];
z q[16];
z q[1];
u1(1.5707963267948966) q[8];
ry(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[1];
tdg q[17];
h q[10];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
s q[6];
z q[2];
u2(1.5707963267948966, 1.5707963267948966) q[17];
ry(1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[11];
u2(1.5707963267948966, 1.5707963267948966) q[13];
tdg q[6];
ry(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[15];
z q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
h q[10];
x q[4];
tdg q[17];
t q[12];
x q[14];
s q[7];
t q[9];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[5];
y q[7];
x q[10];
tdg q[9];
x q[10];
tdg q[8];
z q[12];
y q[2];
rz(1.5707963267948966) q[10];
sdg q[7];
y q[3];
t q[17];
rx(1.5707963267948966) q[13];
