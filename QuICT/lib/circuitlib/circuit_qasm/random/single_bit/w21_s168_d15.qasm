OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
t q[14];
u3(0, 0, 1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
sdg q[0];
h q[13];
rz(1.5707963267948966) q[2];
ry(1.5707963267948966) q[13];
sdg q[4];
rx(1.5707963267948966) q[16];
s q[1];
h q[3];
rz(1.5707963267948966) q[8];
sdg q[9];
rz(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[15];
z q[4];
tdg q[14];
x q[4];
rz(1.5707963267948966) q[3];
tdg q[4];
z q[2];
sdg q[20];
s q[9];
sdg q[13];
t q[14];
x q[11];
t q[5];
ry(1.5707963267948966) q[19];
u3(0, 0, 1.5707963267948966) q[3];
u1(1.5707963267948966) q[17];
ry(1.5707963267948966) q[7];
sdg q[15];
tdg q[3];
u1(1.5707963267948966) q[17];
sdg q[20];
s q[4];
h q[0];
x q[19];
y q[11];
rx(1.5707963267948966) q[11];
u1(1.5707963267948966) q[12];
sdg q[10];
x q[2];
rz(1.5707963267948966) q[10];
t q[11];
u3(0, 0, 1.5707963267948966) q[0];
u2(1.5707963267948966, 1.5707963267948966) q[17];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[5];
x q[0];
u1(1.5707963267948966) q[4];
y q[3];
u1(1.5707963267948966) q[10];
u2(1.5707963267948966, 1.5707963267948966) q[2];
sdg q[6];
y q[10];
x q[12];
z q[8];
s q[6];
h q[9];
z q[18];
u1(1.5707963267948966) q[12];
u1(1.5707963267948966) q[20];
z q[19];
ry(1.5707963267948966) q[19];
s q[17];
t q[4];
rz(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[16];
tdg q[11];
s q[8];
h q[18];
s q[20];
x q[8];
t q[5];
sdg q[1];
y q[2];
s q[20];
u2(1.5707963267948966, 1.5707963267948966) q[10];
x q[19];
sdg q[3];
sdg q[20];
s q[9];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[15];
sdg q[15];
ry(1.5707963267948966) q[17];
s q[0];
u3(0, 0, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[6];
x q[16];
u2(1.5707963267948966, 1.5707963267948966) q[2];
h q[17];
rx(1.5707963267948966) q[17];
u1(1.5707963267948966) q[2];
sdg q[16];
t q[14];
rz(1.5707963267948966) q[8];
ry(1.5707963267948966) q[0];
x q[5];
x q[12];
ry(1.5707963267948966) q[1];
s q[1];
rz(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[1];
u1(1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[17];
u2(1.5707963267948966, 1.5707963267948966) q[12];
sdg q[0];
s q[8];
z q[3];
sdg q[5];
ry(1.5707963267948966) q[11];
tdg q[6];
s q[10];
tdg q[11];
sdg q[2];
tdg q[12];
s q[20];
s q[12];
t q[14];
h q[17];
rx(1.5707963267948966) q[6];
u1(1.5707963267948966) q[14];
rz(1.5707963267948966) q[13];
rz(1.5707963267948966) q[13];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[10];
z q[0];
z q[2];
rx(1.5707963267948966) q[4];
u1(1.5707963267948966) q[12];
h q[17];
t q[13];
rx(1.5707963267948966) q[8];
x q[10];
tdg q[1];
u2(1.5707963267948966, 1.5707963267948966) q[20];
x q[18];
x q[0];
sdg q[4];
ry(1.5707963267948966) q[15];
x q[7];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[5];
rz(1.5707963267948966) q[15];
sdg q[4];
t q[8];
h q[6];
u1(1.5707963267948966) q[0];
ry(1.5707963267948966) q[1];
y q[18];
tdg q[7];
z q[0];
u1(1.5707963267948966) q[8];
t q[10];
rx(1.5707963267948966) q[10];
sdg q[18];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rz(1.5707963267948966) q[6];
tdg q[15];