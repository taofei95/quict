OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
sdg q[7];
z q[2];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u1(1.5707963267948966) q[1];
tdg q[12];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[10];
ry(1.5707963267948966) q[22];
u2(1.5707963267948966, 1.5707963267948966) q[10];
z q[7];
s q[2];
tdg q[18];
u1(1.5707963267948966) q[2];
ry(1.5707963267948966) q[16];
h q[4];
h q[11];
s q[14];
t q[22];
z q[17];
h q[17];
sdg q[6];
y q[17];
x q[13];
u1(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[3];
s q[8];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[15];
sdg q[2];
z q[16];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[21];
x q[20];
rx(1.5707963267948966) q[2];
tdg q[11];
h q[16];
u2(1.5707963267948966, 1.5707963267948966) q[6];
rz(1.5707963267948966) q[3];
s q[11];
tdg q[8];
sdg q[0];
sdg q[6];
s q[10];
u1(1.5707963267948966) q[15];
tdg q[19];
tdg q[8];
s q[6];
u3(0, 0, 1.5707963267948966) q[8];
u1(1.5707963267948966) q[18];
sdg q[0];
x q[12];
u3(0, 0, 1.5707963267948966) q[14];
u3(0, 0, 1.5707963267948966) q[4];
ry(1.5707963267948966) q[6];
tdg q[21];
rz(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[6];
u2(1.5707963267948966, 1.5707963267948966) q[8];
tdg q[12];
t q[12];
sdg q[10];
ry(1.5707963267948966) q[11];
s q[18];
x q[10];
ry(1.5707963267948966) q[16];
s q[22];
z q[2];
tdg q[2];
sdg q[16];
u1(1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[2];
rx(1.5707963267948966) q[6];
tdg q[19];
z q[22];
sdg q[17];
s q[20];
u1(1.5707963267948966) q[18];
y q[6];
u3(0, 0, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[2];
tdg q[10];
x q[8];
ry(1.5707963267948966) q[1];
tdg q[17];
u3(0, 0, 1.5707963267948966) q[7];
u2(1.5707963267948966, 1.5707963267948966) q[20];
z q[4];
sdg q[1];
h q[6];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[5];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[13];
tdg q[15];
u1(1.5707963267948966) q[14];
u1(1.5707963267948966) q[17];
t q[22];
y q[19];
y q[18];
ry(1.5707963267948966) q[13];
u1(1.5707963267948966) q[9];
t q[14];
tdg q[0];
sdg q[4];
y q[0];
ry(1.5707963267948966) q[5];
y q[15];
sdg q[4];
y q[18];
x q[19];
ry(1.5707963267948966) q[22];
s q[18];
rz(1.5707963267948966) q[3];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[8];
tdg q[12];
rx(1.5707963267948966) q[17];
t q[2];
z q[4];
sdg q[3];
t q[22];
x q[15];
s q[19];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[1];
z q[7];
t q[3];
u2(1.5707963267948966, 1.5707963267948966) q[0];
sdg q[17];
t q[13];
t q[11];
rz(1.5707963267948966) q[20];
u1(1.5707963267948966) q[6];
z q[22];
y q[0];
z q[5];
y q[20];
rz(1.5707963267948966) q[12];
x q[19];
sdg q[2];
t q[22];
h q[1];
tdg q[10];
h q[19];
x q[21];
s q[4];
y q[19];
rx(1.5707963267948966) q[5];
x q[5];
t q[12];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[16];
x q[20];
y q[2];
t q[7];
ry(1.5707963267948966) q[19];
y q[11];
u3(0, 0, 1.5707963267948966) q[20];
z q[21];
tdg q[21];
tdg q[6];
sdg q[21];
u3(0, 0, 1.5707963267948966) q[22];
ry(1.5707963267948966) q[0];
t q[0];
h q[19];
t q[8];
z q[17];
t q[19];
sdg q[3];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[14];
s q[11];
t q[17];
z q[14];
rx(1.5707963267948966) q[22];
tdg q[3];
ry(1.5707963267948966) q[16];
rz(1.5707963267948966) q[14];
x q[17];
x q[10];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[20];
