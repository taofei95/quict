OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
h q[19];
y q[4];
z q[9];
sdg q[11];
u3(0, 0, 1.5707963267948966) q[9];
u3(0, 0, 1.5707963267948966) q[20];
s q[21];
u1(1.5707963267948966) q[4];
z q[21];
u2(1.5707963267948966, 1.5707963267948966) q[15];
t q[1];
h q[5];
tdg q[2];
u1(1.5707963267948966) q[10];
tdg q[4];
rz(1.5707963267948966) q[11];
h q[5];
x q[22];
s q[22];
u2(1.5707963267948966, 1.5707963267948966) q[23];
rx(1.5707963267948966) q[0];
h q[7];
u3(0, 0, 1.5707963267948966) q[1];
h q[8];
t q[9];
rz(1.5707963267948966) q[12];
t q[21];
y q[0];
u2(1.5707963267948966, 1.5707963267948966) q[18];
rz(1.5707963267948966) q[20];
rx(1.5707963267948966) q[5];
s q[19];
sdg q[22];
x q[13];
ry(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[5];
tdg q[6];
rz(1.5707963267948966) q[4];
rx(1.5707963267948966) q[23];
s q[0];
t q[16];
u1(1.5707963267948966) q[22];
s q[0];
x q[7];
u3(0, 0, 1.5707963267948966) q[2];
h q[8];
rz(1.5707963267948966) q[6];
sdg q[9];
t q[0];
t q[0];
t q[18];
sdg q[20];
ry(1.5707963267948966) q[11];
h q[12];
rx(1.5707963267948966) q[0];
s q[17];
h q[5];
u2(1.5707963267948966, 1.5707963267948966) q[0];
u1(1.5707963267948966) q[16];
rx(1.5707963267948966) q[19];
y q[7];
t q[2];
u2(1.5707963267948966, 1.5707963267948966) q[14];
t q[24];
u2(1.5707963267948966, 1.5707963267948966) q[8];
rz(1.5707963267948966) q[11];
z q[10];
ry(1.5707963267948966) q[3];
sdg q[4];
s q[16];
t q[23];
s q[16];
rx(1.5707963267948966) q[20];
u1(1.5707963267948966) q[16];
h q[5];
x q[7];
ry(1.5707963267948966) q[0];
u3(0, 0, 1.5707963267948966) q[2];
tdg q[4];
z q[11];
y q[4];
u2(1.5707963267948966, 1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
sdg q[3];
tdg q[7];
rz(1.5707963267948966) q[11];
z q[23];
t q[23];
h q[6];
tdg q[10];
x q[23];
ry(1.5707963267948966) q[15];
h q[4];
rx(1.5707963267948966) q[20];
x q[8];
h q[18];
u1(1.5707963267948966) q[24];
sdg q[2];
rz(1.5707963267948966) q[12];
u1(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[22];
x q[6];
u1(1.5707963267948966) q[24];
rz(1.5707963267948966) q[6];
h q[1];
ry(1.5707963267948966) q[11];
h q[9];
u2(1.5707963267948966, 1.5707963267948966) q[17];
tdg q[10];
sdg q[19];
h q[5];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[10];
z q[11];
u3(0, 0, 1.5707963267948966) q[0];
y q[16];
u1(1.5707963267948966) q[6];
t q[16];
u1(1.5707963267948966) q[7];
rx(1.5707963267948966) q[5];
h q[10];
tdg q[14];
z q[0];
h q[16];
sdg q[14];
u1(1.5707963267948966) q[1];
x q[0];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[6];
s q[14];
u1(1.5707963267948966) q[7];
u3(0, 0, 1.5707963267948966) q[9];
z q[3];
tdg q[12];
z q[12];
z q[22];
h q[7];
rz(1.5707963267948966) q[18];
rx(1.5707963267948966) q[11];
t q[5];
u3(0, 0, 1.5707963267948966) q[6];
u1(1.5707963267948966) q[1];
sdg q[8];
u3(0, 0, 1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[2];
x q[18];
u3(0, 0, 1.5707963267948966) q[7];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[19];