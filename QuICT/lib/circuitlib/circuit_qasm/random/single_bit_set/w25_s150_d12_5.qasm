OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
z q[22];
y q[2];
rz(1.5707963267948966) q[17];
z q[0];
t q[24];
tdg q[19];
u3(0, 0, 1.5707963267948966) q[7];
y q[10];
tdg q[16];
u1(1.5707963267948966) q[17];
rx(1.5707963267948966) q[13];
x q[7];
sdg q[12];
x q[14];
ry(1.5707963267948966) q[15];
u1(1.5707963267948966) q[15];
u1(1.5707963267948966) q[9];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
sdg q[9];
s q[2];
ry(1.5707963267948966) q[14];
rz(1.5707963267948966) q[8];
tdg q[22];
u2(1.5707963267948966, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[14];
s q[16];
u1(1.5707963267948966) q[17];
rx(1.5707963267948966) q[15];
tdg q[22];
z q[18];
rz(1.5707963267948966) q[15];
z q[8];
z q[11];
rz(1.5707963267948966) q[16];
u1(1.5707963267948966) q[13];
tdg q[15];
ry(1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[1];
sdg q[10];
x q[18];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[8];
t q[16];
sdg q[2];
h q[11];
y q[1];
u1(1.5707963267948966) q[12];
y q[3];
s q[16];
rx(1.5707963267948966) q[18];
y q[22];
s q[6];
z q[21];
u1(1.5707963267948966) q[12];
h q[19];
sdg q[7];
sdg q[7];
y q[0];
y q[7];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[24];
z q[18];
y q[6];
h q[21];
rx(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[8];
t q[8];
u1(1.5707963267948966) q[20];
sdg q[17];
s q[20];
s q[5];
u3(0, 0, 1.5707963267948966) q[12];
tdg q[20];
h q[4];
rx(1.5707963267948966) q[9];
u2(1.5707963267948966, 1.5707963267948966) q[21];
y q[7];
rx(1.5707963267948966) q[14];
u1(1.5707963267948966) q[15];
h q[19];
rz(1.5707963267948966) q[2];
z q[18];
rx(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[16];
t q[20];
y q[16];
sdg q[11];
z q[4];
tdg q[12];
tdg q[19];
z q[14];
s q[10];
y q[13];
rx(1.5707963267948966) q[24];
t q[18];
u2(1.5707963267948966, 1.5707963267948966) q[12];
sdg q[18];
sdg q[5];
tdg q[1];
z q[11];
sdg q[8];
y q[10];
u1(1.5707963267948966) q[2];
s q[17];
x q[12];
z q[9];
rx(1.5707963267948966) q[13];
u2(1.5707963267948966, 1.5707963267948966) q[15];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[24];
rz(1.5707963267948966) q[24];
u2(1.5707963267948966, 1.5707963267948966) q[7];
u1(1.5707963267948966) q[24];
x q[23];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[9];
x q[15];
rx(1.5707963267948966) q[22];
y q[23];
u2(1.5707963267948966, 1.5707963267948966) q[14];
ry(1.5707963267948966) q[20];
t q[20];
rx(1.5707963267948966) q[15];
tdg q[23];
u3(0, 0, 1.5707963267948966) q[6];
ry(1.5707963267948966) q[23];
sdg q[18];
tdg q[18];
x q[3];
u3(0, 0, 1.5707963267948966) q[8];
rx(1.5707963267948966) q[20];
tdg q[16];
u1(1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[16];
u3(0, 0, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
t q[8];
rx(1.5707963267948966) q[21];
h q[15];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[22];
s q[20];
t q[20];
sdg q[0];
tdg q[11];
u3(0, 0, 1.5707963267948966) q[17];
z q[18];
u2(1.5707963267948966, 1.5707963267948966) q[16];
