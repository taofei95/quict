OPENQASM 2.0;
include "qelib1.inc";
qreg q[25];
creg c[25];
t q[22];
u2(1.5707963267948966, 1.5707963267948966) q[23];
u1(1.5707963267948966) q[16];
sdg q[10];
ry(1.5707963267948966) q[17];
tdg q[5];
x q[9];
s q[16];
s q[8];
y q[7];
ry(1.5707963267948966) q[2];
s q[8];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[11];
s q[14];
t q[11];
tdg q[11];
u1(1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[2];
u1(1.5707963267948966) q[0];
rx(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[14];
u2(1.5707963267948966, 1.5707963267948966) q[1];
u2(1.5707963267948966, 1.5707963267948966) q[14];
rx(1.5707963267948966) q[23];
rz(1.5707963267948966) q[21];
s q[14];
s q[13];
h q[20];
z q[20];
ry(1.5707963267948966) q[24];
tdg q[14];
z q[12];
rz(1.5707963267948966) q[24];
s q[14];
rx(1.5707963267948966) q[18];
rz(1.5707963267948966) q[7];
x q[15];
t q[11];
s q[10];
tdg q[14];
tdg q[0];
x q[0];
u3(0, 0, 1.5707963267948966) q[11];
y q[14];
u2(1.5707963267948966, 1.5707963267948966) q[11];
s q[5];
u3(0, 0, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];
t q[20];
z q[8];
x q[3];
y q[21];
t q[5];
y q[12];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[19];
t q[14];
t q[11];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[9];
y q[17];
rz(1.5707963267948966) q[18];
ry(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[6];
z q[14];
y q[17];
u2(1.5707963267948966, 1.5707963267948966) q[20];
y q[17];
x q[1];
t q[16];
tdg q[0];
tdg q[18];
tdg q[4];
z q[16];
ry(1.5707963267948966) q[18];
rz(1.5707963267948966) q[0];
sdg q[7];
sdg q[12];
ry(1.5707963267948966) q[8];
rz(1.5707963267948966) q[16];
sdg q[13];
h q[21];
x q[20];
y q[2];
rx(1.5707963267948966) q[15];
x q[4];
u3(0, 0, 1.5707963267948966) q[12];
rx(1.5707963267948966) q[5];
tdg q[23];
u1(1.5707963267948966) q[17];
h q[8];
ry(1.5707963267948966) q[21];
rz(1.5707963267948966) q[15];
z q[24];
u1(1.5707963267948966) q[11];
sdg q[3];
y q[24];
sdg q[3];
rx(1.5707963267948966) q[7];
sdg q[18];
h q[12];
tdg q[2];
s q[2];
rx(1.5707963267948966) q[3];
tdg q[10];
z q[5];
t q[22];
tdg q[10];
u2(1.5707963267948966, 1.5707963267948966) q[11];
t q[12];
z q[6];
u1(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[24];
rz(1.5707963267948966) q[19];
t q[21];
x q[19];
sdg q[4];
u1(1.5707963267948966) q[5];
u1(1.5707963267948966) q[2];
x q[13];
rx(1.5707963267948966) q[16];
t q[2];
x q[12];
