OPENQASM 2.0;
include "qelib1.inc";
qreg q[24];
creg c[24];
rx(1.5707963267948966) q[5];
z q[17];
h q[18];
h q[13];
z q[22];
z q[1];
t q[19];
ry(1.5707963267948966) q[22];
u1(1.5707963267948966) q[23];
z q[20];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[19];
u2(1.5707963267948966, 1.5707963267948966) q[15];
z q[16];
rx(1.5707963267948966) q[3];
y q[4];
rx(1.5707963267948966) q[19];
tdg q[17];
y q[14];
h q[18];
x q[4];
x q[10];
s q[8];
s q[12];
u2(1.5707963267948966, 1.5707963267948966) q[20];
y q[5];
tdg q[2];
y q[12];
ry(1.5707963267948966) q[6];
x q[16];
h q[12];
u1(1.5707963267948966) q[1];
y q[16];
s q[12];
y q[22];
ry(1.5707963267948966) q[12];
rz(1.5707963267948966) q[17];
rx(1.5707963267948966) q[21];
s q[8];
ry(1.5707963267948966) q[5];
z q[2];
t q[9];
u1(1.5707963267948966) q[2];
rx(1.5707963267948966) q[15];
z q[15];
u2(1.5707963267948966, 1.5707963267948966) q[10];
tdg q[2];
sdg q[13];
h q[6];
s q[9];
rz(1.5707963267948966) q[7];
z q[6];
t q[6];
rx(1.5707963267948966) q[20];
sdg q[5];
z q[18];
x q[20];
u3(0, 0, 1.5707963267948966) q[20];
u3(0, 0, 1.5707963267948966) q[4];
t q[2];
s q[22];
u2(1.5707963267948966, 1.5707963267948966) q[2];
t q[12];
u3(0, 0, 1.5707963267948966) q[22];
u3(0, 0, 1.5707963267948966) q[18];
rx(1.5707963267948966) q[3];
s q[0];
u3(0, 0, 1.5707963267948966) q[9];
rz(1.5707963267948966) q[18];
tdg q[10];
z q[13];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[2];
s q[6];
u1(1.5707963267948966) q[1];
rx(1.5707963267948966) q[2];
sdg q[8];
tdg q[15];
s q[10];
x q[18];
y q[4];
x q[15];
t q[6];
tdg q[12];
x q[17];
h q[18];
rx(1.5707963267948966) q[11];
rz(1.5707963267948966) q[21];
h q[21];
u3(0, 0, 1.5707963267948966) q[9];
x q[6];
s q[10];
z q[10];
x q[0];
u3(0, 0, 1.5707963267948966) q[6];
u3(0, 0, 1.5707963267948966) q[17];
rx(1.5707963267948966) q[12];
s q[17];
sdg q[8];
rz(1.5707963267948966) q[6];
y q[9];
y q[0];
h q[20];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[23];
z q[14];
y q[23];
s q[23];
tdg q[15];
u1(1.5707963267948966) q[18];
ry(1.5707963267948966) q[10];
u1(1.5707963267948966) q[23];
u3(0, 0, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[20];
tdg q[10];
t q[3];
s q[2];
s q[23];
